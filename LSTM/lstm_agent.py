
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


cuda_available = torch.cuda.is_available()

device = torch.device("cuda" if cuda_available else "cpu")

       
def buildMLP(input_dim, output_dim, network_shape):
    layers = []
    in_size = input_dim
    for a in network_shape:
        layers.append(nn.Linear(in_size, a))
        layers.append(nn.ReLU())
        in_size = a
    layers.append(nn.Linear(in_size, output_dim))

    return nn.Sequential(*layers)


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size=100, lstm_size = 256, hidden_size=256, num_layers=1, device='cpu'):
        """
        Initializes the LSTMFeatureExtractor.

        Args:
            input_size (int): Dimensionality of each input observation vector.
            hidden_size (int): Dimensionality of the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            device (str): Device to run the LSTM on ('cpu' or 'cuda').
        """
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # Define the LSTM layer with batch_first=True to accept (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # self.linear = nn.Linear(lstm_size, hidden_size)
        
    def forward(self, x, hx=None):
        """
        Forward pass through the LSTM.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, input_size).
            hx (tuple, optional): Tuple of (h_0, c_0) for hidden and cell states.
                Each should be of shape (num_layers, batch_size, hidden_size).
                Defaults to None, which initializes them to zeros.

        Returns:
            output (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_size),
                                   containing hidden states for all timesteps.
            (h_n, c_n) (tuple): Updated hidden and cell states.
        """
        # x is already of shape (batch_size, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x, hx)

        # Apply linear layer to the output of the LSTM
        # output = F.relu(self.linear(output))
        
        return output, (h_n, c_n)



class ValueNetwork(nn.Module):

    def __init__(self, observation_dim, lr = 3e-4, network_shape = [256, 256], tau = 1.0, target = False, update_target_step = 500):
        super().__init__()
        self.use_target = target
        self.tau = tau
        self.model = buildMLP(observation_dim, 1, network_shape).to(device)
        if self.use_target:
            self.target = buildMLP(observation_dim, 1, network_shape).to(device)
            self.updateTarget(self.tau)
        # self.optimizer = optim.Adam(
        #     self.model.parameters(),
        #     lr = lr
        # )
        self.loss_fn = nn.MSELoss()
        self.step = 0
        self.gamma = 0.999
        self.target_update = update_target_step

    def updateTarget(self, tau=None):
        if tau is None:
            tau = self.tau
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
        )


    def forward(self, observation) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        observation = observation.to(device)
        
        assert isinstance(observation, torch.Tensor) == True

        pred = self.model(observation)

        return pred
    
    def update(self, observations, advantages, next_observation, reward, done):

        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        advantages= advantages.unsqueeze(1)


        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward).float()
        reward = reward.to(device)
        # print(advantages.shape)
        # print(observations.shape)

        done = torch.from_numpy(done).float().to(device)

        value_pred = self.model(observations)

        with torch.no_grad():
            # if self.use_target:
            #     value_target = self.target(observations.detach()) + advantages
            # else:
            #     value_target = self.model(observations.detach()) + advantages
            if self.use_target:
                value_target = reward + (1 - done) * self.gamma * self.target(next_observation.detach())
            else:
                value_target = reward + (1 - done) * self.gamma * self.model(next_observation.detach())

        # with torch.no_grad():
        #     value_target = (1-done) * self.gamma * self.model(next_observation) + reward
        # print('val target', value_target.shape)
        # print('val pred', value_pred.shape)
        value_pred = value_pred.squeeze(1)
        value_target = value_target.squeeze(1)
        # print(value_pred[:10], value_target[:10])
        loss = self.loss_fn(value_pred, value_target)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # if self.use_target:
        #     if self.step % self.target_update == 0:
        #         self.updateTarget()
        # self.step+=1

        return loss


class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim, lr=3e-4, discrete = True, network_shape = [256, 256], epsilon = 0.2, high = None, low = None):
        super().__init__()
        #for discrete cases, action_dim will be passed as an array [x, x1, x2], with x being then # of discrete choices along the first dim, x1 being the # of discrete choices along the second dim
        if discrete:
            assert isinstance(action_dim, list)
            self.logits = nn.ModuleList([
                buildMLP(observation_dim, act_dim, network_shape) for act_dim in action_dim
            ])

            # Collect all parameters for optimizer
            self.params = [param for head in self.logits for param in head.parameters()]
        else:
            self.mean = buildMLP(observation_dim, action_dim, network_shape).to(device)
            self.logstd = nn.Parameter(
                    torch.zeros(action_dim, dtype=torch.float32, device=device)            
                    )
            self.params = itertools.chain([self.logstd], self.mean.parameters())

        # self.optimizer = optim.Adam(
        #     params,
        #     lr
        # )

        self.discrete = discrete
        self.epsilon = epsilon
        if high is not None:
            self.high = torch.from_numpy(high).float()
            self.low = torch.from_numpy(low).float()
    
    @torch.no_grad()
    def get_action(self, observation, mask=None):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        assert observation.dim() == 1
        observation = observation.to(device)
        if self.discrete:
            log_prob = 0
            sampled_action = []
            distribution = self.forward(observation=observation, mask=mask)
            for dis in distribution:
                ac = dis.sample()
                log_prob += dis.log_prob(ac).cpu().item()
                sampled_action.append(ac.cpu().numpy())
            sampled_action = np.array(sampled_action)
        else:
            mean = self.mean(observation)
            distribution = torch.distributions.Normal(loc=mean, scale=torch.exp(self.logstd))
            sampled_action = distribution.sample().cpu().numpy()
            log_prob = distribution.log_prob(sampled_action).sum(dim=-1).cpu().numpy()
        # if self.low is not None:
        #     sampled_action = torch.clamp(sampled_action, min=self.low, max=self.high)
        
        return sampled_action, log_prob
    
    def forward(self, observation, mask=None):
        """
        for the discrete case, this function return a LIST of torch distributions
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool()

        if self.discrete:
            logits = [head(observation) for head in self.logits]  # List of tensors
            distribution = []
            for ind, logit in enumerate(logits):
                # Apply mask if provided
                if mask is not None:
                    # Create a large negative value to effectively zero out masked logits
                    masked_logits = logit.clone()
                    masked_logits[~mask[ind]] = float('-inf')
                else:
                    masked_logits = logit
                # Create distribution with masked logits
                distribution.append(torch.distributions.Categorical(logits=masked_logits))
        else:
            mean = self.mean(observation)
            # print("mean shape", mean.shape)
            distribution = torch.distributions.Normal(loc = mean, scale = torch.exp(self.logstd))
        
        return distribution


class PPOLSTMAgent(nn.Module):

    def __init__(self, observation_dim, action_dim, action_rep_dim, lstm_dim = 256, latent_dim = 256, lstm_layers = 1, discrete = True, lr = 3e-4, gamma=0.99, lamb = 0.95, epsilon = 0.2, tau=1.0, value_network_shape = [256, 256], high=None, low= None):
        super().__init__()
        self.value = ValueNetwork(observation_dim=latent_dim, lr=lr, network_shape=value_network_shape)
        self.actor = Actor(observation_dim=latent_dim, lr=lr, action_dim=action_dim, discrete=discrete, high = high, low= low)
        self.lstm = LSTMFeatureExtractor(input_size=observation_dim + action_rep_dim, lstm_size= lstm_dim, hidden_size=latent_dim, num_layers=lstm_layers)

        self.optimizer = optim.Adam(
            itertools.chain(self.lstm.parameters(), self.value.parameters(), self.actor.params),
            lr=lr
        )
        self.tau = tau
        self.lamb = lamb
        self.gamma = gamma
        self.high = high
        self.low = low
        self.num_val = 10
        self.discrete = discrete
        self.action_rep_dim = action_rep_dim
        self.obs_dim = observation_dim
        self.ac_dim = action_dim
        self.latent_dim = latent_dim

        self.epsilon = epsilon

        self.prev_prob = None

    def calculate_advantage(self, observation, next_observation, rewards, dones, prev_action, hn_t, cn_t, cur_action):
        """
        hn_t is the hidden state of the FIRST observation in the sequence
        cn_t too
        """
        observation = torch.tensor(observation, dtype=torch.float32).to(device) if isinstance(observation, np.ndarray) else observation.to(device)
        next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device) if isinstance(next_observation, np.ndarray) else next_observation.to(device)
        cur_action = torch.tensor(cur_action, dtype=torch.float32).to(device) if isinstance(cur_action, np.ndarray) else cur_action.to(device)
        
        if prev_action is None:
            prev_action = torch.zeros((self.action_rep_dim,), dtype=torch.float32, device=device)
        else:
            prev_action = torch.tensor(prev_action, dtype=torch.float32).to(device) if isinstance(prev_action, np.ndarray) else prev_action.to(device)

        size = observation.shape[0]  # Assuming seq_len

        # Append the last observation and action
        last_next_observation = next_observation[-1].unsqueeze(0)  # Shape: (1, feature_dim)
        new_observation = torch.cat((observation, last_next_observation), dim=0)  # Shape: (seq_len +1, feature_dim)

        last_cur_action = cur_action[-1].unsqueeze(0)  # Shape: (1, action_dim)
        new_actions = torch.cat((prev_action, last_cur_action), dim=0)  # Shape: (seq_len +1, action_dim)

        # Concatenate observations and actions along the feature dimension
        obs_ac = torch.cat((new_observation, new_actions), dim=-1)  # Shape: (seq_len +1, feature_dim + action_dim)
        hn_t = hn_t.to(device)
        cn_t = cn_t.to(device)
        # Add batch dimension if necessary
        obs_ac = obs_ac.unsqueeze(0)  # Shape: ( seq_len +1, 1, feature_dim + action_dim)
        assert obs_ac.shape == ( 1, size + 1, self.obs_dim + self.action_rep_dim), f"Expected shape (1, {size +1}, {self.obs_dim + self.action_rep_dim}), but got {obs_ac.shape}"


        output, (_, _) = self.lstm(obs_ac, (hn_t, cn_t))

        latent_states = output.squeeze(0)
        

        advantages = np.zeros(size+1)
        
        for i in reversed(range(size)):
            if dones[i]:
                delta = rewards[i] - self.value.forward(latent_states[i]).detach().cpu().numpy().squeeze()
            else:
                delta = rewards[i] + self.gamma*self.value.forward(latent_states[i+1]).detach().cpu().numpy().squeeze() - self.value.forward(latent_states[i]).detach().cpu().numpy().squeeze()

            advantages[i] = delta + self.gamma*self.lamb * advantages[i+1]

        advantages = advantages[:-1]
        # print(advantages)
        return advantages
    
    def get_action(self, observation, prev_action = None, hx = None):
        
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        if prev_action is None:
            prev_action = torch.zeros((self.action_rep_dim,))
        if isinstance(prev_action, np.ndarray):
            prev_action = torch.from_numpy(prev_action).float()
        prev_action = prev_action.to(device)
        
        obs_ac = torch.cat((observation, prev_action))
        obs_ac = obs_ac.unsqueeze(0)
        obs_ac = obs_ac.unsqueeze(0)
        # print(self.action_rep_dim)
        assert obs_ac.shape == (1, 1, self.obs_dim + self.action_rep_dim), f"Expected shape (1, 1, {self.obs_dim + self.action_rep_dim}), but got {obs_ac.shape}"
        if hx is not None:
            (hn, cn) = hx
            hn= hn.to(device)
            cn= cn.to(device)
            latent_state, (h_n, c_n) = self.lstm(obs_ac, (hn,cn))
        else:
            latent_state, (h_n, c_n) = self.lstm(obs_ac, None)
        h_n = h_n.detach().cpu()
        c_n = c_n.detach().cpu()

        latent_state=latent_state.squeeze(0)
        latent_state=latent_state.squeeze(0)
        assert latent_state.shape == (self.latent_dim,)

        return self.actor.get_action(latent_state), (h_n, c_n)
    
    def update_actor(self, latent_state, actions, advantages, old_log_probs, batch_size):
       
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        if isinstance(old_log_probs, np.ndarray):
            old_log_probs = torch.from_numpy(old_log_probs).float()
        old_log_probs = old_log_probs.to(device)
        # h_nt = h_nt.to(device)
        # c_nt = c_nt.to(device)
        # batch_size = len(actions)

        # obs_ac = torch.cat((observation, prev_actions), dim=-1)
        # obs_ac = obs_ac.unsqueeze(1)
        # assert obs_ac.shape == (batch_size, 1, self.obs_dim + self.action_rep_dim)
        # hnt = torch.cat(h_nt, dim=1)
        # cnt = torch.cat(c_nt, dim=1)


        dist = self.actor.forward(latent_state)
        
        if self.discrete:
            log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
            for i in range(len(dist)):
                log_probs += dist[i].log_prob(actions[:,i])
        else:
            log_probs = dist.log_prob(actions).sum(-1)

        old_log_probs = old_log_probs.squeeze(1)
        # print("log_probs shape", log_probs.shape)
        # print("old log probs shape ", old_log_probs.shape)
        r = (log_probs - old_log_probs).exp()

        clipped = torch.clamp(r, min = 1.0 - self.epsilon, max=1.0 + self.epsilon)

        clipped_obj = clipped * advantages
        unclipped_obj = r * advantages

        min_obj = torch.min(clipped_obj, unclipped_obj)
        loss = -min_obj.mean()

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return loss

    def learn(self, sample):
        """
        passed as 
         "observations": self.observation[rand_indices],
            "actions": self.action[rand_indices],
            "rewards": self.reward[rand_indices],
            "next_observations": self.next_observation[rand_indices],
            "dones": self.done[rand_indices],
            "advantages": self.advantage[rand_indices],
            "log_probs": self.log_probs[rand_indices],
        """
        observations = sample["observations"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        next_observations = sample["next_observations"]
        dones = sample["dones"]
        advantages = sample["advantages"]
        log_probs = sample["log_probs"]
        prev_actions = sample["prev_actions"]
        hnt = sample["hn"]
        cnt = sample["cn"]

        batch_size = len(actions)
        # print(advasntages)

        # hnt = torch.cat(hn_t, dim=1)
        # cnt = torch.cat(cn_t, dim=1)
    
        hnt = hnt.to(device)
        cnt = cnt.to(device)
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(device)
        if isinstance(next_observations, np.ndarray):
            next_observations = torch.from_numpy(next_observations).float()
        next_observations = next_observations.to(device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions = actions.to(device)
        if isinstance(prev_actions, np.ndarray):
            prev_actions = torch.from_numpy(prev_actions).float()
        prev_actions = prev_actions.to(device)

        obs_ac = torch.cat((observations, prev_actions), dim=-1)
        obs_ac = obs_ac.unsqueeze(1)
        assert obs_ac.shape == (batch_size, 1, self.obs_dim + self.action_rep_dim)

        latent_state, (hn1, cn1) = self.lstm(obs_ac, (hnt, cnt))

        latent_state = latent_state.squeeze(1)

        assert latent_state.shape == (batch_size, self.latent_dim)

        hn1 = hn1.detach()
        cn1 = cn1.detach()

        next_obs_ac = torch.cat((next_observations, actions), dim =-1 )
        next_obs_ac = next_obs_ac.unsqueeze(1)
        assert next_obs_ac.shape == (batch_size, 1, self.obs_dim + self.action_rep_dim)

        next_latent_state, (_, _) = self.lstm(next_obs_ac, (hn1, cn1))
        next_latent_state = next_latent_state.squeeze(1)

        # for a in range(self.num_val):
        val_loss = self.value.update(latent_state, advantages, next_latent_state, rewards, dones)
        

        actor_loss = self.update_actor(latent_state,actions, advantages, log_probs, batch_size)


        total_loss = actor_loss + val_loss
        self.optimizer.zero_grad()
        
        total_loss.backward()
        # for name, param in self.value.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.abs().mean().item():.6f}")
        #     else:
        #         print(f"Parameter {name} has no gradient.")

        self.optimizer.step()

        return val_loss, actor_loss
    


class ValueNetwork(nn.Module):

    def __init__(self, observation_dim, lr = 3e-4, network_shape = [256, 256], tau = 1.0, target = False, update_target_step = 500):
        super().__init__()
        self.use_target = target
        self.tau = tau
        self.model = buildMLP(observation_dim, 1, network_shape).to(device)
        if self.use_target:
            self.target = buildMLP(observation_dim, 1, network_shape).to(device)
            self.updateTarget(self.tau)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = lr
        )
        self.loss_fn = nn.MSELoss()
        self.step = 0
        self.gamma = 0.9
        self.target_update = update_target_step

    def updateTarget(self, tau):
        for param, target_param in zip(self.model.parameters(), self.target.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


    def forward(self, observation) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        
        observation = observation.to(device)
        
        assert isinstance(observation, torch.Tensor) == True

        pred = self.model(observation)

        return pred
    
    def update(self, observations, advantages, next_observation, reward, done):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        observations = observations.to(device)

        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)

        if isinstance(next_observation, np.ndarray):
            next_observation = torch.from_numpy(next_observation).float()
        next_observation = next_observation.to(device)

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward).float()
        reward = reward.to(device)

        done = torch.from_numpy(done).float().to(device)

        value_pred = self.model(observations)

        with torch.no_grad():
            if self.use_target:
                value_target = self.target(observations) + advantages
            else:
                value_target = self.model(observations) + advantages

        # with torch.no_grad():
        #     value_target = (1-done) * self.gamma * self.model(next_observation) + reward
        
        loss = self.loss_fn(value_pred, value_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_target:
            if self.step % self.target_update == 0:
                self.updateTarget()
        self.step+=1

        return loss
