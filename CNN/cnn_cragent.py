import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models

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


class MaskMaker():
    def __init__(self):
        pass
    def getBoardMask(self, type):
        if type == "troop_n":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #enemy half 0
            mask[:15*18] = 0

            return mask
        elif type == "troop_r":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0
            
            #mask out left side
            mask[11*18:11*18+9] = 0
            mask[12*18:12*18+9] = 0
            mask[13*18:13*18+9] = 0
            mask[14*18:14*18+9] = 0
            return mask
        elif type == "troop_l":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0
            
            #mask out right side
            mask[11*18+9:11*18+18] = 0
            mask[12*18+9:12*18+18] = 0
            mask[13*18+9:13*18+18] = 0
            mask[14*18+9:14*18+18] = 0
            return mask
        elif type == "troop_b":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            #river in the middle
            mask[14*18:14*18+1] = 0
            mask[14*18+17:14*18+18] = 0
            mask[17*18:17*18+1] = 0
            mask[17*18+17:17*18+18] = 0

            mask[15*18:17*18]=0

            #left princess tower
            mask[24*18+2:24*18+5] = 0
            mask[25*18+2:25*18+5] = 0
            mask[26*18+2:26*18+5] = 0

            #right princess tower
            mask[24*18+13:24*18+16] = 0
            mask[25*18+13:25*18+16] = 0
            mask[26*18+13:26*18+16] = 0

            #center king tower
            mask[30*18+7:30*18+11]=0
            mask[29*18+7:29*18+11]=0
            mask[28*18+7:28*18+11]=0
            mask[27*18+7:27*18+11]=0

            #mask out top
            mask[0:11*18] = 0

            return mask
        elif type == "troop_noaction":
            mask = np.zeros(self.getBoardDim())

            mask[0] = 1
            return mask
        elif type == "spell":
            mask = np.ones(self.getBoardDim())

            #corner of the board
            mask[:6] = 0
            mask[12:18] = 0
            mask[31*18:31*18+6] = 0
            mask[31*18+12:31*18+18] = 0

            return mask

    def getDeckMask(self,arr):
        #arr is an array of all the cards that are in the deck and can or cannot be placed
        mask = np.ones(self.getCardDim()+1)
        for i in range(len(arr)):
            mask[i+1] = 0
    
    def getMasks(self):
        return [self.getBoardMask("troop_n"), self.getBoardMask("troop_r"), self.getBoardMask("troop_l"), self.getBoardMask("troop_b"), self.getBoardMask("troop_noaction"), self.getBoardMask("spell")]
    def getBoardDim(self):
        return 18*32
    
    def getActionDim(self):
        return 2
        
    def getCardDim(self):
        return 4

            
class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim, discrete = True, network_shape = [256, 256], epsilon = 0.2, high = None, low = None):
        super().__init__()
        if discrete:
                assert isinstance(action_dim, list)
                self.logits = nn.ModuleList([
                    buildMLP(observation_dim, act_dim, network_shape).to(device) for act_dim in action_dim
                ])

                # Collect all parameters for optimizer
                self.params = [param for head in self.logits for param in head.parameters()]
        else:
            self.mean = buildMLP(observation_dim, action_dim, network_shape).to(device)
            self.logstd = nn.Parameter(
                    torch.zeros(action_dim, dtype=torch.float32, device=device)            
                    )
            self.params = itertools.chain([self.logstd], self.mean.parameters())


        self.discrete = discrete
        self.epsilon = epsilon
        if high is not None:
            self.high = torch.from_numpy(high).float()
            self.low = torch.from_numpy(low).float()
        self.maskmaker = MaskMaker()
        self.board_masks = torch.tensor(self.maskmaker.getMasks(), dtype=torch.bool).to(device)
    
    @torch.no_grad()
    def get_action(self, observation, mask1 = None, mask2= None):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        assert observation.dim() == 1
        observation = observation.to(device)

        if isinstance(mask1, np.ndarray):
        # `mask` is just a single NumPy array
            mask1 = torch.from_numpy(mask1).bool().to(device)
        mask1.to(device)
        # elif isinstance(mask, list):
        #     # `mask` is a list of NumPy arrays
        #     mask = [torch.from_numpy(m).bool() for m in mask]
        if self.discrete:
            log_prob = 0
            sampled_action = []
            masked_logits = self.forward(observation=observation, mask = mask1)

      
            dis = torch.distributions.Categorical(logits=masked_logits[0])
            ac = dis.sample()
            log_prob += dis.log_prob(ac).cpu().item()
            sampled_action.append(ac.cpu().numpy())
            
            masked = masked_logits[1].clone()
            
            masked[~self.board_masks[mask2[ac.cpu().item()]]] = float('-inf')
            # print(masked)
            dis2 = torch.distributions.Categorical(logits=masked)
            ac2 = dis2.sample()
            log_prob += dis2.log_prob(ac2).cpu().item()
            sampled_action.append(ac2.cpu().numpy())

            sampled_action = np.array(sampled_action)
            return sampled_action, log_prob, mask1.cpu(), mask2[ac.cpu().item()] 
        else:
            mean = self.mean(observation)
            distribution = torch.distributions.Normal(loc=mean, scale=torch.exp(self.logstd))
            sampled_action = distribution.sample().cpu().numpy()
            log_prob = distribution.log_prob(sampled_action).sum(dim=-1).cpu().numpy()
        # if self.low is not None:
        #     sampled_action = torch.clamp(sampled_action, min=self.low, max=self.high)
        
        return sampled_action, log_prob
    
    def forward(self, observation, mask =None):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
    

        if self.discrete:
            logits = [head(observation) for head in self.logits]  # List of tensors
            masked = []

            masked_logits1 = logits[0]
            masked_logits1[~mask] = float('-inf')

            masked_logits2 = logits[1]
            masked.append(masked_logits1)
            masked.append(masked_logits2)
            return masked
            # for ind, logit in enumerate(logits):
            #     # Apply mask if provided
            #     if mask is not None and len(mask) > ind:
            #         # Create a large negative value to effectively zero out masked logits
            #         masked_logits = logit.clone()
            #         masked_logits[~mask[ind]] = float('-inf')
            #     else:
            #         masked_logits = logit
            #     # Create distribution with masked logits
            #     masked.append(masked_logits)
            #     # distribution.append(torch.distributions.Categorical(logits=masked_logits))
            # return masked
        else:
            mean = self.mean(observation)
            # print("mean shape", mean.shape)
            distribution = torch.distributions.Normal(loc = mean, scale = torch.exp(self.logstd))
        
        return distribution
    def forward_no_mask(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
    

        if self.discrete:
            logits = [head(observation) for head in self.logits]
            return logits



class ValueNetwork(nn.Module):

    def __init__(self, observation_dim,  network_shape = [256, 256], tau = 1.0, target = False, update_target_step = 500):
        super().__init__()
        self.use_target = target
        self.tau = tau
        self.model = buildMLP(observation_dim, 1, network_shape).to(device)
        if self.use_target:
            self.target = buildMLP(observation_dim, 1, network_shape).to(device)
            self.updateTarget(self.tau)
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

        # with torch.no_grad():
        #     if self.use_target:
        #         value_target = self.target(observations) + advantages
        #     else:
        #         value_target = self.model(observations) + advantages

        with torch.no_grad():
            value_target = (1-done) * self.gamma * self.model(next_observation.detach()) + reward
        
        loss = self.loss_fn(value_pred, value_target)

        return loss


class CRPPOAgent(nn.Module):

    def __init__(self, observation_dim, action_dim, discrete = True, lr = 3e-4, gamma=0.99, lamb = 0.95, epsilon = 0.2, tau=1.0, value_network_shape = [1024, 1024, 512], high=None, low= None, temp=0.01):
        super().__init__()
        self.hidden = 512
        self.value = ValueNetwork(observation_dim=observation_dim + self.hidden,  network_shape=value_network_shape)
        self.actor = Actor(observation_dim=observation_dim + self.hidden,  action_dim=action_dim, discrete=discrete, high = high, low= low)
        self.feature_extractor = models.resnet18(pretrained=True).to(device)
        self.feature_extractor.fc = nn.Identity()

        self.optimizer = optim.Adam(
            itertools.chain(self.feature_extractor.parameters(), self.value.parameters(), self.actor.params),
            lr=lr
        )

        self.tau = tau
        self.lamb = lamb
        self.gamma = gamma
        self.high = high
        self.low = low
        self.num_val = 10
        self.discrete = discrete

        self.epsilon = epsilon

        self.prev_prob = None
        self.maskmaker = MaskMaker()
        self.board_masks = torch.tensor(self.maskmaker.getMasks(), dtype=torch.bool).to(device)
        self.temp = temp
    

    def calculate_advantage(self, images, next_images, observation, next_observation, rewards, dones):
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        if isinstance(next_observation, np.ndarray):
            next_observation = torch.from_numpy(next_observation).float()
        next_observation = next_observation.to(device)
        images = torch.from_numpy(images).float().to(device)
        next_images = torch.from_numpy(next_images).float().to(device)
        # print(observation)
        features = self.feature_extractor(images)
        observation = torch.cat([features, observation], dim=-1)

        next_features = self.feature_extractor(next_images)
        next_observation = torch.cat([next_features, next_observation], dim=-1)

        size = observation.shape[0]
        

        advantages = np.zeros(size+1)
        
        for i in reversed(range(size)):
            if dones[i]:
                delta = rewards[i] - self.value.forward(observation[i]).detach().cpu().numpy().squeeze()
            else:
                delta = rewards[i] + self.gamma*self.value.forward(next_observation[i]).detach().cpu().numpy().squeeze() - self.value.forward(observation[i]).detach().cpu().numpy().squeeze()

            advantages[i] = delta + self.gamma*self.lamb * advantages[i+1]

        advantages = advantages[:-1]
        return advantages

    def get_action(self, image, observation, mask1=None, mask2 = None):
        assert image.shape == (3, 224, 224), f"Image shape is {image.shape}"
        image = torch.from_numpy(image).float().to(device)
        image = image.unsqueeze(0)
        features = self.feature_extractor(image).squeeze() #squeezing it changes it from a vector of shape [1,512] to [512]
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        observation = torch.cat([features, observation], dim=-1)
        return self.actor.get_action(observation, mask1=mask1, mask2=mask2)
    
    def update_actor(self, observation, actions, advantages, old_log_probs, mask1, mask2):
        # mask is given by dim (2, batch, mask_dim)
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        observation = observation.to(device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions = actions.to(device)
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float()
        advantages = advantages.to(device)
        if isinstance(old_log_probs, np.ndarray):
            old_log_probs = torch.from_numpy(old_log_probs).float()
        old_log_probs = old_log_probs.to(device)

        

        mask1 = mask1.to(device)

        batch_size = observation.shape[0]

        if actions.dim() != 2:
            actions = actions.unsqueeze(1)

        if self.discrete:
            logits = self.actor.forward_no_mask(observation)
            log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
            
            for i in range(batch_size):
                masked_logits1 = logits[0][i].clone()
                masked_logits1[~mask1[i]] = float('-inf')

                masked_logits2 = logits[1][i].clone()
                masked_logits2[~self.board_masks[mask2[i]]] = float('-inf')
                dist1 = torch.distributions.Categorical(logits=masked_logits1)
                dist2 = torch.distributions.Categorical(logits=masked_logits2)
                
                entropy = dist1.entropy() + dist2.entropy()
                log_probs[i] += dist1.log_prob(actions[i,0])
                log_probs[i] += dist2.log_prob(actions[i,1])
        else:
            dist = self.actor.forward(observation)
            log_probs = dist.log_prob(actions).sum(-1)
        print(len(log_probs), len(old_log_probs))
        r = (log_probs - old_log_probs).exp()

        clipped = torch.clamp(r, min = 1.0 - self.epsilon, max=1.0 + self.epsilon)

        clipped_obj = clipped * advantages
        unclipped_obj = r * advantages

        min_obj = torch.min(clipped_obj, unclipped_obj) + self.temp*entropy
        loss = -min_obj.mean()

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
        images = sample["images"]
        next_images = sample["next_images"]
        observations = sample["observations"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        next_observations = sample["next_observations"]
        dones = sample["dones"]
        advantages = sample["advantages"]
        log_probs = sample["log_probs"]
        mask1 = sample["mask1"]
        mask2 = sample["mask2"]
        batch_size = observations.shape[0]
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
        images = torch.from_numpy(images).float().to(device)
        next_images = torch.from_numpy(next_images).float().to(device)

        assert images.shape == (batch_size, 3, 224, 224), f"Images shape is {images.shape}"
        assert next_images.shape == (batch_size, 3, 224, 224), f"Next images shape is {next_images.shape}"

        features = self.feature_extractor(images)
        observations = torch.cat([features, observations], dim=-1)

        next_features = self.feature_extractor(next_images)
        next_observations = torch.cat([next_features, next_observations], dim=-1)

        # for a in range(self.num_val):
        val_loss = self.value.update(observations, advantages, next_observations, rewards, dones)

        actor_loss = self.update_actor(observations,actions, advantages, log_probs, mask1=mask1, mask2=mask2)

        total_loss = actor_loss + val_loss 

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        print(val_loss, actor_loss)
        return val_loss, actor_loss
    

    def save_model(self, file_path: str = "model.pth"):
        """
        Saves the entire model's state dictionary to the given file path.
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str = "model.pth"):
        """
        Loads the entire model's state dictionary from the given file path.
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint)
        self.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")

    
        