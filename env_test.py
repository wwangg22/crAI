import numpy as np
import torch
import gym
from agents import *
from ReplayBuffer import ReplayBuffer

# Assuming all your classes (buildMLP, LSTMFeatureExtractor, ValueNetwork, Actor, PPOLSTMAgent) are already defined above.
# Also, ensure that the corrected ReplayBuffer class is defined as above.

# Set device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# Define the main training loop
def train_agent(env_name='CartPole-v1', num_episodes=1500, max_steps=1000, batch_size=32, buffer_capacity=100):
    env = gym.make(env_name)
    observation_dim = env.observation_space.shape[0]
    action_space = env.action_space
    print("finished making env")

    # Determine if action space is discrete or continuous
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = [action_space.n]
        discrete = True
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape
        discrete = False
    else:
        raise NotImplementedError("Unsupported action space type.")

    # Action representation dimension (for LSTM input)
    action_rep_dim = 1

    # Define hidden state shape
    lstm_layers = 1
    hidden_size = 16
    hidden_shape = (lstm_layers, hidden_size)

    # Initialize Agent
    agent = PPOLSTMAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_rep_dim=action_rep_dim,
        latent_dim=hidden_size,
        lstm_layers=lstm_layers,
        discrete=discrete,
        lr=3e-4,
        gamma=0.99,
        lamb=0.95,
        epsilon=0.2,
        tau=1.0,
        value_network_shape=[256, 256],
        high=action_space.high if not discrete else None,
        low=action_space.low if not discrete else None
    ).to(device)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        size=buffer_capacity
    )

    # Initialize hidden states
    hidden = None

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        hidden = None  # Reset hidden state at the start of each episode
        prev_action = None

        # Lists to store trajectory data
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        log_probs = []
        prev_actions = []
        hn_states = None
        cn_states = None

        while not done and step < max_steps:
            # Get action and log probability from the agent
            (action, log_prob), (hn, cn) = agent.get_action(obs, prev_action, hidden)
            # print(action)
            next_obs, reward, done, truncated, _ = env.step(action[0])
            done = done or truncated
            total_reward += reward

            # Store data in lists
            observations.append(obs)
            actions.append(action)
            rewards.append([reward])  # Shape (1,)
            next_observations.append(next_obs)
            dones.append([float(done)])  # Shape (1,)
            log_probs.append([log_prob])
            prev_actions.append(prev_action if prev_action is not None else np.zeros(action_rep_dim, dtype=np.float32))
            hn = hn.detach()
            cn = cn.detach()
            if hn_states is None:
                hn_states = hn
                cn_states = cn
            else:
                hn_states = torch.cat((hn_states, hn), dim=1)
                cn_states = torch.cat((cn_states, cn), dim =1)
            hidden = (hn, cn)

            

            # Prepare for next step
            obs = next_obs
            prev_action = action

            if step % 32 ==0 and hn_states is not None:
                observations_np = np.array(observations, dtype=np.float32)
                actions_np = np.array(actions, dtype=np.int64)
                rewards_np = np.array(rewards, dtype=np.float32)
                next_observations_np = np.array(next_observations, dtype=np.float32)
                dones_np = np.array(dones, dtype=np.bool_)
                log_probs_np = np.array(log_probs, dtype=np.float32)
                prev_actions_np = np.array(prev_actions, dtype=np.int64)

                advantages_np = agent.calculate_advantage(observations_np, next_observations_np, rewards_np, dones_np, prev_actions_np, hn_states[:, 0, :].unsqueeze(1), cn_states[:, 0, :].unsqueeze(1), actions_np)
                replay_buffer.add_sequence(
                    observation=observations_np,
                    next_observation=next_observations_np,
                    action=actions_np,
                    reward=rewards_np,
                    done=dones_np,
                    advantage=advantages_np,
                    log_probs=log_probs_np,
                    hn=hn_states,
                    cn=cn_states,
                    prev_action=prev_actions_np
                )
                observations = []
                actions = []
                rewards = []
                next_observations = []
                dones = []
                log_probs = []
                prev_actions = []
                hn_states = None
                cn_states = None
            step += 1



        print(total_reward)
        if step != 1 and hn_states is not None:
            observations_np = np.array(observations, dtype=np.float32)
            actions_np = np.array(actions, dtype=np.int64)
            rewards_np = np.array(rewards, dtype=np.float32)
            next_observations_np = np.array(next_observations, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.bool_)
            log_probs_np = np.array(log_probs, dtype=np.float32)
            prev_actions_np = np.array(prev_actions, dtype=np.int64)

            advantages_np = agent.calculate_advantage(observations_np, next_observations_np, rewards_np, dones_np, prev_actions_np, hn_states[:, 0, :].unsqueeze(1), cn_states[:, 0, :].unsqueeze(1), actions_np)
            replay_buffer.add_sequence(
                observation=observations_np,
                next_observation=next_observations_np,
                action=actions_np,
                reward=rewards_np,
                done=dones_np,
                advantage=advantages_np,
                log_probs=log_probs_np,
                hn=hn_states,
                cn=cn_states,
                prev_action=prev_actions_np
            )
        # After the episode, calculate advantages
        # For simplicity, we'll use the Temporal Difference (TD) residuals as advantages
        # In practice, you might want to use Generalized Advantage Estimation (GAE)

        # Convert lists to NumPy arrays
        

        # Add the sequence to the replay buffer
        


        # Calculate advantages
        # with torch.no_grad():
        #     # Get value predictions for current and next observations
        #     # Pass through the LSTM to get latent states
        #     obs_tensor = torch.from_numpy(observations_np).float().to(device)
        #     actions_tensor = torch.from_numpy(actions_np).float().to(device)
        #     prev_actions_tensor = torch.from_numpy(prev_actions_np).float().to(device)

        #     # Concatenate observations and previous actions
        #     obs_ac = torch.cat((obs_tensor, prev_actions_tensor), dim=-1)  # Shape: (seq_len, obs_dim + action_dim)
        #     obs_ac = obs_ac.unsqueeze(0)  # Add batch dimension: (1, seq_len, obs_dim + action_dim)

        #     # Initialize hidden states
        #     hn_initial = torch.from_numpy(hn_np[0]).float().unsqueeze(1).to(device)  # Shape: (num_layers, 1, hidden_size)
        #     cn_initial = torch.from_numpy(cn_np[0]).float().unsqueeze(1).to(device)  # Shape: (num_layers, 1, hidden_size)

        #     # Pass through LSTM
        #     latent_states, (hn_final, cn_final) = agent.lstm(obs_ac, (hn_initial, cn_initial))
        #     latent_states = latent_states.squeeze(0)  # Shape: (seq_len, hidden_size)

        #     # Get value predictions
        #     value_preds = agent.value(latent_states).squeeze(-1).cpu().numpy()  # Shape: (seq_len,)
        #     next_value_preds = agent.value(latent_states).squeeze(-1).cpu().numpy()  # Placeholder for next value

        #     # Calculate advantages using TD residuals
        #     advantages = np.zeros_like(rewards_np)
        #     last_advantage = 0
        #     for t in reversed(range(len(rewards_np))):
        #         if dones_np[t]:
        #             last_advantage = rewards_np[t][0] - value_preds[t]
        #         else:
        #             last_advantage = rewards_np[t][0] + agent.gamma * value_preds[t + 1] - value_preds[t] if t + 1 < len(rewards_np) else rewards_np[t][0] - value_preds[t]
        #         advantages[t][0] = last_advantage

        # # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        # Perform learning step if enough samples are available
        if len(replay_buffer) >= batch_size:
            try:
                sample = replay_buffer.sample(batch_size)
                # Update the agent
                val_loss, actor_loss = agent.learn(sample)
                print(f"Training Update - Value Loss: {val_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
            except Exception as e:
                print(f"Training update failed: {e}")

if __name__ == "__main__":
    print("hello")
    train_agent()
