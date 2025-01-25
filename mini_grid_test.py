import numpy as np
import torch
import gym
from agents import *  # Ensure your agent classes are correctly imported
from ReplayBuffer import ReplayBuffer
import minigrid  # Correct import for the new package

# Set device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# Helper function to preprocess MiniGrid observations
def preprocess_obs(obs):
    """
    Flatten the 'image' and concatenate the 'direction' to form a single flat vector.
    
    Args:
        obs (dict): Observation from MiniGrid environment.
    
    Returns:
        np.ndarray: Flattened observation vector.
    """
    image = obs['image']  # Shape: (7, 7, 3) by default
    direction = obs['direction']  # Scalar
    flattened_image = image.flatten()  # Shape: (147,)
    return np.concatenate([flattened_image, np.array([direction], dtype=np.float32)])  # Shape: (148,)

# Define the main training loop
def train_agent(env_name='MiniGrid-Empty-8x8-v0', num_episodes=1500, max_steps=1000, batch_size=32, buffer_capacity=64):
    env = gym.make(env_name)
    initial_reset = env.reset()
    initial_obs, _ = initial_reset  # Unpack the tuple
    obs_processed = preprocess_obs(initial_obs)
    observation_dim = obs_processed.shape[0]
    action_space = env.action_space
    print("Finished making environment")

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
    action_rep_dim = 1  # Adjust if actions have higher dimensions

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
        initial_reset = env.reset()
        initial_obs, _ = initial_reset  # Unpack the tuple
        obs = preprocess_obs(initial_obs)
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
            # Execute action in the environment
            next_obs, reward, done, truncated, _ = env.step(action[0])
            done = done or truncated
            total_reward += reward

            # Preprocess next observation
            next_obs_processed = preprocess_obs(next_obs)

            # Store data in lists
            observations.append(obs)
            actions.append(action)
            rewards.append([reward])  # Shape (1,)
            next_observations.append(next_obs_processed)
            dones.append([float(done)])  # Shape (1,)
            log_probs.append([log_prob])
            prev_actions.append(prev_action if prev_action is not None else np.zeros(action_rep_dim, dtype=np.float32))
            
            # Keep hn and cn as tensors and concatenate
            hn = hn.detach()
            cn = cn.detach()
            if hn_states is None:
                hn_states = hn  # Shape: (num_layers, 1, hidden_size)
                cn_states = cn
            else:
                hn_states = torch.cat((hn_states, hn), dim=1)  # Concatenate along the sequence dimension
                cn_states = torch.cat((cn_states, cn), dim=1)
            hidden = (hn, cn)

            # Prepare for next step
            obs = next_obs_processed
            prev_action = action

            # Periodically add to replay buffer
            if step % 32 == 0 and hn_states is not None:
                observations_np = np.array(observations, dtype=np.float32)
                actions_np = np.array(actions, dtype=np.int64)
                rewards_np = np.array(rewards, dtype=np.float32)
                next_observations_np = np.array(next_observations, dtype=np.float32)
                dones_np = np.array(dones, dtype=np.bool_)
                log_probs_np = np.array(log_probs, dtype=np.float32)
                prev_actions_np = np.array(prev_actions, dtype=np.int64)

                advantages_np = agent.calculate_advantage(
                    observations_np,
                    next_observations_np,
                    rewards_np,
                    dones_np,
                    prev_actions_np,
                    hn_states[:, 0, :].unsqueeze(1),  # Shape: (num_layers, 1, hidden_size)
                    cn_states[:, 0, :].unsqueeze(1),
                    actions_np
                )
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
                # Clear trajectory lists
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

            advantages_np = agent.calculate_advantage(
                observations_np,
                next_observations_np,
                rewards_np,
                dones_np,
                prev_actions_np,
                hn_states[:, 0, :].unsqueeze(1),
                cn_states[:, 0, :].unsqueeze(1),
                actions_np
            )
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
    print("Starting training...")
    train_agent()
