import numpy as np
import torch
import gym
import random  # Import random module
from agents import *
from ReplayBuffer import ReplayBuffer

# Set device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
train_per_epoch = 10

# Define the main training loop
def train_agent(env_name='CartPole-v1', num_episodes=100000, max_steps=500, batch_size=64, buffer_capacity=200):
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
    lstm_size = 128
    hidden_size = 16
    hidden_shape = (lstm_layers, hidden_size)

    # Initialize Agent
    agent = PPOLSTMAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_rep_dim=action_rep_dim,
        lstm_dim=lstm_size,
        latent_dim=hidden_size,
        lstm_layers=lstm_layers,
        discrete=discrete,
        lr=3e-5,
        gamma=0.9,
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
    reward_history=[]
    total_step = 0


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
            # Introduce noise: with 50% probability, set a random index in obs to 0

            # Get action and log probability from the agent using the potentially noisy observation
            (action, log_prob), (hn, cn) = agent.get_action(obs, prev_action, hidden)

            next_obs, reward, done, truncated, _ = env.step(action[0])

            # if random.random() < 0.8:

                # noisy_obs = next_observation.copy()
            next_obs[2] = 0
            next_obs[3] = 0
            # noisy_obs = np.zeros_like(next_obs)
            # random_index = random.randint(0, (len(noisy_obs) - 1))
            # noisy_obs[random_index] = next_obs[random_index]
            # #     # noisy_obs[random_index + 2] = 0.0
            # next_obs = noisy_obs
            # Optional: You can log or print which index was zeroed
            # print(f"Zeroed index {random_index} in observation")
            
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
                cn_states = torch.cat((cn_states, cn), dim=1)
            hidden = (hn, cn)

            # Prepare for next step
            obs = next_obs
            prev_action = action
            
            if step % 16 == 0 and hn_states is not None:
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
                observations = []
                actions = []
                rewards = []
                next_observations = []
                dones = []
                log_probs = []
                prev_actions = []
                hn_states = None
                cn_states = None

                # sample = replay_buffer.sample(batch_size)
                #    # Update the agent
                # val_loss, actor_loss = agent.learn(sample)
            step += 1
        total_step += step
        reward_history.append(total_reward)
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
        # After the episode, calculate advantages
        # For simplicity, we'll use the Temporal Difference (TD) residuals as advantages
        # In practice, you might want to use Generalized Advantage Estimation (GAE)

        # print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        if episode % 100 ==0:
            # print(reward_history)
            print(np.mean(reward_history[-100:]), " num episode: ", episode, " total steps: ", total_step)
        # Perform learning step if enough samples are available
        if len(replay_buffer) >= batch_size:
            try:
                   for a in range(train_per_epoch):
                        sample = replay_buffer.sample(batch_size)
                        # Update the agent
                        val_loss, actor_loss = agent.learn(sample)
                   # print(f"Training Update - Value Loss: {val_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")
            except Exception as e:
                print(f"Training update failed: {e}")

if __name__ == "__main__":
    print("hello")
    train_agent()
