import numpy as np
import torch

class ReplayBuffer():

    def __init__(self, size):

        self.observation = None
        self.next_observation = None
        self.reward = None
        self.action = None
        self.done = None
        self.logprobs = None
        self.advantage = None
        self.hn = None
        self.cn = None
        self.prev_action = None
        self.maxsize = size
        self.size = 0

    def sample(self, batch_size):
        if self.observation is None:
            return None
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.maxsize
        return {
            "observations": self.observation[rand_indices],
            "actions": self.action[rand_indices],
            "rewards": self.reward[rand_indices],
            "next_observations": self.next_observation[rand_indices],
            "dones": self.done[rand_indices],
            "advantages": self.advantage[rand_indices],
            "log_probs": self.log_probs[rand_indices],
            "prev_actions": self.prev_action[rand_indices],
            "hn": self.hn[rand_indices].permute(1, 0, 2),
            "cn": self.cn[rand_indices].permute(1, 0, 2)
        }

    def __len__(self):
        return self.size

    def add_sequence(self, observation, next_observation, action, reward, done, advantage, log_probs, hn, cn, prev_action):
        #batches need to be given in [batch_size, data_shape]
        size = len(observation)
        hnt = hn.permute(1, 0, 2) #change from (num_layers×num_directions, batch_size, hidden_size) to (batch_size, num_layers×num_directions, hidden_size)
        cnt = cn.permute(1, 0, 2)
        if self.observation is None:
            self.observation = np.empty((self.maxsize, *observation[0].shape), dtype= observation.dtype)
            self.next_observation = np.empty((self.maxsize, *next_observation[0].shape), dtype= next_observation.dtype)
            self.reward = np.empty((self.maxsize, *reward[0].shape), dtype = reward.dtype)
            self.action = np.empty((self.maxsize, *action[0].shape), dtype=action.dtype)
            self.done = np.empty((self.maxsize, *done[0].shape), dtype=done.dtype)
            self.advantage = np.empty((self.maxsize, *advantage[0].shape), dtype=advantage.dtype)
            self.log_probs = np.empty((self.maxsize, *log_probs[0].shape), dtype=log_probs.dtype)
            self.prev_action = np.empty((self.maxsize, *prev_action[0].shape), dtype=action.dtype)
            self.hn = torch.empty((self.maxsize, *hnt[0].shape), dtype=torch.float32)
            self.cn = torch.empty((self.maxsize, *cnt[0].shape), dtype=torch.float32)



        start_idx = self.size % self.maxsize
        end_idx   = start_idx + size

        if end_idx <= self.maxsize:
            # ----- (A) NO WRAP-AROUND -----
            self.observation[start_idx:end_idx]      = observation
            self.action[start_idx:end_idx]           = action
            self.reward[start_idx:end_idx]           = reward
            self.next_observation[start_idx:end_idx] = next_observation
            self.done[start_idx:end_idx]             = done
            self.advantage[start_idx:end_idx]        = advantage
            self.log_probs[start_idx:end_idx]        = log_probs
            self.prev_action[start_idx:end_idx]      = prev_action
            self.hn[start_idx:end_idx]               = hnt
            self.cn[start_idx:end_idx]               = cnt
        else:
            # ----- (B) WRAP-AROUND -----
            # how many items fit until the end of the buffer
            first_part_len = self.maxsize - start_idx
            
            # fill from start_idx to the end of the buffer
            self.observation[start_idx:self.maxsize]      = observation[:first_part_len]
            self.action[start_idx:self.maxsize]           = action[:first_part_len]
            self.reward[start_idx:self.maxsize]           = reward[:first_part_len]
            self.next_observation[start_idx:self.maxsize] = next_observation[:first_part_len]
            self.done[start_idx:self.maxsize]             = done[:first_part_len]
            self.advantage[start_idx:self.maxsize]        = advantage[:first_part_len]
            self.log_probs[start_idx:self.maxsize]        = log_probs[:first_part_len]
            self.prev_action[start_idx:self.maxsize]      = prev_action[:first_part_len]
            self.hn[start_idx:self.maxsize]               = hnt[:first_part_len]
            self.cn[start_idx:self.maxsize]               = cnt[:first_part_len]

            
            # the remainder wraps to the beginning of the buffer
            second_part_len = end_idx % self.maxsize  # = N - first_part_len
            self.observation[0:second_part_len]      = observation[first_part_len:]
            self.action[0:second_part_len]           = action[first_part_len:]
            self.reward[0:second_part_len]           = reward[first_part_len:]
            self.next_observation[0:second_part_len] = next_observation[first_part_len:]
            self.done[0:second_part_len]             = done[first_part_len:]
            self.advantage[0:second_part_len]        = advantage[first_part_len:]
            self.log_probs[0:second_part_len]        = log_probs[first_part_len:]
            self.prev_action[0:second_part_len]      = prev_action[first_part_len:]
            self.hn[0:second_part_len]               = hnt[first_part_len:]
            self.cn[0:second_part_len]               = cnt[first_part_len:]

        
        # 4. Update total insertion count
        self.size += size
