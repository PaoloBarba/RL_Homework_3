import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self,
                 capacity,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001,
                 batch_size=32):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = np.empty(self.capacity, dtype=object)
        self.buffer_index = 0
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.batch_size = batch_size

    def add(self, experience):
        priority = max(self.priorities) if self.buffer_index else 1.0 # if buffer is empty, set priority to 1.0
        if self.buffer_index == self.capacity: # if memory is full, remove the least important one, alternatively remove the oldest one
            min_priority_index = np.argmin(self.priorities)
            self.buffer[min_priority_index] = experience
            self.priorities[min_priority_index] = priority
        else:
            self.buffer[self.buffer_index]= experience
            self.priorities[self.buffer_index] = priority
            self.buffer_index += 1

    def sample(self):
        priorities = np.array(self.priorities[:self.buffer_index])
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.buffer_index,
                                    self.batch_size,
                                    p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.buffer_index * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, new_priorities):
        self.priorities[indices] = new_priorities

    def process_samples(self, samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sample in samples:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.vstack(actions), np.vstack(rewards), np.array(next_states), np.vstack(dones)


class QNetwork(nn.Module):
    def __init__(self,
                 num_frames, #TODO: only one frame is used, but the network can be designed to handle multiple frames
                 inpute_size,
                 action_size,
                 seed=0,
                 device=torch.device('cpu')):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_frames = num_frames
        self.input_shape = inpute_size
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=8, stride=4) # Input: (batchsize, 1, 84, 84) --> (batchsize, 16, 20, 20)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # input: (batchsize, 16, 20, 20) --> (batchsize, 32, 9, 9)
        self.fc1 = nn.Linear(32*9*9, 256) # input: (batchsize, 32*9*9) --> (batchsize, 256)
        self.fc2 = nn.Linear(256, action_size)
        self.device = device

    def rbg2gray(self, image):
        return rgb_to_grayscale(image)/255.0 #cpnvert to grayscale and normalize

    def crop(self, image):
        return image[:, 0:84, 6:90, :] # environment specific cropping to remove score and speedometer

    def forward(self, state):
        x = self.crop(state)
        x = x.permute(0, 3, 1, 2) # Permute for torch standards: (BATCHSIZE, 96, 96, 3) --> (BATCHSIZE, 3, 96, 96)
        x = self.rbg2gray(x).squeeze(0) # (BATCHSIZE, 96, 96)
        #x = torch.tensor(x).to(device=self.device)
        if len(x.shape) != 4:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Policy(nn.Module):
    continuous = False # you can change this
    def __init__(self,
                 device="auto",
                 lr=1e-4,
                 gamma=0.9999,
                 epsilon=1,
                 epsilon_decay=0.995,
                 final_epsilon=0.02,
                 replay_period=4,
                 tau=0.001,
                 num_frames=1, #TODO: implement image stack
                 num_episodes=800,
                 max_steps_per_episode=600,
                 warming_up_phase = 1000,
                 checkpoint_dir='checkpoint_v6/',
                 checkpoint_filepath='ddqn_per.pth',
                 ):
        super(Policy, self).__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.env = gym.make('CarRacing-v2', continuous=False)


        ## hyperparameters
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        ## RL agent setup
        self.batch_size = 32
        self.number_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filepath = checkpoint_filepath
        self.memory = ReplayBuffer(30000,
                                   batch_size=self.batch_size)

        self.qnetwork_policy = QNetwork(num_frames, # TODO: implement image stack, num_frames obsolete
                                        self.env.observation_space.shape,
                                        self.env.action_space.n,
                                        device=self.device).to(self.device)
        self.qnetwork_target = QNetwork(num_frames,
                                        self.env.observation_space.shape,
                                        self.env.action_space.n,
                                        device=self.device).to(self.device)
        self.replay_period = replay_period

        ## training parameters and setup
        self.warming_up_phase = warming_up_phase
        self.max_timesteps = num_episodes * max_steps_per_episode
        self.final_epsilon = final_epsilon
        self.optimizer = torch.optim.Adam(self.qnetwork_policy.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def update_epsilon(self):# from assigment 2
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)

    def decay_epsilon(self, step):
        # 10% exploratiom rate
        epsilon = self.epsilon_start - (step * ((self.epsilon_start - self.final_epsilon) / int(0.1*self.max_timesteps)))
        self.epsilon = max(epsilon, self.final_epsilon)

    def policy(self, state): # do not touch
        return torch.argmax(self.qnetwork_policy(state)).item()

    def forward(self, x):
        return self.qnetwork_policy(x)

    def act(self, state):
        if random.random()<self.epsilon:
            return self.env.action_space.sample()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.policy(state)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def compute_delta(self, state, action, reward, next_state, done):
        return reward + self.gamma * self.qnetwork_target(next_state).max(1)[0].unsqueeze(1) * (1 - done) - self.qnetwork_policy(state).gather(1, action)

    def learn_from_experiences(self):
        samples, indices, weights = self.memory.sample()
        states, actions, rewards, next_states, dones = self.memory.process_samples(samples)
        # convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)


        deltas = self.compute_delta(states, actions, rewards, next_states, dones)
        self.memory.update_priorities(indices, deltas.detach().squeeze().abs().cpu().numpy()+1e-5)

        loss = torch.mean((deltas*weights)**2) # weighted loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_policy.parameters(), 1)
        self.optimizer.step()
        #self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict()) # update target network
        self.soft_update(self.qnetwork_policy, self.qnetwork_target, self.tau)
        return loss.item()

    # update target network parameters slowly from local network parameters to avoid sudden changes that could destabilize learning
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self):
        print(f'ep | eps rew | mean reward | epsilon')
        scores = []
        train_loss = []
        steps_so_far = 0
        best_mean_reward = -np.inf

        for episode in range(self.number_episodes):
            state,_ = self.env.reset()
            score = 0
            done = False
            for _ in range(50): # Zoom-in phase skipped
               state, _, _, _, _ = self.env.step(0)
            for timestep in range(self.max_steps_per_episode):
                action = self.act(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                if (timestep >= self.warming_up_phase) or (timestep % self.replay_period == 0):
                    loss = self.learn_from_experiences()
                    train_loss.append(loss)
                state = next_state
                score += reward
                self.decay_epsilon(steps_so_far)
                # self.update_epsilon()
                if (steps_so_far % 10000  == 0) and (steps_so_far!=0):
                    self.plot_rewards(scores)
                    model_path = self.checkpoint_dir + f"checkpoint_{steps_so_far}_{self.checkpoint_filepath}"
                    print(f"saving model to {model_path}")
                    torch.save(self.qnetwork_target.state_dict(), model_path)

                if (timestep > 50_000) and (np.mean(scores[-100:]) >= best_mean_reward):
                  best_mean_reward = np.mean(scores[-100:])
                  print(f'Current best model with average reward over the last 100 episodes: {best_mean_reward}')
                  self.plot_rewards(scores)
                  model_path = self.checkpoint_dir + f"current_best_model_after_{steps_so_far}_{self.checkpoint_filepath}"
                  print(f"saving model to {model_path}")
                  torch.save(self.qnetwork_target.state_dict(), model_path)

                if steps_so_far ==  self.max_timesteps:
                    break
                steps_so_far += 1
                if done or truncated:
                    break




            scores.append(score)
            average_score = np.mean(scores[-100:])
            print(f'{episode+1} | {scores[-1]:.2f} | {average_score:.2f} | {self.epsilon:.2f}')

            # save model if average score is greater than 110.0
            if average_score >= 400.0: #
                print(f"Decent Agent, average score after {episode} episodes: {average_score:.2f}")
                model_path =  self.checkpoint_dir + f"final_model{self.checkpoint_filepath}"
                print(f"saving model to {model_path}")
                torch.save(self.state_dict(), model_path)
                break
            # save model if max timesteps is reached
            if steps_so_far == self.max_timesteps:
                model_path = self.checkpoint_dir + f"last_checkpoint_{steps_so_far}_{self.checkpoint_filepath}"
                print(f"saving last model to {model_path}")
                torch.save(self.state_dict(), model_path)
                break
            self.plot_rewards(scores)
        return scores

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def plot_rewards(self, rewards):
        plt.plot(np.array(rewards))
        # plot the moving average
        window_size = 100
        window = np.ones(int(window_size))/float(window_size)
        plt.plot(np.convolve(rewards, window, 'same'))
        plt.legend(['Episode reward', 'Moving average'], loc='upper left')
        plt.title('Episode rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(self.checkpoint_dir + "episode_rewards.png")
        plt.close()

    def load(self):
        self.qnetwork_policy.load_state_dict(torch.load('checkpoint_v0/checkpoint_240000_ddqn_per.pth', map_location=self.device))
        self.qnetwork_target.load_state_dict(torch.load('checkpoint_v0/checkpoint_240000_ddqn_per.pth', map_location=self.device))
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


