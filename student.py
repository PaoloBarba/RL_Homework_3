import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple # mine
from collections import deque # mine
import matplotlib.pyplot as plt #mine
from torchvision.transforms.functional import rgb_to_grayscale

import cv2 # mine

import random



class Experience_replay_buffer:
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

    def append(self, experience):
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


# Set the device we work on, unfortunately i don't have gpu :(
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CarRacing-v2', continuous= False)

# Define the network class
class Net(nn.Module):
    def __init__(self, n_outputs, bias=True):
        super().__init__()



    def forward(self, x):
        x = self.activation_function( self.layer1(x) )
        x = self.activation_function( self.layer2(x) )
        y = self.layer3(x)
        return y

# Define the Q-Network
class Q_network(nn.Module):

    def __init__(self, env, learning_rate=1e-4 , bias = True):
        super(Q_network, self).__init__()

        n_outputs = env.action_space.n


        self.conv_1 = nn.Conv2d(1, 16 , kernel_size = 8, stride=4)
        self.conv_2 = nn.Conv2d(16, 32 , kernel_size = 4, stride=2)

        self.layer1 = nn.Linear( 2592 , 64 , bias=bias)

        self.layer2 = nn.Linear(64,32,bias=bias)

        self.layer3 = nn.Linear(32,5,bias=bias)

        
    

    def crop(self , img):



        img = img[: , 0:84, 6:90, : ] # CarRacing-v2-pecific cropping
        img = img.permute(0, 3, 1, 2)
        img = rgb_to_grayscale(img)/ 255.0
        img = img.squeeze(0) 


        return img
    
    def forward(self , state):
        x = self.crop(state)
        if len(x.shape) != 4:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        
        x = x.reshape(x.size(0), -1)  # flatten the tensor
        #x = F.relu(self.layer1(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

    def get_qvals(self, state):
        out = self.forward(state)
        return out


class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = env   # Set the env
        self.lr = 0.001  # Set the learning rate
        self.tau = 0.001
        self.epsilon = 0.5
        self.epsilon_start = 0.5
        self.gamma = 0.2
        self.epsilon_decay = 0.9
        self.window = 50
        self.reward_threshold = 400
        #self.inizialize()
        self.step_count = 0
        self.warming_up_phase = 100
        self.number_episodes = 100
        self.rewards = 0
        self.env = gym.make('CarRacing-v2', continuous=False)


        # RL agent
        self.batch_size = 64
        self.number_episodeds = 100
        self.max_steps_per_episode = 100
        self.buffer =  Experience_replay_buffer(30000,
                                                batch_size=self.batch_size)

        self.qnetwork_policy = Q_network(self.env)
        self.qnetwork_target = Q_network(self.env)

        self.replay_period = 4

        # Training parameters
        self.warming_up = 100
        self.max_timesteps =  self.number_episodeds * self.max_steps_per_episode
        self.final_epsilon = 0.001
        self.optimizer = torch.optim.Adam(self.qnetwork_policy.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def epsilon_update(self):
        self.epsilon = max(self.final_epsilon , self.epsilon * self.epsilon_decay )
    
    def decay_epsilon(self, step):
        epsilon = self.epsilon_start - (step * ((self.epsilon_start - self.final_epsilon) / int(0.1*self.max_timesteps)))
        self.epsilon = max(epsilon, self.final_epsilon)

    def policy(self,state):
        return torch.argmax(self.qnetwork_policy(state)).item()
    
    def forward(self, x):
        return self.qnetwork_policy(x)
    
    def act(self , state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.policy(state)
    
    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def compute_delta(self, state, action, reward, next_state, done):
        return reward + self.gamma * self.qnetwork_target(next_state).max(1)[0].unsqueeze(1) * (1 - done) - self.qnetwork_policy(state).gather(1, action)

    def learn_from_experiences(self):
        samples, indices, weights = self.buffer.sample()
        states, actions, rewards, next_states, dones = self.buffer.process_samples(samples)
        # convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)


        deltas = self.compute_delta(states, actions, rewards, next_states, dones)
        self.buffer.update_priorities(indices, deltas.detach().squeeze().abs().cpu().numpy()+1e-5)

        loss = torch.mean((deltas*weights)**2) # weighted loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.qnetwork_policy.parameters(), 1)
        self.optimizer.step()
        #self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict()) # update target network
        self.soft_update(self.qnetwork_policy, self.qnetwork_target, self.tau)
        return loss.item()



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
                    #model_path = self.checkpoint_dir + f"checkpoint_{steps_so_far}_{self.checkpoint_filepath}"
                    print(f"saving model to {model_path}")
                    torch.save(self.qnetwork_target.state_dict(), "target_model.pt")

                if (timestep > 50_000) and (np.mean(scores[-100:]) >= best_mean_reward):
                  best_mean_reward = np.mean(scores[-100:])
                  print(f'Current best model with average reward over the last 100 episodes: {best_mean_reward}')
                  self.plot_rewards(scores)
                  #model_path = self.checkpoint_dir + f"current_best_model_after_{steps_so_far}_{self.checkpoint_filepath}"
                  print(f"saving model to {model_path}")
                  torch.save(self.qnetwork_target.state_dict(), "target_model.pt")

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
                #print(f"Decent Agent, average score after {episode} episodes: {average_score:.2f}")
                #model_path =  self.checkpoint_dir + f"final_model{self.checkpoint_filepath}"
                #print(f"saving model to {model_path}")
                torch.save(self.state_dict(), 'model.pt')
                break
            # save model if max timesteps is reached
            if steps_so_far == self.max_timesteps:
                #model_path = self.checkpoint_dir + f"last_checkpoint_{steps_so_far}_{self.checkpoint_filepath}"
                #print(f"saving last model to {model_path}")
                torch.save(self.state_dict(), "model.pt")
                break
            self.plot_rewards(scores)
        return scores
    
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
        #plt.savefig(self.checkpoint_dir + "episode_rewards.png")
        plt.close()

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


def evaluate(env=None, n_episodes=1, render=False):
    agent = Policy()
    agent.load()

    env = gym.make('CarRacing-v2', continuous=agent.continuous)
    if render:
        env = gym.make('CarRacing-v2', continuous=agent.continuous, render_mode='human')

    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)

            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    print('Mean Reward:', np.mean(rewards))


def train():
    agent = Policy()
    agent.train()
    agent.save()


def main():

    if True: #args.train:
        train()

    if False: # args.evaluate:
        evaluate(render=args.render)

if __name__ == '__main__':
    main()
