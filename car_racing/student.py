import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = torch.distributions.Categorical(F.softmax(output, dim=-1))
        return distribution
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value






class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.gamma  = .99
        self.seed = 123
        self.nsteps = 1024
        self.total_time_steps = int(1e6)
        self.lr =  3e-4
        self.batch_size = 64
        self.tau = .95
        self.damping = .1
        self.max_kl = .01
        self.done = False
        self.env =  gym.make('CarRacing-v2', continuous= Policy.continuous)
        self.critic = Critic(self , self.env.observation_space.shape[0] , self.env.action_space.shape[0])
        self.actor = Actor(self , self.env.observation_space.shape[0] , self.env.action_space.shape[0])



        



    def forward(self, x):
        state_value = self.critic
        return x
    
    def act(self, state):
        # TODO
        return 

    def train(self):
        num_updates = self.total_time_steps // self.nsteps
        obs = self.env.observation_space
        final_reward = 0 
        episode_reward = 0
        self.done = False
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            for step in range(self.nsteps):
                with torch.no_grad():
                    obs_tensor = self.get_tensor(obs)
                    #value , pi = self.


        


        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
