import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def ppo_loss(old_probs, advantages, values, new_probs, epsilon=0.2, c1=1.0, c2=0.01):
    ratio = torch.exp(new_probs - old_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    actor_loss = -torch.mean(torch.min(surr1, surr2))

    critic_loss = 0.5 * F.mse_loss(values, advantages)

    entropy_loss = -torch.mean(new_probs * torch.log(new_probs + 1e-8))

    total_loss = actor_loss + c1 * critic_loss + c2 * entropy_loss
    return total_loss






class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        # Hyperparameters
        self.lr = 3e-4
        self.gamma  = .95
        self.lambda_ = .95
        self.epsilon = .2
        self.value_scale = .5
        self.entropy_scale = .01
        self.horizon = 128
        self.num_epochs = 10
        self.batch_size = 128
        self.num_envs = 16

        # Training
        self.model_name = "Car_racing_paolo"
        self.save_interval = 1000
        self.eval_interval = 200
        self.record_episodes = True

    def crop(self , frame):
        # Crop to 84x84
        return frame[:-12 , 6:-6]
    
    def rgb_to_gray(self, frame):
        # Change to gray scale
        return np.dot(frame[...,0:3] , [[0.299, 0.587, 0.114]])  # ????
    
    def normalize(self, frame):
        return frame / 255
    
    def process_frame(self ,rame):
        frame = self.crop(frame)
        frame = self.rgb_to_gray(frame)
        frame = self.normalize(frame)
        frame = frame * 2 - 1

    def make_env(self):
        return gym.make('CarRacing-v2', continuous= Policy.continuous)
    

    def train(self , model_name , save_interval = 1000 , eval_interval = 200,
            record_episode = True , restart = False):
        
        try:
            env = gym.make('CarRacing-v2', continuous= Policy.continuous)
            # Traning parameters
            initial_lr = self.lr
            discount_factor = self.gamma
            gae_lambda = self.lambda_
            ppo_epsilon = self.epsilon
            value_scale = self.value_scale
            entropy_scale = self.entropy_scale
            horizon = self.horizon
            num_epochs = self.num_epochs
            batch_size = self.batch_size
            num_envs = self.num_envs

            def lr_scheduler(step_idx): return initial_lr * \
            0.85 ** (step_idx // 10000)

            # Environment constants
            frame_stack_size = 4
            input_shape = (84, 84, frame_stack_size)
            num_actions = env.action_space.shape[0]
            action_min = env.action_space.low
            action_max = env.action_space.high


            state_size = env.observation_space.shape[0]
            action_size = env.action_space.shape[0]
            # Actor and Critic Networks
            actor_network = Actor(state_size, action_size)
            critic_network = Critic(state_size)
            
            optimizer_actor = torch.optim.Adam(actor_network.parameters(), lr= initial_lr)
            optimizer_critic = torch.optim.Adam(critic_network.parameters(), lr=initial_lr)

            # Training loop

            for epoch in range(num_epochs):
                all_states = []
                all_actions = []
                all_rewards = []
                all_values = []
                for episode in range(num_epochs):
                    state = env.reset()
                    episode_states = []
                    episode_actions = []
                    episode_rewards = []
                    episode_values = []
                    for step in range(3): # loop over the number of steps per epochs
                        state = torch.FloatTensor(state)
                        action_prob = actor_network(state)
                        
                        value = critic_network(state)
                        
                        action = np.random.normal(loc=action_prob.detach().numpy(), scale=0.2)
                        action = np.clip(action, -1.0, 1.0)
                        
                        next_state, reward, done, _ = env.step(action)

                        episode_states.append(state)
                        episode_actions.append(torch.FloatTensor(action))
                        episode_rewards.append(reward)
                        episode_values.append(value)

                        state = next_state

                        if done:
                            break

            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            all_rewards.extend(episode_rewards)
            all_values.extend(episode_values)

        # Compute advantages
            next_value = critic_network(torch.FloatTensor(next_state)).detach().numpy()
            all_advantages = []
            cumulative_advantage = 0

            for value, reward, next_value, done in zip(reversed(all_values), reversed(all_rewards), reversed([next_value] + [0] * (num_steps_per_episode - 1)), reversed([done] + [False] * (num_steps_per_episode - 1))):
                td_error = reward + discount_factor * next_value * (1 - int(done)) - value.detach().numpy()
                cumulative_advantage = td_error + discount_factor * cumulative_advantage
                all_advantages.insert(0, cumulative_advantage)

            all_states = torch.stack(all_states)
            all_actions = torch.stack(all_actions)
            all_advantages = torch.FloatTensor(all_advantages)

            # Update Actor and Critic
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            new_policy_probs = actor_network(all_states)
            old_policy_probs = actor_network(all_states.detach())
            values = critic_network(all_states)

            loss = ppo_loss(old_policy_probs, all_advantages, values, new_policy_probs, epsilon=epsilon)
            loss.backward()

            optimizer_actor.step()
            optimizer_critic.step()

            if epoch % 10 == 0:
                total_reward = 0
                state = env.reset()

                for _ in range(500):
                    state = torch.FloatTensor(state)
                    action = actor_network(state).detach().numpy()
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    state = next_state

                    if done:
                        break

            print(f"Epoch: {epoch}, Total Reward: {total_reward}")

            env.close()