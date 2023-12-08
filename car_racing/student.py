import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple # mine
from collections import deque # mine
import matplotlib.pyplot as plt #mine
import cv2 # mine





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

        self.layer1 = nn.Linear( 32 * 9 *9 , 64 , bias=bias)

        self.layer2 = nn.Linear(64,32,bias=bias)

        self.layer3 = nn.Linear(32,n_outputs,bias=bias)


        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                        lr=learning_rate)
        
    

    def crop(self , img):
        img = img[:96, 6:90] # CarRacing-v2-specific cropping
    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        return img
    
    def forward(self , state):
        x = self.crop(state)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.reshape(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.layer1(x.reshape(x.size(0) , -1 )))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

    def get_qvals(self, state):
        out = self.network(state)
        return out


class Experience_replay_buffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, s_0, a, r, d, s_1):
        self.replay_memory.append(
            self.Buffer(s_0, a, r, d, s_1))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size



# Define a function to transform tuple to tensor object
def from_tuple_to_tensor(tuple_of_np):
    tensor = torch.zeros((len(tuple_of_np), tuple_of_np[0].shape[0]))
    for i, x in enumerate(tuple_of_np):
        tensor[i] = torch.FloatTensor(x)
    return tensor



class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.env = env   # Set the env
        self.lr = 0.001  # Set the learning rate
        self.buffer = Experience_replay_buffer()
        self.network = Q_network(env, self.lr).to(device)        # Define 1st network
        self.target_network = Q_network(env, self.lr).to(device) # Define 2nd network
        self.epsilon = 0.5
        self.batch_size = 64
        self.window = 50
        self.reward_threshold = 400
        #self.inizialize()
        self.step_count = 0
        self.episode = 0
        self.rewards = 0





    def forward(self, x):
        # TODO
        return x
    
    def act(self, state , mode = 'exploit'):
        # TODO
        # Choose action with esp-greedy
        if mode == 'explore':
                action = self.env.action_space.sample()
        else:
                action = self.network.greedy_action(torch.FloatTensor(self.s_0).to(device))

        #simulate action
        s_1, r, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        #put experience in the buffer
        self.buffer.append(self.s_0, action, r, terminated, s_1)

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0, _ = self.env.reset()
        return done 
    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
                action = self.env.action_space.sample()
        else:
                action = self.network.greedy_action(torch.FloatTensor(self.s_0).to(device))

        #simulate action
        s_1, r, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        #put experience in the buffer
        self.buffer.append(self.s_0, action, r, terminated, s_1)

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0, _ = self.env.reset()
        return done

    def train(self , gamma = .99 , max_episodes=10000,
            network_update_frequency=10,
            network_sync_frequency=200):
        # TODO
        self.gamma = gamma

        self.loss_function = nn.MSELoss()
        self.s_0, _ = self.env.reset()
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        ep = 0
        training = True
        self.populate = False
        while training:
            self.s_0, _ = self.env.reset()

            self.rewards = 0
            done = False
            while not done:
                if ((ep % 5) == 0):
                    self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='exploit')
                    # print("train")
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.7
                    ep += 1
                    if self.rewards > 2000:
                        self.training_rewards.append(2000)
                    elif self.rewards > 1000:
                        self.training_rewards.append(1000)
                    elif self.rewards > 500:
                        self.training_rewards.append(500)
                    else:
                        self.training_rewards.append(self.rewards)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f}\t\t".format(
                            ep, mean_rewards, self.rewards, mean_loss), end="")

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break
        self.save()                
        return
    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()

    def calculate_loss(self, batch):
        #extract info from batch
        states, actions, rewards, dones, next_states = list(batch)

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(device)
        states = from_tuple_to_tensor(states).to(device)
        next_states = from_tuple_to_tensor(next_states).to(device)

        ###############
        # DDQN Update #
        ###############
        # Q(s,a) = ??
        qvals = self.network.get_qvals(states)
        qvals = torch.gather(qvals, 1, actions)

        # target Q(s,a) = ??
        next_qvals= self.target_network.get_qvals(next_states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        # loss = self.loss_function( Q(s,a) , target_Q(s,a))
        loss = self.loss_function(qvals, target_qvals)

        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        loss.backward()
        self.network.optimizer.step()

        self.update_loss.append(loss.item())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0

    def evaluate(self, eval_env):
        done = False
        s, _ = eval_env.reset()
        rew = 0
        while not done:
            action = self.network.greedy_action(torch.FloatTensor(s).to(device))
            s, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            rew += r

        print("Evaluation cumulative reward: ", rew)


    


    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


agent = Policy()
agent.train()


eval_env = gym.make("CarRacing-v2", render_mode="human")
agent.evaluate(eval_env)