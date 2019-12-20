import numpy as np
import copy
import random
from collections import namedtuple, deque
from model_vrona_asymgoal import Actor, Critic


class vronaDDPG():
    #Reinforcement Learning agent that learns using DDPG.
    def __init__(self, task):
        self.task = task
        self.visionstate_size = task.visionstate_size  # self.state_size = task.state_size
        self.motionstate_size = task.motionstate_size
        self.goal_size = task.goal_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.visionstate_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.visionstate_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.motionstate_size, self.goal_size, self.action_size)
        self.critic_target = Critic(self.motionstate_size, self.goal_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.8 # 0.15
        self.exploration_sigma = 0.2 # 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 32
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.98  # discount factor
        self.tau = 0.01  # for soft update of target parameters 0.02


    def reset_episode(self):
        self.noise.reset()
        state_vi, state_mo, state_go = self.task.reset()
        # state_vi = state_vi.reshape(-1, 89, 120, 3)
        self.last_state_vi = state_vi
        self.last_state_mo = state_mo
        self.last_state_go = state_go
        return state_vi, state_mo, state_go


    def step(self, action, reward, next_state_vi, next_state_mo, next_state_go, done):
         # Save experience / reward
        self.memory.add(self.last_state_vi, self.last_state_mo, self.last_state_go, action, reward, next_state_vi, next_state_mo, next_state_go, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state_vi = next_state_vi
        self.last_state_mo = next_state_mo
        self.last_state_go = next_state_go
    
    def act(self, state_vi, state_mo, state_go):
        # Returns actions for given state(s) as per current policy.
        state_vi = state_vi.reshape(-1, 89, 120, 3)
        state_mo = state_mo.reshape(-1, self.motionstate_size) # np.reshape(states_mo, [-1, self.motionstate_size])
        state_go = state_mo.reshape(-1, self.goal_size)
        action = self.actor_local.model.predict(state_vi)
        return list(action + abs(self.noise.sample()))   # add some noise for exploration

    def learn(self, experiences):
        
        # losses_Atrain = []
        # losses_Ctrain = []
        # Update policy and value parameters using given batch of experience tuples.
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states_vi = np.vstack([e.state_vi for e in experiences if e is not None]).reshape(-1, 89, 120, 3)
        states_mo = np.vstack([e.state_mo for e in experiences if e is not None])# .reshape(-1, self.motionstate_size)
        states_go = np.vstack([e.state_go for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.goal_size)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size) 
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states_vi = np.vstack([e.next_state_vi for e in experiences if e is not None]).reshape(-1, 89, 120, 3)
        next_states_mo = np.vstack([e.next_state_mo for e in experiences if e is not None]).reshape(-1, self.motionstate_size)
        next_states_go = np.vstack([e.next_state_go for e in experiences if e is not None]).reshape(-1, self.goal_size)
 
        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = self.critic_target.model.predict_on_batch([next_states_mo,
        #                                                            self.actor_target.model.predict_on_batch(next_states_vi)])
        actions_next = self.actor_target.model.predict_on_batch(next_states_vi)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states_mo, next_states_go, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states_mo, states_go, actions], y=Q_targets)
         
        # losses_Ctrain.append(self.critic_local.model.train_on_batch(x=[states_mo, states_go, actions], y=Q_targets))
        # loss_Ctrain_mean = np.mean(losses_Ctrain)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states_mo, states_go, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states_vi, action_gradients, 1])  # custom training function  ###
        
        # losses_Atrain.append(self.actor_local.train_fn([states_vi, action_gradients, 1])[0])
        # loss_Atrain_mean = np.mean(losses_Atrain)
        
        # print(losses_Atrain)
        # print(losses_Ctrain)
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    
    def soft_update(self, local_model, target_model):
        # Soft update model parameters.# 
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


class OUNoise:
    # Ornstein-Uhlenbeck process.# 

    def __init__(self, size, mu, theta, sigma):
        # Initialize parameters and noise process.# 
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        # Reset the internal state (= noise) to mean (mu).# 
        self.state = copy.copy(self.mu)

    def sample(self):
        # Update internal state and return it as a noise sample.# 
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

class ReplayBuffer:
    # Fixed-size buffer to store experience tuples.# 

    def __init__(self, buffer_size, batch_size):
        # Initialize a ReplayBuffer object.
        
        # Params
        #    buffer_size: maximum size of buffer
        #    batch_size: size of each training batch
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_vi", "state_mo", "state_go", "action", "reward", "next_state_vi", "next_state_mo", "next_state_go", "done"])

    def add(self, state_vi, state_mo, state_go, action, reward, next_state_vi, next_state_mo, next_state_go, done):
        # Add a new experience to memory.
        e = self.experience(state_vi, state_mo, state_go, action, reward, next_state_vi, next_state_mo, next_state_go, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        # Randomly sample a batch of experiences from memory.# 
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        # Return the current size of internal memory.# 
        return len(self.memory)