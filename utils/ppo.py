import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from models import resnet_cifar
from torch.optim import Adam
import numpy as np

class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.num_classes = env.num_classes

        # ALG STEP #1
        self.actor = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [1,1,1,1], 3, self.num_classes)
        self.critic = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [1,1,1,1], 3, 1)
        self.classifier = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [3,4,6,3], 3, self.num_classes)

        self._init_hyperparameters()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # self.classifier_optim = Adam(self.classifier.parameters(), lr=self.lr)


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        # self.max_timesteps_per_episode = 1600
        self.gamma = 0.95 # discount factor
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005


    def rollout(self):
        # Collect trajectories
        batch_obs = []             # batch observations (number of timesteps per batch, dimension of observation)
        batch_acts = []            # batch actions (number of timesteps per batch, dimension of action)
        batch_log_probs = []       # log probs of each action (number of timesteps per batch)
        batch_rews = []            # batch rewards (number of episodes, number of timesteps per episode)
        batch_rtgs = []            # batch rewards-to-go (number of timesteps per batch)
        batch_lens = []            # episodic lengths in batch (number of episodes)

        t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
#
            # Rewards this episode
            ep_rews = []

            obs = self.env.reset()
            done = False

            # Increment timesteps ran this batch so far
            t += 1

            # Collect observation
            batch_obs.append(obs)

            action, log_prob = self.get_action(obs)
            obs, rew, done, _ = self.env.step(action)

            # Collect reward, action, and log prob
            ep_rews.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
#
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
            

    def get_action(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()


    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs


    def evaluate(self, batch_obs, batch_acts):
        # Query critic for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs log_probs
        return V, log_probs


    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps: # ALG STEP #2
            # ALG STEP #3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch   
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP #5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs, batch_log_probs)

                # Calculate surrogate losses
                surr1= ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss(V, batch_rtgs)

                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()
            
import gym
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)