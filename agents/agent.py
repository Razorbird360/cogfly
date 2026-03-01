import torch
import torch.nn as nn
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import random
import time
import resource

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available() else "cpu"
)


class Agent:

  # load hyperparams from yaml file
  def __init__(self, hyperparameter_set):
    with open("agents/hyperparameters.yml", "r") as file:
      all_hyperparameter_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]

    self.hyperparameter_set = hyperparameter_set
    self.env_id = hyperparameters["env_id"]
    self.replay_memory_size = hyperparameters["replay_memory_size"]
    self.mini_batch_size = hyperparameters["mini_batch_size"]
    self.epsilon_init = hyperparameters["epsilon_init"]
    self.epsilon_decay = hyperparameters["epsilon_decay"]
    self.epsilon_min = hyperparameters["epsilon_min"]
    self.n_episodes = hyperparameters["n_episodes"]
    self.network_sync_rate = hyperparameters["network_sync_rate"]
    self.learning_rate = hyperparameters["learning_rate"]
    self.discount_factor_g = hyperparameters["discount_factor_g"]
    self.save_results = hyperparameters.get("save_results", False)

    self.loss_fn = nn.MSELoss()
    self.optimizer = None

  def run(self, is_training=True, render=False):
    start_time = time.time()

    env = gymnasium.make(
      self.env_id, render_mode="human" if render else None
    )  # creates the environment

    num_states = env.observation_space.shape[
      0
    ]  # dynamically reads env state and actions instead of hardcoding a value specific to lunarlander
    num_actions = env.action_space.n

    rewards_per_episode = []
    epsilon_history = []

    # instantiates Q network which takes a state as input and ouputs Q values for each action
    policy_dqn = DQN(num_states, num_actions).to(device)

    if is_training:
      self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
      memory = ReplayMemory(self.replay_memory_size)

      epsilon = self.epsilon_init

      target_dqn = DQN(num_states, num_actions).to(device)
      target_dqn.load_state_dict(policy_dqn.state_dict())  # initialize target DQN with same weights as policy DQN

      step_count = 0

    for episode in range(self.n_episodes):  # resets the env to start with a new episode
      state, _ = env.reset()  # returns initial state
      state = torch.tensor(state, dtype=torch.float, device=device)
      terminated = False
      episode_reward = 0

      while True:

        if is_training and random.random() < epsilon:
          action = env.action_space.sample()
          action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
          with torch.no_grad():
            # DQN expects batched input, so add/remove batch dim around single state
            # tensor([1, 2, 3, ...]) ==> tensor([[1, 2, 3, ...]])
            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

        # Processing:
        new_state, reward, terminated, truncated, info = env.step(action.item())

        episode_reward += reward
        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        if is_training:
          memory.append((state, action, reward, new_state, terminated))

          step_count += 1

        state = new_state

        # Ends the loop if agent dies or lands, or never lands
        if terminated or truncated:
          break

      print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")

      rewards_per_episode.append(episode_reward)

      if is_training:
        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
        epsilon_history.append(epsilon)

        if len(memory) > self.mini_batch_size:
          mini_batch = memory.sample(self.mini_batch_size)
          self.optimize(mini_batch, policy_dqn, target_dqn)

          if step_count > self.network_sync_rate:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0

    env.close()  # cleanup

    if self.save_results:
      elapsed = time.time() - start_time
      hours = int(elapsed // 3600)
      minutes = int((elapsed % 3600) // 60)
      seconds = int(elapsed % 60)
      peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

      with open(f"agents/{self.hyperparameter_set}_results.txt", "w") as f:
        for i, r in enumerate(rewards_per_episode):
          f.write(f"Episode {i + 1}: {r:.2f}\n")
        f.write(f"\n--- Run Summary ---\n")
        f.write(f"Episodes: {self.n_episodes}\n")
        f.write(f"Time elapsed: {hours}h {minutes}m {seconds}s\n")
        f.write(f"Peak RAM: {peak_ram_mb:.1f} MB\n")

  def optimize(self, mini_batch, policy_dqn, target_dqn):
    for state, action, reward, new_state, terminated in mini_batch:
      if terminated:
        target_q = reward
      else:
        with torch.no_grad():
          # Bellman equation: reward + discounted best future Q-value from target network
          target_q = reward + self.discount_factor_g * target_dqn(new_state.unsqueeze(dim=0)).squeeze().max()

      # Get policy network's Q-value for the action that was actually taken
      current_q = policy_dqn(state.unsqueeze(dim=0)).squeeze()[action]

      loss = self.loss_fn(current_q, target_q)

      self.optimizer.zero_grad()  # Clear gradients
      loss.backward()             # Compute gradients (backpropagation)
      self.optimizer.step()       # Update network weights and biases


if __name__ == "__main__":
  agent = Agent("lunarlander1")
  agent.run(is_training=True, render=False)