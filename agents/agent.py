import torch
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import yaml

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class Agent:
    
    # load hyperparams from yaml file
    def __init__(self, hyperparameter_set):
      with open('agents/hyperparameters.yml', 'r') as file:
          all_hyperparameter_sets = yaml.safe_load(file)
          hyperparameters = all_hyperparameter_sets[hyperparameter_set]

      self.hyperparameter_set = hyperparameter_set
      self.env_id = hyperparameters['env_id']
      self.replay_memory_size = hyperparameters['replay_memory_size']
      self.mini_batch_size = hyperparameters['mini_batch_size']
      self.epsilon_init = hyperparameters['epsilon_init']
      self.epsilon_decay = hyperparameters['epsilon_decay']
      self.epsilon_min = hyperparameters['epsilon_min']

    def run(self, n_episodes=10, is_training=True, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None) # creates the environment

        num_states = env.observation_space.shape[0] # dynamically reads env state and actions instead of hardcoding a value specific to lunarlander
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions).to(device) # instantiates Q network which takes a state as input and ouputs Q values for each action
        
        if is_training:
          memory = ReplayMemory(self.replay_memory_size)

        for episode in range(n_episodes): # resets the env to start with a new episode 
            state, _ = env.reset() # returns initial state
            terminated = False
            episode_reward = 0 

            while True:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action) # executes the action, returns new state & reward
                episode_reward += reward
                
                if is_training:
                  memory.append((state, action, reward, new_state, terminated))
                  
                state = new_state

                # Ends the loop if agent dies or lands, or never lands
                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")

            rewards_per_episode.append(episode_reward)
        
        env.close() # cleanup


if __name__ == "__main__":
    agent = Agent('lunarlander1')
    agent.run()
