import torch
import gymnasium
from dqn import DQN

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class Agent:
    def run(self, n_episodes=10, is_training=True, render=False):
        env = gymnasium.make("LunarLander-v3", render_mode="human" if render else None) # creates the environment

        num_states = env.observation_space.shape[0] # dynamically reads env state and actions instead of hardcoding a value specific to lunarlander
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device) # instantiates Q network which takes a state as input and ouputs Q values for each action

        for episode in range(n_episodes): # resets the env to start with a new episode 
            obs, _ = env.reset() # returns initial state
            total_reward = 0 

            while True:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                obs, reward, terminated, truncated, info = env.step(action) # executes the action, returns new state & reward
                total_reward += reward

                # Ends the loop if agent dies or lands, or never lands
                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

        env.close() # cleanup


if __name__ == "__main__":
    agent = Agent()
    agent.run()
