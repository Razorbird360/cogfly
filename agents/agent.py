import torch
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import random

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
        self.save_results = hyperparameters.get("save_results", False)

    def run(self, is_training=True, render=False):
        env = gymnasium.make(
            self.env_id, render_mode="human" if render else None
        )  # creates the environment

        num_states = env.observation_space.shape[
            0
        ]  # dynamically reads env state and actions instead of hardcoding a value specific to lunarlander
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(
            device
        )  # instantiates Q network which takes a state as input and ouputs Q values for each action

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

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

                state = new_state

                # Ends the loop if agent dies or lands, or never lands
                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")

            rewards_per_episode.append(episode_reward)

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

        env.close()  # cleanup

        if self.save_results:
            with open(f"agents/{self.hyperparameter_set}_results.txt", "w") as f:
                for i, r in enumerate(rewards_per_episode):
                    f.write(f"Episode {i + 1}: {r:.2f}\n")


if __name__ == "__main__":
    agent = Agent("lunarlander1")
    agent.run(is_training=True, render=False)
