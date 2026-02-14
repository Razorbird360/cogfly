import gymnasium as gym


def run_random_agent(n_episodes=10):
    env = gym.make("LunarLander-v3")

    for episode in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    run_random_agent()
