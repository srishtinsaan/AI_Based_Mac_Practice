from src.env.switch_env import SwitchEnvironment
from src.agent import DQNAgent
from src.utils.helpers import ReplayBuffer, set_seed, save_model
from src.utils.plotter import plot_rewards, plot_epsilon, plot_loss
from src.utils.config import config

def main():
    set_seed(42)

    env    = SwitchEnvironment()
    agent  = DQNAgent()
    buffer = ReplayBuffer()

    episode_rewards = []
    epsilon_values  = []
    loss_values     = []

    print("Starting DQN Training...\n")

    for episode in range(config["num_episodes"]):
        state       = env.reset()
        total_reward = 0
        done         = False

        while not done:
            action                       = agent.select_action(state)
            next_state, reward, done     = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            agent.train(buffer)
            state        = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        if (episode + 1) % 50 == 0:
            avg = sum(episode_rewards[-50:]) / 50
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Avg(50): {avg:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Steps: {agent.step_count}")

    save_model(agent)
    plot_rewards(episode_rewards)
    plot_epsilon(epsilon_values)

    print("\nTraining complete.")

if __name__ == "__main__":
    main()