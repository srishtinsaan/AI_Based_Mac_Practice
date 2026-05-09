from src.env import SwitchEnvironment
from src.agent import DQNAgent
from src.utils.helpers import ReplayBuffer, save_model
from src.utils.plotter import plot_rewards, plot_epsilon
from src.utils.config import config
from src.states import describe_state


def train():
    env    = SwitchEnvironment()
    agent  = DQNAgent()
    buffer = ReplayBuffer()

    episode_rewards = []
    epsilon_values  = []

    print("=" * 60)
    print("         DQN TRAINING — SWITCH ENVIRONMENT")
    print("=" * 60)
    print(f"  Episodes     : {config['num_episodes']}")
    print(f"  Batch Size   : {config['batch_size']}")
    print(f"  Gamma        : {config['gamma']}")
    print(f"  LR (alpha)   : {config['alpha']}")
    print(f"  Target Update: every {config['target_update_freq']} steps")
    print(f"  Epsilon      : {config['epsilon_start']} → {config['epsilon_min']}")
    print("=" * 60 + "\n")

    for episode in range(config["num_episodes"]):
        state        = env.reset()
        total_reward = 0.0
        done         = False
        step         = 0

        while not done:
            action                   = agent.select_action(state)
            next_state, reward, done = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            agent.train(buffer)

            state        = next_state
            total_reward += reward
            step         += 1

        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_50 = (
                sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            )
            print(
                f"  Ep {episode+1:4d} | "
                f"Steps: {step:3d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Avg(50): {avg_50:7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"GlobalStep: {agent.step_count}"
            )

        # Detailed state snapshot every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"\n  --- State snapshot at episode {episode+1} ---")
            print(describe_state(state))
            print()

        # Save checkpoint every 200 episodes
        if (episode + 1) % 200 == 0:
            save_model(agent, path=f"models/checkpoint_ep{episode+1}.pth")

    # Final save and plots
    save_model(agent, path="models/dqn_final.pth")
    plot_rewards(episode_rewards)
    plot_epsilon(epsilon_values)

    print("\n" + "=" * 60)
    print("  Training Complete.")
    print(f"  Total Episodes : {config['num_episodes']}")
    print(f"  Final Epsilon  : {agent.epsilon:.4f}")
    print(f"  Final Avg(50)  : {sum(episode_rewards[-50:]) / 50:.2f}")
    print("=" * 60)