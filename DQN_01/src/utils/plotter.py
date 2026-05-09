import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(episode_rewards, save_path="plots/rewards.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rolling_avg = np.convolve(
        episode_rewards,
        np.ones(20) / 20,
        mode="valid"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards, alpha=0.4, color="steelblue", label="Raw Reward")
    plt.plot(range(19, len(episode_rewards)), rolling_avg, color="red", linewidth=2, label="20-ep Average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training — Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to {save_path}")


def plot_epsilon(epsilon_values, save_path="plots/epsilon.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(epsilon_values, color="darkorange", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay over Training")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Epsilon plot saved to {save_path}")


def plot_loss(loss_values, save_path="plots/loss.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(loss_values, alpha=0.6, color="crimson", linewidth=1)
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title("DQN Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")