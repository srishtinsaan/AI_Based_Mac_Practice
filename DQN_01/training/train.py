# src/training/train.py

import torch
import numpy as np
import os
from src.config import (
    GAMMA,
    BATCH_SIZE,
    MAX_EPISODES,
    MAX_STEPS,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    TARGET_UPDATE_N
)
from src.actions          import get_action_name, NUM_ACTIONS
from src.agent.replay_buffer import ReplayBuffer
from src.training.target_update import TargetUpdater
from src.rewards          import episode_summary


class Trainer:
    """
    Main training loop for DQN.
    Connects environment, agent, replay buffer,
    networks, and target updater together.
    Runs episodes, collects transitions,
    trains prediction network, logs results.
    """

    def __init__(self, env, networks):
        """
        Parameters:
            env      : SwitchEnvironment instance
            networks : DQNNetworks instance
        """
        self.env            = env
        self.networks       = networks
        self.replay_buffer  = ReplayBuffer()
        self.target_updater = TargetUpdater(networks)

        # ── epsilon ──────────────────────────
        self.epsilon        = EPSILON_START

        # ── tracking ─────────────────────────
        self.total_steps    = 0       # across all episodes
        self.training_steps = 0      # number of times network trained
        self.episode_rewards= []     # total reward per episode
        self.episode_losses = []     # avg loss per episode
        self.best_reward    = float('-inf')

        # ── results folder ───────────────────
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/logs',   exist_ok=True)

    # ─────────────────────────────────────────
    # SELECT ACTION — epsilon greedy
    # ─────────────────────────────────────────
    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon  → random action (explore)
        With probability 1-epsilon → greedy action (exploit)

        Parameters:
            state : numpy array (3,)

        Returns:
            action : int (0-6)
        """
        if np.random.random() < self.epsilon:
            # explore — random action
            return np.random.randint(0, NUM_ACTIONS)
        else:
            # exploit — best action from prediction network
            q_values = self.networks.get_q_values(state)
            return int(np.argmax(q_values))

    # ─────────────────────────────────────────
    # TRAIN STEP — one mini batch update
    # ─────────────────────────────────────────
    def train_step(self):
        """
        One complete training step:
        1. Sample 64 transitions from replay buffer
        2. Calculate y_i using target network
        3. Calculate predicted Q using prediction network
        4. Compute MSE loss
        5. Backpropagation
        6. SGD weight update
        7. Check target network update

        Returns:
            loss : float
        """
        # ── sample mini batch ────────────────
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        # ── convert to tensors ───────────────
        device = self.networks.device

        states_t      = torch.FloatTensor(states).to(device)       # (64, 3)
        actions_t     = torch.LongTensor(actions).to(device)       # (64,)
        rewards_t     = torch.FloatTensor(rewards).to(device)      # (64,)
        next_states_t = torch.FloatTensor(next_states).to(device)  # (64, 3)
        dones_t       = torch.FloatTensor(dones).to(device)        # (64,)

        # ── get predicted Q values ───────────
        # prediction network forward pass
        # only Q-value of action actually taken
        q_predicted = self.networks.get_predicted_q(
            states_t,
            actions_t
        )  # (64, 1)

        # ── calculate y_i (target) ───────────
        # target network forward pass on next states
        max_q_next = self.networks.get_target_q(
            next_states_t
        )  # (64, 1)

        # Bellman equation:
        # y_i = reward + gamma * max Q(s', a'; theta-)
        # if done → y_i = reward only (no future)
        # (1 - dones_t) zeroes out future Q when done=True
        y_i = rewards_t.unsqueeze(1) + \
              GAMMA * max_q_next * (1 - dones_t.unsqueeze(1))
        # y_i shape: (64, 1)

        # ── compute loss ─────────────────────
        # MSE between target and prediction
        # Loss = (1/64) * sum((y_i - Q_predicted)^2)
        loss = self.networks.loss_fn(q_predicted, y_i.detach())
        # .detach() stops gradient flowing through y_i
        # gradient only flows through q_predicted (prediction network)
        # target network never receives gradients

        # ── backpropagation ──────────────────
        self.networks.optimizer.zero_grad()  # clear previous gradients
        loss.backward()                      # compute gradients

        # ── gradient clipping ────────────────
        # prevents exploding gradients
        # clips gradient norm to max value of 1.0
        torch.nn.utils.clip_grad_norm_(
            self.networks.prediction_network.parameters(),
            max_norm=1.0
        )

        # ── SGD weight update ────────────────
        self.networks.optimizer.step()       # update all weights

        # ── target network check ─────────────
        self.training_steps += 1
        updated = self.target_updater.step()

        if updated:
            print(f"    Target network updated at "
                  f"training step {self.training_steps}")

        return loss.item()

    # ─────────────────────────────────────────
    # RUN ONE EPISODE
    # ─────────────────────────────────────────
    def run_episode(self, episode_num):
        """
        Runs one complete episode.
        Collects transitions, trains network.

        Parameters:
            episode_num : int

        Returns:
            episode_reward : float
            episode_loss   : float
            steps          : int
        """
        # ── reset environment ────────────────
        state = self.env.reset()

        episode_reward = 0.0
        episode_losses = []
        done           = False
        step           = 0

        while not done and step < MAX_STEPS:
            step            += 1
            self.total_steps += 1

            # ── select action ────────────────
            action = self.select_action(state)

            # ── environment step ─────────────
            next_state, reward, done, info = self.env.step(action)

            # ── store transition ─────────────
            self.replay_buffer.store(
                state,
                action,
                reward,
                next_state,
                done
            )

            # ── train if buffer ready ────────
            if self.replay_buffer.is_ready():
                loss = self.train_step()
                episode_losses.append(loss)

            # ── log step ─────────────────────
            if step % 50 == 0:
                print(f"  Ep {episode_num:4d} | "
                      f"Step {step:4d} | "
                      f"Action: {info['action_name']:<16} | "
                      f"Outcome: {info['outcome']:<24} | "
                      f"Reward: {reward:+.3f} | "
                      f"Situation: {info['situation']:<15} | "
                      f"ε: {self.epsilon:.3f}")

            episode_reward += reward
            state           = next_state

        # ── average loss this episode ────────
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        return episode_reward, avg_loss, step

    # ─────────────────────────────────────────
    # DECAY EPSILON
    # ─────────────────────────────────────────
    def decay_epsilon(self):
        """
        Reduces epsilon after each episode.
        Agent explores less as training progresses.
        """
        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY
        )

    # ─────────────────────────────────────────
    # SAVE BEST MODEL
    # ─────────────────────────────────────────
    def save_best(self, episode_reward, episode_num):
        """
        Saves model if this episode achieved best reward so far.
        """
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.networks.save('results/models/best_model.pth')
            print(f"  ★ New best reward: {self.best_reward:.3f} "
                  f"at episode {episode_num}")

    # ─────────────────────────────────────────
    # TRAIN — main training loop
    # ─────────────────────────────────────────
    def train(self):
        """
        Main training loop.
        Runs MAX_EPISODES episodes.
        Logs results, saves best model.
        """
        print("\n════════════════════════════════════════")
        print("          DQN TRAINING STARTED          ")
        print("════════════════════════════════════════")
        self.networks.print_architecture()

        # ── log file ─────────────────────────
        log_file = open('results/logs/training_log.txt', 'w')
        log_file.write("episode,reward,avg_loss,steps,epsilon\n")

        for episode in range(1, MAX_EPISODES + 1):

            # ── run episode ──────────────────
            episode_reward, avg_loss, steps = \
                self.run_episode(episode)

            # ── decay epsilon ────────────────
            self.decay_epsilon()

            # ── track results ────────────────
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)

            # ── save best model ──────────────
            self.save_best(episode_reward, episode)

            # ── episode summary ──────────────
            stats = self.env.get_episode_stats()
            print(f"\nEpisode {episode:4d}/{MAX_EPISODES} | "
                  f"Reward: {episode_reward:+8.3f} | "
                  f"Avg Loss: {avg_loss:.5f} | "
                  f"Steps: {steps:3d} | "
                  f"ε: {self.epsilon:.3f} | "
                  f"Buffer: {len(self.replay_buffer):,}")

            episode_summary(
                episode_reward,
                steps,
                stats['outcome_log']
            )

            # ── write to log file ────────────
            log_file.write(
                f"{episode},"
                f"{episode_reward:.4f},"
                f"{avg_loss:.6f},"
                f"{steps},"
                f"{self.epsilon:.4f}\n"
            )
            log_file.flush()

            # ── periodic checkpoint ──────────
            if episode % 100 == 0:
                path = f'results/models/checkpoint_ep{episode}.pth'
                self.networks.save(path)
                self._print_progress(episode)

        # ── training complete ─────────────────
        log_file.close()
        self.networks.save('results/models/final_model.pth')

        print("\n════════════════════════════════════════")
        print("         DQN TRAINING COMPLETE          ")
        print("════════════════════════════════════════")
        print(f"  Total Episodes  : {MAX_EPISODES}")
        print(f"  Total Steps     : {self.total_steps:,}")
        print(f"  Training Steps  : {self.training_steps:,}")
        print(f"  Best Reward     : {self.best_reward:.3f}")
        print(f"  Final Epsilon   : {self.epsilon:.3f}")
        print(f"  Models saved to : results/models/")
        print(f"  Logs saved to   : results/logs/")
        print("════════════════════════════════════════\n")

    # ─────────────────────────────────────────
    # PRINT PROGRESS — every 100 episodes
    # ─────────────────────────────────────────
    def _print_progress(self, episode):
        """
        Prints rolling average reward every 100 episodes.
        Shows training trend.
        """
        last_100 = self.episode_rewards[-100:]
        avg      = np.mean(last_100)
        best     = np.max(last_100)
        worst    = np.min(last_100)

        print(f"\n── Progress at Episode {episode} ────────")
        print(f"  Last 100 Avg Reward : {avg:+.3f}")
        print(f"  Last 100 Best       : {best:+.3f}")
        print(f"  Last 100 Worst      : {worst:+.3f}")
        print(f"  Total Steps         : {self.total_steps:,}")
        print(f"  Training Steps      : {self.training_steps:,}")
        print(f"  Buffer Size         : {len(self.replay_buffer):,}")
        print(f"  Epsilon             : {self.epsilon:.4f}")
        print("────────────────────────────────────────\n")