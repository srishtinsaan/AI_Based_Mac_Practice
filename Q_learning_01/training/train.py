from src.agent.q_agent import QAgent
from src.env.switch_env import SwitchEnv

def run_training(data_path, total_episodes=2000):
    env = SwitchEnv(data_path)
    # Start with high epsilon (1.0) to explore [cite: 26]
    agent = QAgent(epsilon_start=1.0, decay_rate=0.99) 
    
    rewards_history = []
    epsilons = []

    for ep in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(20): # Steps per episode [cite: 14]
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state) # [cite: 17, 220]
            
            state = next_state
            episode_reward += reward
        
        # Decay epsilon at the end of each episode
        agent.decay_epsilon()
        
        rewards_history.append(episode_reward)
        epsilons.append(agent.epsilon)

    return agent, rewards_history, epsilons