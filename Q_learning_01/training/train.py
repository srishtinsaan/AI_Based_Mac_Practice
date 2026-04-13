from src.agent.q_agent import QAgent
from src.env.switch_env import SwitchEnv

def run_training(data_path):
    env = SwitchEnv(data_path)
    agent = QAgent(states=env.encoder.total_states())
    
    for ep in range(2000):
        state = env.reset() # This line will now work
        for _ in range(10):
            action = agent.choose_action(state)
            next_s, reward, _ = env.step(action)
            agent.update(state, action, reward, next_s)
            state = next_s
        agent.decay_epsilon()
    return agent, env.encoder