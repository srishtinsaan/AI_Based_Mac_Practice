from training.train import run_training
import pandas as pd

if __name__ == "__main__":
    DATA_FILE = "data/raw/traffic_scenarios.csv"
    
    print("Training Agent with Epsilon Decay...")
    agent, rewards, eps = run_training(DATA_FILE)
    
    # Final Q-Table Output 
    q_df = pd.DataFrame(
        agent.q_table, 
        index=['S1_Normal', 'S2_Moderate', 'S3_Critical'],
        columns=['LEARN', 'EVICT', 'FLOOD', 'ASSIGN', 'BLOCK', 'UPDATE']
    )
    
    print("\nLearned Policy (Final Q-Table):")
    print(q_df)
    print(f"\nFinal Epsilon: {agent.epsilon:.4f}") # Should be near epsilon_min