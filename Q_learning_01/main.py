from training.train import run_training
import pandas as pd

if __name__ == "__main__":
    DATA_FILE = "data/raw/traffic_scenarios.csv"
    agent, encoder = run_training(DATA_FILE)
    
    q_df = pd.DataFrame(agent.q_table, 
                        index=[f"Scenario_{i}" for i in range(encoder.total_states())],
                        columns=['LEARN', 'EVICT', 'FLOOD', 'ASSIGN', 'BLOCK', 'UPDATE'])
    print("\nTraining Complete. Final Learned Policy:")
    print(q_df.round(3))