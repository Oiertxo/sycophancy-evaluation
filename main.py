from src.dataset_test import run_grouped_test
from src.judge import run_grouped_judge
from src.metrics_engine import calculate_metrics

def main():
    print("\n--- Generating model responses ---")
    run_grouped_test()
    
    print("\n--- LLM-as-a-Judge ---")
    run_grouped_judge()
    
    print("\n--- Calculating KPIs ---")
    calculate_metrics("data/results/revised.jsonl")
    
    print("\nProcess completed")

if __name__ == "__main__":
    main()