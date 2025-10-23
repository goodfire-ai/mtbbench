import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Run the benchmark with specific parameters.")
    parser.add_argument("--doctor_model", type=str, default="google/gemma-3-12b-it", help="Name of doctor model to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./data/agent_logs/gemma/", help="Directory to save the result of the run.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset to evaluate on.")
    return parser.parse_args()