# cli.py
import argparse
from finetuneLora import main_finetune
import sys
print(sys.path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script with CLI for LLM using LoRa")
    parser.add_argument("--main_path", required=True, help="Main path where the experiment will be conducted")
    parser.add_argument("--experiment_name", required=True, help="Name of the experiment")
    parser.add_argument("--training_data_path", required=True, help="Path to the training data CSV file")
    args = parser.parse_args()
    
    main_finetune(args.main_path, args.experiment_name, args.training_data_path)

if __name__ == "__main__":
    main()


