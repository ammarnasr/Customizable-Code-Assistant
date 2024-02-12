# finetuneLoRa1B.py
import os
import argparse
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from finetuning_datasets import ConstantLengthDataset
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

def main_finetune(main_path, experiment_name, training_data_path):
    

    # Construct the full path for the training data
    training_data_path = os.path.join(main_path, training_data_path)

    # Load the training data
    data = pd.read_csv(training_data_path, low_memory=False)
    # data = data.rename(columns={'code': 'content'})
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    shuffled_data.to_csv(os.path.join(main_path, 'JS_filesShuffled.csv'), index=False)
    dataset = load_dataset("csv", data_files=os.path.join(main_path, 'JS_filesShuffled.csv'))
    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_valid_split['train'],
        'test': test_valid_split['test']
    })

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=2048)

    # Apply tokenization to each split
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_ds = tokenized_datasets["train"]
    valid_ds = tokenized_datasets["validation"]
    train_dataset = ConstantLengthDataset(tokenizer, train_ds, infinite=True, seq_length=2048)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_ds, infinite=False, seq_length=2048)

    # Set up LoraConfig and model
    lora_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias='all', task_type='CAUSAL_LM', target_modules=["qkv_proj", "out_proj", "lm_head", "fc_in", "fc_out"])
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Custom callback for early stopping
    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, early_stopping_patience: int):
            self.early_stopping_patience = early_stopping_patience
            self.best_metric = None
            self.patience_counter = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            current_metric = metrics["eval_loss"]
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    control.should_training_stop = True

    # Update TrainingArguments
    training_args = TrainingArguments(
        output_dir=os.path.join(main_path, experiment_name),
        run_name=experiment_name + '-wandb',
        max_steps=10000,
        evaluation_strategy="steps",
        logging_steps=1,
        learning_rate=5e-5,
        warmup_steps=100,
        lr_scheduler_type='cosine',
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    # Initialize Trainer with early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training
    trainer.train(resume_from_checkpoint=False)

    # Save the best model
    model_path = os.path.join(main_path, experiment_name)
    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
            print(f"Folder '{model_path}' created successfully.")
        except OSError as e:
            print(f"Failed to create folder '{model_path}': {e}")
    else:
        print(f"Folder '{model_path}' already exists.")
    trainer.save_model(model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_path", required=True, help="Main path where the experiment will be conducted")
    parser.add_argument("--experiment_name", required=True, help="Name of the experiment")
    parser.add_argument("--training_data_path", required=True, help="Path to the training data CSV file")
    args = parser.parse_args()
    main_finetune(args.main_path, args.experiment_name, args.training_data_path)
