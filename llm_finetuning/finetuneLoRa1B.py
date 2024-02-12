import os

# Set the GPU (CUDA device) that you want to use
import torch
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed
)
from finetuning_datasets import ConstantLengthDataset
# model_id = 'Salesforce/codegen-350M-mono'
# tokenizer_id = 'Salesforce/codegen-350M-mono'
lang = 'js'
dataset_id = f'ammarnasr/the-stack-{lang}-clean'
effective_seq_length_train = 2048
effective_seq_length_eval  = 2048
lora_rank = 64
lora_alpha = lora_rank*2
lora_dropout = 0.05
lora_bias = 'all'
lora_task_type = 'CAUSAL_LM'
lora_target_modules = ["qkv_proj", "out_proj", "lm_head", "fc_in", "fc_out"]
dataloader_drop_last = True
max_steps = 10000
eval_steps = 50
save_steps = 100
eval_strategy = 'steps'
logging_steps = 1
learning_rate = 5e-5
warmup_steps = 100
lr_scheduler_type = 'cosine'
gradient_checkpointing = True
gradient_accumulation_steps = 1
per_device_train_batch_size  = 4
per_device_eval_batch_size = 4
fp16 = True

exp_name = f"codegen-{lang}-LoRa-v7-run-1"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

###
import pandas as pd
import numpy as np

# Read the CSV file
data = pd.read_csv("/PATH/JS_files.csv")

# Rename the 'code' column to 'content'
data = data.rename(columns={'code': 'content'})

# Fix the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Shuffle the dataset
shuffled_data = data.sample(frac=1).reset_index(drop=True)

# Save the shuffled data to a new CSV file
shuffled_data.to_csv('/PATH/JS_filesSuffled.csv', index=False)


from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

# Load the dataset from a CSV file
dataset = load_dataset("csv", data_files="/PATH/JS_filesSuffled.csv")

# Split the dataset into train, validation, and test sets
train_test_split = dataset['train'].train_test_split(test_size=0.2)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': test_valid_split['train'],
    'test': test_valid_split['test']
})

effective_seq_length_train = 2048
effective_seq_length_eval = 2048

# The CodeGen tokenizer always during exceuction lacks the padding token if the tokenizer has a padding token and add one if not
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=effective_seq_length_train)

# Apply tokenization to each split
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Check the tokenized dataset
print(tokenized_datasets['train'][0])

#########
train_ds = tokenized_datasets["train"]
valid_ds = tokenized_datasets["validation"]
print(train_ds[0])

train_dataset = ConstantLengthDataset(tokenizer, train_ds, infinite=True, seq_length=effective_seq_length_train)
valid_dataset = ConstantLengthDataset(tokenizer, valid_ds, infinite=False, seq_length=effective_seq_length_eval)
###

#===================================
lora_config = LoraConfig(r = lora_rank, lora_alpha = lora_alpha, lora_dropout = lora_dropout, bias = lora_bias, task_type = lora_task_type, target_modules = lora_target_modules)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#===================================
from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
# model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

from transformers import TrainerCallback

# Custom callback for early stopping
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int):
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_metric = metrics["eval_loss"]  # Change this to your evaluation metric
        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered")
                control.should_training_stop = True
exp_name = "CodeGen2_1BEarlyStopping"
# Update TrainingArguments
training_args = TrainingArguments(
    output_dir=exp_name,
    run_name=exp_name+'-wandb',
    dataloader_drop_last=dataloader_drop_last,
    max_steps=max_steps,
    eval_steps=eval_steps,
    save_steps=save_steps,
    evaluation_strategy=eval_strategy,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=gradient_checkpointing,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    fp16=fp16,
    load_best_model_at_end=True,  # Load best model at the end of training
    metric_for_best_model="eval_loss",  # Change this to your evaluation metric
    greater_is_better=False,  # Set to True if higher metric is better
    save_total_limit=2,  # Save only 2 checkpoints
)

# Initialize Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Set your patience
)

# Start training
trainer.train(resume_from_checkpoint=False)
# After training, save the best model
model_path = "/PATH/CodeGen2_1BEarlyStoppingJS/best_model"
import os

model_path = "/PATH/CodeGen2_1BEarlyStoppingJS/best_model"

# Check if the folder exists
if not os.path.exists(model_path):
    # Create the folder if it doesn't exist
    try:
        os.makedirs(model_path)
        print(f"Folder '{model_path}' created successfully.")
    except OSError as e:
        print(f"Failed to create folder '{model_path}': {e}")
else:
    print(f"Folder '{model_path}' already exists.")

trainer.save_model(model_path)

print(f"Best model saved to {model_path}")