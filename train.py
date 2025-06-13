#!/usr/bin/env python3
"""
Production-level CodeLlama fine-tuning with Unsloth and DeepSpeed
Supports multi-GPU training with memory optimization
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import deepspeed
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    
@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters"""
    r: int = 64
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None

@dataclass
class DataConfig:
    """Configuration for data processing"""
    csv_path: str = "training_data.csv"
    instruction_column: str = "Instruction"
    output_column: str = "Output"
    test_size: float = 0.1
    validation_size: float = 0.1
    max_length: int = 2048
    random_state: int = 42

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_and_split_data(self) -> tuple:
        """Load CSV data and split into train/val/test sets"""
        logger.info(f"Loading data from {self.config.csv_path}")
        
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.config.csv_path}")
            
        df = pd.read_csv(self.config.csv_path)
        
        # Validate columns
        required_cols = [self.config.instruction_column, self.config.output_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
        # Clean data
        df = df.dropna(subset=required_cols)
        df = df[df[self.config.instruction_column].str.strip() != ""]
        df = df[df[self.config.output_column].str.strip() != ""]
        
        logger.info(f"Loaded {len(df)} samples after cleaning")
        
        # Split data
        train_df, temp_df = train_test_split(
            df, test_size=self.config.test_size + self.config.validation_size,
            random_state=self.config.random_state
        )
        
        if self.config.validation_size > 0:
            val_size = self.config.validation_size / (self.config.test_size + self.config.validation_size)
            val_df, test_df = train_test_split(
                temp_df, test_size=1-val_size,
                random_state=self.config.random_state
            )
        else:
            val_df = temp_df
            test_df = pd.DataFrame()
            
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def format_prompts(self, examples: Dict) -> Dict:
        """Format examples into instruction-following format"""
        instructions = examples[self.config.instruction_column]
        outputs = examples[self.config.output_column]
        
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful coding assistant. Generate code based on the given instruction.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
            texts.append(text)
        
        return {"text": texts}
    
    def create_datasets(self) -> DatasetDict:
        """Create Hugging Face datasets"""
        train_df, val_df, test_df = self.load_and_split_data()
        
        datasets = {}
        
        # Create train dataset
        train_dataset = Dataset.from_pandas(train_df)
        train_dataset = train_dataset.map(
            self.format_prompts,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        datasets["train"] = train_dataset
        
        # Create validation dataset
        if len(val_df) > 0:
            val_dataset = Dataset.from_pandas(val_df)
            val_dataset = val_dataset.map(
                self.format_prompts,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            datasets["validation"] = val_dataset
        
        # Create test dataset
        if len(test_df) > 0:
            test_dataset = Dataset.from_pandas(test_df)
            test_dataset = test_dataset.map(
                self.format_prompts,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            datasets["test"] = test_dataset
            
        return DatasetDict(datasets)

class MetricsCallback(TrainerCallback):
    """Custom callback for logging metrics"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Log key metrics
            if "train_loss" in logs:
                logger.info(f"Step {state.global_step}: Train Loss = {logs['train_loss']:.4f}")
            if "eval_loss" in logs:
                logger.info(f"Step {state.global_step}: Eval Loss = {logs['eval_loss']:.4f}")

class CodeLlamaTrainer:
    """Main trainer class"""
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig, data_config: DataConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.data_config = data_config
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Initialize model and tokenizer with Unsloth optimizations"""
        logger.info("Setting up model and tokenizer...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=self.model_config.dtype,
            load_in_4bit=self.model_config.load_in_4bit,
        )
        
        # Setup LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
            random_state=self.lora_config.random_state,
            use_rslora=self.lora_config.use_rslora,
            loftq_config=self.lora_config.loftq_config,
        )
        
        # Configure tokenizer with chat template
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3.1",
        )
        
        logger.info("Model setup completed")
        
    def create_training_args(self, output_dir: str, deepspeed_config: str) -> TrainingArguments:
        """Create training arguments with DeepSpeed configuration"""
        return TrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            output_dir=output_dir,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            deepspeed=deepspeed_config,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to=["tensorboard"],
            run_name="codellama-finetune",
        )
    
    def train(self, output_dir: str = "./results", deepspeed_config: str = "./ds_config.json"):
        """Main training loop"""
        logger.info("Starting training process...")
        
        # Setup model
        self.setup_model()
        
        # Process data
        data_processor = DataProcessor(self.data_config)
        datasets = data_processor.create_datasets()
        
        # Create training arguments
        training_args = self.create_training_args(output_dir, deepspeed_config)
        
        # Initialize trainer with SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            dataset_text_field="text",
            max_seq_length=self.model_config.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences
            args=training_args,
            callbacks=[
                MetricsCallback(),
                EarlyStoppingCallback(early_stopping_patience=3)
            ],
        )
        
        # Enable training on responses only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        
        # Start training
        logger.info("Beginning training...")
        trainer_stats = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        if trainer.state.log_history:
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
                
        logger.info("Training completed successfully!")
        return trainer_stats

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with Unsloth and DeepSpeed")
    
    # Model arguments
    parser.add_argument("--model_name", default="codellama/CodeLlama-7b-Instruct-hf", 
                       help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                       help="Maximum sequence length")
    
    # Data arguments
    parser.add_argument("--csv_path", required=True, 
                       help="Path to CSV training data")
    parser.add_argument("--instruction_column", default="Instruction", 
                       help="Name of instruction column")
    parser.add_argument("--output_column", default="Output", 
                       help="Name of output column")
    parser.add_argument("--test_size", type=float, default=0.1, 
                       help="Test set size")
    parser.add_argument("--validation_size", type=float, default=0.1, 
                       help="Validation set size")
    
    # Training arguments
    parser.add_argument("--output_dir", default="./results", 
                       help="Output directory for model and logs")
    parser.add_argument("--deepspeed_config", default="./ds_config.json", 
                       help="DeepSpeed configuration file")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, 
                       help="LoRA dropout")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    data_config = DataConfig(
        csv_path=args.csv_path,
        instruction_column=args.instruction_column,
        output_column=args.output_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        max_length=args.max_seq_length
    )
    
    # Initialize trainer
    trainer = CodeLlamaTrainer(model_config, lora_config, data_config)
    
    # Start training
    try:
        trainer.train(args.output_dir, args.deepspeed_config)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
