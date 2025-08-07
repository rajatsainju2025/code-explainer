"""Advanced trainer for code explanation models."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

from .utils import load_config, setup_logging, get_device

logger = logging.getLogger(__name__)


class CodeExplainerTrainer:
    """Advanced trainer for code explanation models with proper evaluation and monitoring."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(
            level=self.config["logging"]["level"],
            log_file=self.config["logging"]["log_file"]
        )
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model(self) -> None:
        """Load tokenizer and model."""
        model_name = self.config["model"]["name"]
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_dataset(self, data_path: str = None) -> DatasetDict:
        """Load and prepare dataset for training.
        
        Args:
            data_path: Path to dataset file (JSON format)
            
        Returns:
            DatasetDict with train/eval splits
        """
        if data_path is None:
            # Use default hardcoded dataset for demo
            data = self._get_default_dataset()
        else:
            with open(data_path, 'r') as f:
                data = json.load(f)
                
        logger.info(f"Loaded {len(data)} examples")
        
        # Split data into train/eval
        train_data, eval_data = train_test_split(
            data, test_size=0.2, random_state=42
        )
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'eval': Dataset.from_list(eval_data)
        })
        
        return dataset_dict
        
    def _get_default_dataset(self) -> List[Dict[str, str]]:
        """Get default demo dataset."""
        return [
            {
                "code": "def add(a, b):\n    return a + b", 
                "explanation": "This is a Python function named 'add' that takes two arguments, 'a' and 'b', and returns their sum."
            },
            {
                "code": "x = [1, 2, 3]\nprint(x[0])", 
                "explanation": "This code initializes a list named 'x' with three numbers. It then prints the first element of the list, which is 1."
            },
            {
                "code": "for i in range(3):\n    print(i)", 
                "explanation": "This is a 'for' loop that iterates three times. It will print the numbers 0, 1, and 2, each on a new line."
            },
            {
                "code": "import math\nprint(math.sqrt(16))", 
                "explanation": "This code imports Python's built-in 'math' module. It then calculates and prints the square root of 16, which is 4.0."
            },
            {
                "code": "def greet(name):\n    return f'Hello, {name}!'", 
                "explanation": "This defines a function 'greet' that takes one argument, 'name'. It returns a formatted string that says hello to the provided name."
            },
            {
                "code": "a = 5\nb = 10\na, b = b, a", 
                "explanation": "This code swaps the values of two variables. Initially, 'a' is 5 and 'b' is 10. After execution, 'a' becomes 10 and 'b' becomes 5."
            },
            {
                "code": "d = {'key': 'value'}\nprint(d['key'])", 
                "explanation": "This initializes a dictionary 'd' with one key-value pair. It then prints the value associated with the key 'key', which is 'value'."
            },
            {
                "code": "class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height", 
                "explanation": "This defines a Python class called 'Rectangle' with an initializer method that takes width and height parameters and stores them as instance attributes."
            },
            {
                "code": "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')", 
                "explanation": "This code uses exception handling to catch a division by zero error. When the error occurs, it prints a helpful message instead of crashing."
            },
            {
                "code": "numbers = [1, 2, 3, 4, 5]\neven_numbers = [n for n in numbers if n % 2 == 0]", 
                "explanation": "This code uses a list comprehension to filter even numbers from a list. It creates a new list containing only the numbers that are divisible by 2."
            }
        ]
        
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess dataset for training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Tokenized dataset ready for training
        """
        def tokenize_function(examples):
            prompt_template = self.config["prompt"]["template"]
            prompt = prompt_template.format(code=examples['code'])
            text = prompt + examples['explanation']
            
            tokenized = self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=self.config["model"]["max_length"]
            )
            
            # Set labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
            
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=False,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
        
    def setup_trainer(self, dataset: DatasetDict) -> None:
        """Setup the Hugging Face trainer.
        
        Args:
            dataset: Preprocessed dataset
        """
        training_config = self.config["training"]
        
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            warmup_steps=training_config["warmup_steps"],
            weight_decay=training_config["weight_decay"],
            logging_dir="./logs",
            logging_steps=training_config["logging_steps"],
            eval_steps=training_config["eval_steps"],
            save_steps=training_config["save_steps"],
            evaluation_strategy=training_config["evaluation_strategy"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            metric_for_best_model=training_config["metric_for_best_model"],
            greater_is_better=training_config["greater_is_better"],
            report_to=["tensorboard"],
            save_safetensors=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
    def train(self, data_path: str = None) -> None:
        """Train the model.
        
        Args:
            data_path: Optional path to training data
        """
        logger.info("Starting training process...")
        
        # Load model and data
        self.load_model()
        dataset = self.load_dataset(data_path)
        tokenized_dataset = self.preprocess_dataset(dataset)
        
        # Setup trainer
        self.setup_trainer(tokenized_dataset)
        
        # Train model
        logger.info("Beginning model training...")
        self.trainer.train()
        logger.info("Training completed!")
        
        # Save model and tokenizer
        output_dir = self.config["training"]["output_dir"]
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        self._save_training_metrics(output_dir)
        
    def _save_training_metrics(self, output_dir: str) -> None:
        """Save training metrics and configuration."""
        metrics_path = Path(output_dir) / "training_metrics.json"
        
        metrics = {
            "config": self.config,
            "final_metrics": self.trainer.state.log_history[-1] if self.trainer.state.log_history else {},
            "device": self.device,
            "model_size": sum(p.numel() for p in self.model.parameters()),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Training metrics saved to {metrics_path}")
