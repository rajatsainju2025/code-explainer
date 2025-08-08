"""Advanced trainer for code explanation models."""

import logging
from typing import Dict, Any, Optional, List, cast
from pathlib import Path
import json

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from sklearn.model_selection import train_test_split

from .utils import load_config, setup_logging, get_device
from .data.datasets import build_dataset_dict, DatasetConfig
from .augment.simple import augment_dataset
from .metrics.evaluate import compute_bleu, compute_rouge_l

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
        
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model: Optional[PreTrainedModel] = None
        self.trainer: Optional[Trainer] = None
        
    def load_model(self) -> None:
        """Load tokenizer and model."""
        model_name = self.config["model"]["name"]
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        assert self.tokenizer is not None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_dataset(self, data_path: Optional[str] = None) -> DatasetDict:
        """Load and prepare dataset for training.
        
        Args:
            data_path: Path to dataset file (JSON format)
            
        Returns:
            DatasetDict with train/eval splits
        """
        if data_path is None:
            # Use config files if available
            dc = self.config.get("data", {})
            ds_dict = build_dataset_dict(DatasetConfig(
                train_file=dc.get("train_file"),
                eval_file=dc.get("eval_file"),
                test_file=dc.get("test_file"),
                max_examples=dc.get("max_examples"),
            ))
        else:
            with open(data_path, 'r') as f:
                ds_list = json.load(f)
            tr, ev = train_test_split(ds_list, test_size=0.2, random_state=42)
            ds_dict = {"train": tr, "eval": ev}
        
        # Optional augmentation
        if self.config.get("data", {}).get("augment_ratio", 0):
            ratio = float(self.config["data"]["augment_ratio"])  # type: ignore[arg-type]
            ds_dict["train"] = augment_dataset(ds_dict["train"], ratio=ratio)
        
        logger.info(f"Loaded {len(ds_dict['train'])} train and {len(ds_dict.get('eval', []))} eval examples")
        
        # Convert to HF datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(ds_dict['train']),
            'eval': Dataset.from_list(ds_dict.get('eval', ds_dict['train'][: max(1, int(0.2*len(ds_dict['train']))) ]))
        })
        
        return dataset_dict

    def _get_prompted_text(self, code: str, explanation: str) -> str:
        prompt_template = self.config["prompt"]["template"]
        return prompt_template.format(code=code) + explanation

    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess dataset for training."""
        assert self.tokenizer is not None, "Tokenizer not loaded"
        tok = self.tokenizer
        
        def tokenize_function(examples):
            text = self._get_prompted_text(examples['code'], examples['explanation'])
            encoded = tok(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.config["model"]["max_length"],
            )
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, list):
                encoded["labels"] = list(input_ids)
            else:
                encoded["labels"] = input_ids
            return encoded
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=False,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
        
    def setup_trainer(self, dataset: DatasetDict) -> None:
        """Setup the Hugging Face trainer."""
        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"
        
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
            save_steps=training_config["save_steps"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            report_to=["tensorboard"],
            save_safetensors=True,
        )
        
        tok = self.tokenizer
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            pred_texts: List[str] = []
            ref_texts: List[str] = []
            try:
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                for p, l in zip(predictions, labels):
                    try:
                        pred_texts.append(tok.decode(list(p), skip_special_tokens=True))  # type: ignore[arg-type]
                        ref_texts.append(tok.decode(list(l), skip_special_tokens=True))    # type: ignore[arg-type]
                    except Exception:
                        pred_texts.append("")
                        ref_texts.append("")
            except Exception:
                pass
            return {
                "bleu": compute_bleu(ref_texts, pred_texts),
                "rougeL": compute_rouge_l(ref_texts, pred_texts),
            }
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics,
        )
        
    def train(self, data_path: Optional[str] = None) -> None:
        """Train the model.
        
        Args:
            data_path: Optional path to training data
        """
        logger.info("Starting training process...")
        
        self.load_model()
        dataset = self.load_dataset(data_path)
        tokenized_dataset = self.preprocess_dataset(dataset)
        self.setup_trainer(tokenized_dataset)
        
        assert self.trainer is not None, "Trainer not set up"
        logger.info("Beginning model training...")
        self.trainer.train()
        logger.info("Training completed!")
        
        output_dir = self.config["training"]["output_dir"]
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        cast(PreTrainedTokenizerBase, self.tokenizer).save_pretrained(output_dir)
        
        self._save_training_metrics(output_dir)
        
    def _save_training_metrics(self, output_dir: str) -> None:
        assert self.trainer is not None and self.model is not None
        metrics_path = Path(output_dir) / "training_metrics.json"
        
        metrics = {
            "config": self.config,
            "final_metrics": getattr(self.trainer.state, "log_history", [])[-1] if getattr(self.trainer, "state", None) and self.trainer.state.log_history else {},
            "device": self.device,
            "model_size": sum(p.numel() for p in self.model.parameters()),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_path}")
