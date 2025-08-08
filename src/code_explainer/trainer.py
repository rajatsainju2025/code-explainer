"""Advanced trainer for code explanation models."""

import logging
from typing import Dict, Any, Optional, List, cast
from pathlib import Path
import json
import warnings

import torch
# Runtime import of datasets with alias to avoid static type issues
try:
    import datasets as hf_datasets  # type: ignore
except Exception:  # pragma: no cover
    hf_datasets = None  # type: ignore

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
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
        
        self.arch: str = self.config.get("model", {}).get("arch", "causal")
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model: Optional[PreTrainedModel] = None
        self.trainer: Optional[Trainer] = None
        
    def load_model(self) -> None:
        """Load tokenizer and model based on architecture and config flags."""
        model_name = self.config["model"]["name"]
        dtype_str = self.config.get("model", {}).get("torch_dtype", "auto")
        load_in_8bit = bool(self.config.get("model", {}).get("load_in_8bit", False))

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(str(dtype_str).lower(), "auto")

        if load_in_8bit and self.device in ("cpu", "mps"):
            warnings.warn("8-bit quantization is not supported on CPU/MPS. Disabling load_in_8bit.")
            load_in_8bit = False

        logger.info(f"Loading model '{model_name}' (arch={self.arch}, dtype={torch_dtype}, 8bit={load_in_8bit})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        assert self.tokenizer is not None
        # Ensure pad token exists
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Use eos as pad if missing
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[assignment]

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }
        if load_in_8bit:
            # device_map helps place modules automatically
            model_kwargs.update({"load_in_8bit": True, "device_map": "auto"})

        if self.arch == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            # For causal models, set pad token id to eos if needed
            assert self.model is not None
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id  # type: ignore[assignment]
        
        # Move to device when not using 8bit (8bit uses device_map)
        if not load_in_8bit:
            assert self.model is not None
            cast(Any, self.model).to(self.device)
        
    def load_dataset(self, data_path: Optional[str] = None) -> Any:
        """Load and prepare dataset for training.
        
        Args:
            data_path: Path to dataset file (JSON format)
            
        Returns:
            HF DatasetDict with train/eval splits
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
        
        if hf_datasets is None:
            raise ImportError("The 'datasets' package is required. Please install with `pip install datasets`.")
        
        # Convert to HF datasets
        dataset_dict = hf_datasets.DatasetDict({
            'train': hf_datasets.Dataset.from_list(ds_dict['train']),
            'eval': hf_datasets.Dataset.from_list(ds_dict.get('eval', ds_dict['train'][: max(1, int(0.2*len(ds_dict['train']))) ]))
        })
        
        return dataset_dict

    def _get_prompted_text(self, code: str, explanation: str) -> str:
        prompt_template = self.config["prompt"]["template"]
        return prompt_template.format(code=code) + explanation

    def preprocess_dataset(self, dataset: Any) -> Any:
        """Preprocess dataset for training with causal or seq2seq tokenization."""
        assert self.tokenizer is not None, "Tokenizer not loaded"
        tok = self.tokenizer

        max_source_len = int(self.config["model"]["max_length"])  # cap for inputs
        max_tgt_len = int(self.config.get("prompt", {}).get("max_explanation_length", max_source_len))

        if self.arch == "seq2seq":
            def tokenize_function_seq2seq(examples):
                source_text = self.config["prompt"]["template"].format(code=examples['code'])
                target_text = examples['explanation']
                # Tokenize source
                model_inputs = tok(
                    source_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_source_len,
                )
                # Tokenize labels/targets
                with tok.as_target_tokenizer():
                    labels = tok(
                        target_text,
                        truncation=True,
                        padding="max_length",
                        max_length=max_tgt_len,
                    )
                label_ids = labels["input_ids"]
                # Replace pad token ids in labels with -100
                pad_id = tok.pad_token_id
                if not isinstance(pad_id, int):
                    eos = tok.eos_token_id
                    pad_id = eos if isinstance(eos, int) else 0
                if isinstance(label_ids, list):
                    label_ids = [(-100 if t == pad_id else t) for t in label_ids]
                model_inputs["labels"] = label_ids
                return model_inputs

            logger.info("Tokenizing dataset (seq2seq)...")
            tokenized_dataset = dataset.map(
                tokenize_function_seq2seq,
                batched=False,
                desc="Tokenizing (seq2seq)"
            )
        else:
            def tokenize_function_causal(examples):
                # Build prompt and target, mask prompt tokens in labels
                prompt = self.config["prompt"]["template"].format(code=examples['code'])
                target = examples['explanation']
                full_text = prompt + target
                encoded = tok(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_source_len,
                )
                # Compute prompt token length
                prompt_ids = tok(
                    prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=max_source_len,
                )["input_ids"]
                if isinstance(prompt_ids, list):
                    non_pad = tok.pad_token_id
                    if not isinstance(non_pad, int):
                        eos = tok.eos_token_id
                        non_pad = eos if isinstance(eos, int) else -1
                    prompt_len = len([t for t in prompt_ids if t != non_pad])
                else:
                    prompt_len = int(getattr(prompt_ids, 'shape', [0])[0])
                input_ids = encoded.get("input_ids")
                # Initialize labels from input_ids
                labels = list(input_ids) if isinstance(input_ids, list) else input_ids
                # Mask prompt part and padding with -100
                if isinstance(labels, list):
                    pad_id = tok.pad_token_id
                    if not isinstance(pad_id, int):
                        eos = tok.eos_token_id
                        pad_id = eos if isinstance(eos, int) else -1
                    for i in range(len(labels)):
                        if i < prompt_len or labels[i] == pad_id:
                            labels[i] = -100
                encoded["labels"] = labels
                return encoded

            logger.info("Tokenizing dataset (causal)...")
            tokenized_dataset = dataset.map(
                tokenize_function_causal,
                batched=False,
                desc="Tokenizing (causal)"
            )
        
        return tokenized_dataset
        
    def setup_trainer(self, dataset: Any) -> None:
        """Setup the Hugging Face trainer with config-driven flags."""
        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"
        
        training_config = self.config["training"]
        fp16 = bool(training_config.get("fp16", False))
        bf16 = bool(training_config.get("bf16", False))
        gradient_checkpointing = bool(training_config.get("gradient_checkpointing", False))
        torch_compile = bool(training_config.get("torch_compile", False))

        if self.arch == "seq2seq":
            args = Seq2SeqTrainingArguments(
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
                fp16=fp16,
                bf16=bf16,
                gradient_checkpointing=gradient_checkpointing,
                torch_compile=torch_compile,
                predict_with_generate=True,
                generation_max_length=int(self.config["model"]["max_length"]),
            )
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, label_pad_token_id=-100)
            trainer_cls = Seq2SeqTrainer
        else:
            args = TrainingArguments(
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
                fp16=fp16,
                bf16=bf16,
                gradient_checkpointing=gradient_checkpointing,
                torch_compile=torch_compile,
            )
            data_collator = None
            trainer_cls = Trainer
        
        tok = self.tokenizer

        def _replace_ignore_tokens(label_ids: List[int], pad_id: int) -> List[int]:
            return [pad_id if t == -100 else t for t in label_ids]

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # Convert logits to ids for causal models
            if self.arch != "seq2seq" and getattr(predictions, "ndim", 0) == 3:
                try:
                    predictions = predictions.argmax(axis=-1)
                except Exception:
                    pass
            # Handle tuple predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            # Convert to list for decoding
            try:
                pred_texts: List[str] = tok.batch_decode(predictions, skip_special_tokens=True)  # type: ignore[arg-type]
            except Exception:
                # Fallback per-example
                pred_texts = []
                for p in predictions:
                    try:
                        pred_texts.append(tok.decode(list(p), skip_special_tokens=True))
                    except Exception:
                        pred_texts.append("")
            # Replace -100 in labels before decoding
            pad = tok.pad_token_id
            if not isinstance(pad, int):
                eos = tok.eos_token_id
                pad = eos if isinstance(eos, int) else 0
            label_texts: List[str] = []
            for l in labels:
                try:
                    l_list = list(l)
                    l_proc = _replace_ignore_tokens(l_list, pad)
                    label_texts.append(tok.decode(l_proc, skip_special_tokens=True))
                except Exception:
                    label_texts.append("")
            return {
                "bleu": compute_bleu(label_texts, pred_texts),
                "rougeL": compute_rouge_l(label_texts, pred_texts),
            }
        
        self.trainer = trainer_cls(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
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
