import json
from code_explainer.trainer import CodeExplainerTrainer


def test_trainer_smoke(tmp_path):
    # Minimal config
    cfg = {
        "model": {"name": "distilgpt2", "max_length": 64, "temperature": 0.7, "top_p": 0.9, "top_k": 50},
        "training": {
            "output_dir": str(tmp_path / "out"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "logging_steps": 1,
            "eval_steps": 1,
            "save_steps": 1,
            "load_best_model_at_end": False,
        },
        "prompt": {"template": "Explain this Python code:\n{code}\nExplanation:"},
        "logging": {"level": "INFO", "log_file": None},
        "data": {}
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    trainer = CodeExplainerTrainer(config_path=str(p))
    # Only load/prepare to keep tests fast
    trainer.load_model()
    ds = trainer.load_dataset(None)
    tokenized = trainer.preprocess_dataset(ds)
    trainer.setup_trainer(tokenized)
