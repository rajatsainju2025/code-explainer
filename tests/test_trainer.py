import json
from code_explainer.trainer import CodeExplainerTrainer

TINY_CAUSAL = "sshleifer/tiny-gpt2"


def test_trainer_smoke(tmp_path):
    # Minimal config with tiny model
    cfg = {
        "model": {
            "arch": "causal",
            "name": TINY_CAUSAL,
            "max_length": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "torch_dtype": "auto",
        },
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
        "logging": {"level": "ERROR", "log_file": None},
        "data": {},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    trainer = CodeExplainerTrainer(config_path=str(p))
    # Only load/prepare to keep tests fast
    trainer.load_model()
    ds = trainer.load_dataset(None)
    tokenized = trainer.preprocess_dataset(ds)
    trainer.setup_trainer(tokenized)
