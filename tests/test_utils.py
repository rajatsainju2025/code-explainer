from code_explainer.utils import get_device, load_config


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    cfg = load_config(str(p))
    assert cfg["a"] == 1


def test_get_device_returns_string():
    dev = get_device()
    assert dev in {"cuda", "mps", "cpu"}
