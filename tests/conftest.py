# Ensure the 'code_explainer' package is importable by adding the src directory to sys.path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Add both the src directory (for `code_explainer`) and the repo root (for `src.code_explainer`)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
