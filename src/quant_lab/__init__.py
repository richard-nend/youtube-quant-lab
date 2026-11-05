"""quant_lab: every number quoted in the series must be produced by code in this package."""

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]          # .../trading/quant_lab
PROJECT_ROOT = LAB_ROOT.parent                         # .../trading
CACHE_DIR = LAB_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "src" / "data" / "results"  # consumed by Remotion components
