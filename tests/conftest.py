"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure src is importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
