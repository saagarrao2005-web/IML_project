"""
Root entrypoint for Streamlit Web Dashboard.

Enables running `streamlit run app.py` from the project root directory.
Imports and runs the main() function from model_development/app.py.
"""
from pathlib import Path
import sys

MODEL_DEV_DIR = Path(__file__).resolve().parent / "model_development"
if str(MODEL_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DEV_DIR))

# Import the actual Streamlit app module and run it
from app import main

main()
