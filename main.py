import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vgd.orchestration_layer.main import run_demo

if __name__ == "__main__":
    run_demo()
