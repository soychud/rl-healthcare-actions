#!/bin/bash
set -e
pip install torch numpy pandas pyarrow scikit-learn scipy
python3 -m pytest tests/ -v
