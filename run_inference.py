#!/usr/bin/env python3
"""GRADE inference entry point.

Usage:
    python run_inference.py --model-dir /path/to/model --target-name CCR5_HUMAN --peptides candidates.fasta --output-csv out.csv
"""

from grade.inference import main

if __name__ == "__main__":
    main()
