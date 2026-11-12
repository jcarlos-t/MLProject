#!/usr/bin/env python3
"""Run from any working directory; the test is only used after freeze()."""
import argparse
import os

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

from mlcore.experiment import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'cv', 'feedback', 'freeze', 'evaluate', 'all'], default='all')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    run(args.stage, args.jobs)
