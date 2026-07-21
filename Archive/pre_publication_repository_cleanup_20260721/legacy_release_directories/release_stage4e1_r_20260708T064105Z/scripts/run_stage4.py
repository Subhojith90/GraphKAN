#!/usr/bin/env python3
"""Run the frozen Stage 4 N=8 five-seed replication using the validated Stage 3F engine."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('run_stage3f.py')), run_name='__main__')
