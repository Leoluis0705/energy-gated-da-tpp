#!/usr/bin/env python3
"""Run a legacy script deterministically with PyTorch 2.6+ load compatibility."""
from __future__ import annotations

import argparse
import os
import random
import runpy
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def install_legacy_torch_load_default() -> None:
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    set_seed(args.seed)
    install_legacy_torch_load_default()
    script_dir = os.path.dirname(os.path.abspath(args.script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv = [args.script] + args.script_args
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
