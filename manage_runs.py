#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from datetime import datetime

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_MLRUNS = os.path.join(SCRIPT_DIR, "mlruns")
DEFAULT_MLARTIFACTS = os.path.join(SCRIPT_DIR, "mlartifacts")

DEFAULT_TARGET = os.path.join(SCRIPT_DIR, "get_runs_target")

def datetype(s):
    formats = [
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d_%H%M%S",
    ]
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
        except Exception as e:
            dt = None
        if dt is not None:
            break
    if dt is None:
        raise ValueError(f"Unknown date format for \"{s}\". Known formats are {formats}.")
    return dt


def collect(hashes, target, mlruns_dir, mlartifacts_dir, **kwargs):
    print("collecting...")
    os.makedirs(target, exist_ok=True)
    target_mlruns = os.path.join(target, "mlruns", "0")
    target_mlartifacts = os.path.join(target, "mlartifacts", "0")

    def copydir(src, dst):
        if os.path.exists(dst):
            print(f"Warning: {dst} already exists. Removing it before copying the new files there.")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    for hash in hashes:
        src_run = os.path.join(mlruns_dir, hash)
        dst_run = os.path.join(target_mlruns, hash)
        src_artifact = os.path.join(mlartifacts_dir, hash)
        dst_artifact = os.path.join(target_mlartifacts, hash)
        os.makedirs(target_mlruns, exist_ok=True)
        copydir(src_run, dst_run)
        if os.path.exists(src_artifact):
            os.makedirs(target_mlartifacts, exist_ok=True)
            copydir(src_artifact, dst_artifact)

def remove(hashes, mlruns_dir, mlartifacts_dir, **kwargs):
    for hash in hashes:
        src_run = os.path.join(mlruns_dir, hash)
        src_run_name_path = os.path.join(src_run, "tags", "mlflow.runName")
        src_artifact = os.path.join(mlartifacts_dir, hash)
        has_artifact = os.path.isdir(src_artifact)
        msg = f"Remove run {hash}"
        if os.path.isfile(src_run_name_path):
            with open(src_run_name_path, "r") as f:
                msg += " (" + f.read() + ")"
        if has_artifact:
            msg += " and its artifact"
        msg += "?"
        choice = input(msg + " [y/N] ").lower().strip()
        if choice == "y":
            print(f"(Removing {hash})")
            shutil.rmtree(src_run)
            if has_artifact:
                shutil.rmtree(src_artifact)
        else:
            print(f"(Not removing)")


def main():
    MODES = {
        "collect": collect,
        "remove": remove,
    }
    parser = argparse.ArgumentParser("Extract mlflow runs and their artifacts")
    parser.add_argument("mode", choices=MODES.keys(), help="Action to perform")
    parser.add_argument("--mlruns", dest="mlruns", type=str, default=DEFAULT_MLRUNS,
                        help="Default directory for mlruns")
    parser.add_argument("--mlartifacts", dest="mlartifacts", type=str, default=DEFAULT_MLARTIFACTS,
                        help="Default directory for mlartifacts")
    parser.add_argument("--target", dest="target", type=str, default=DEFAULT_TARGET,
                        help="Default target directory")
    parser.add_argument("-S", "--suffix", dest="suffix", type=str, default=None, help="Suffix on the run name")
    parser.add_argument("--mindate", dest="mindate", type=datetype, default=None, help="minimum date")
    parser.add_argument("--maxdate", dest="maxdate", type=datetype, default=None, help="maximum date")
    args = parser.parse_args()

    mlruns_dir = os.path.join(args.mlruns, "0")
    mlartifacts_dir = os.path.join(args.mlartifacts, "0")

    print("args.mlruns:", args.mlruns)
    print("args.mlartifacts:", args.mlartifacts)
    print("args.target:", args.target)
    print("args.suffix:", args.suffix)
    print("args.mindate:", args.mindate)
    print("args.maxdate:", args.maxdate)

    if not os.path.isdir(mlruns_dir):
        raise ValueError(f"Could not find mlruns dir {mlruns_dir}")
    if not os.path.isdir(mlartifacts_dir):
        raise ValueError(f"Could not find mlartifacts dir {mlartifacts_dir}")

    hashes = []
    for f in os.listdir(mlruns_dir):
        valid = True
        valid &= bool(len(f) == 32)
        for c in f:
            valid &= bool(c in "0123456789abcdef")
        if not valid:
            continue

        fpath = os.path.join(mlruns_dir, f)
        mtime = os.path.getmtime(fpath)
        runname_path = os.path.join(fpath, "tags", "mlflow.runName")

        if args.suffix is not None and os.path.isfile(runname_path):
            with open(runname_path, "r") as fh:
                valid &= fh.read().strip().endswith(args.suffix)

        if args.mindate is not None:
            valid &= bool(mtime >= args.mindate.timestamp())
        if args.maxdate is not None:
            valid &= bool(mtime <= args.maxdate.timestamp())

        if valid:
            hashes.append(f)

    if len(hashes) == 0:
        print("No runs found")
        return

    MODES[args.mode](hashes=hashes, mlruns_dir=mlruns_dir, mlartifacts_dir=mlartifacts_dir, target=args.target)

if __name__ == "__main__":
    main()
    print("Done")
