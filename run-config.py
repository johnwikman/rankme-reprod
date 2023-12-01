#!/usr/bin/env python3

"""
command that I usually run

mlflow run . -P device="cuda" -P workers=12 -P verbosity=0 -P dataset=cifar10 --env-manager local
"""

import argparse
import os
import subprocess

import src.config

def main():
    parser = argparse.ArgumentParser("Run training configuration")
    parser.add_argument("trainer", choices=src.config.HYPERPARAMETERS.keys())
    parser.add_argument("--device", dest="device", default="cuda", help="Device to run on")
    parser.add_argument("-L", "--local-env-manager", dest="use_local_env_manager", action="store_true")
    args = parser.parse_args()

    for run, params in enumerate(src.config.HYPERPARAMETERS[args.trainer]):
        cmd = [
            "mlflow", "run", os.path.abspath(os.path.dirname(__file__)),
            "-P", f"device={args.device}", # auto choose this
            "-P", f"workers={os.cpu_count()}",
            "-P", "verbosity=0",
            "-P", "dataset=imagenet",
            "-P", "eval_dataset=imagenet",
        ]
        for k,v in params.items():
            cmd += ["-P", f"{k}={v}"]


        if args.use_local_env_manager:
            cmd += ["--env-manager", "local"]

        print(f"running command: \"{' '.join(cmd)}\"")
        with open("run.stdout", "a+") as fout, open("run.stderr", "a+") as ferr:
            ret = subprocess.run(cmd, stdout=fout, stderr=ferr, env=os.environ)
        if ret.returncode != 0:
            print("WARNING: Non-zero return code!")

    print("Done")

if __name__ == "__main__":
    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise EnvironmentError("Environment variable MLFLOW_TRACKING_URI not "
                               "set. Example: set it with\n"
                               "export MLFLOW_TRACKING_URI=\"http://127.0.0.1:8080\"")
    main()
