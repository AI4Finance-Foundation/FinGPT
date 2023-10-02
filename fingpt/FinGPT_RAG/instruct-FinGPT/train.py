# Acknowledgement
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Run with default settings:
$ python3 train.py

"""

import argparse
import warnings
import subprocess
import os
import datetime
import time

step_dirs = {
    1: "training/supervised_finetuning",
}
model_type = {1: "actor"}
dse_url = "https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor-model",
        type=lambda x: x.replace("facebook/opt-", "").replace("decapoda-research/", "").replace("-hf", ""),
        default="1.3b",
        choices=("1.3b", "6.7b", "13b", "66b", "llama-13b", "sent-1.3b", "sent-llama-7b", "sent-llama2-7b"),
        help="Which facebook/opt-* model to use for Actor (step 1)",
    )
    parser.add_argument(
        "--actor-zero-stage",
        type=str,
        default="",
        choices=("", "0", "1", "2", "3"),
        help="ZeRO stage for step 1 (Actor) training",
    )
    parser.add_argument(
        "--output-dir",
        type=lambda x: os.path.abspath(x),
        default="./output",
        help="Directory for output of each step",
    )
    parser.add_argument(
        "--deployment-type",
        type=str,
        default="single_gpu",
        choices=("single_gpu", "single_node", "multi_node"),
        help="Number of GPUs to run the actor/reward models on",
    )
    args = parser.parse_args()

    if args.actor_zero_stage != "":
        warnings.warn(
            "Non-default zero stages may result in OOM errors or worse performance."
        )

    return args


def get_model_size(args, step_num):
    return getattr(args, f"{model_type[step_num]}_model")


def get_zero_stage(args, step_num):
    return getattr(args, f"{model_type[step_num]}_zero_stage")


def get_output_dir(args, step_num):
    model_size = get_model_size(args, step_num)
    output_dir = os.path.join(args.output_dir,
                              f"{model_type[step_num]}-models",
                              f"{model_size}")
    return output_dir


def get_script(args, step_num):
    model_size = get_model_size(args, step_num)
    script = os.path.join(
        os.getcwd(),
        step_dirs[step_num],
        "training_scripts",
        args.deployment_type,
        f"run_{model_size}.sh",
    )
    assert os.path.isfile(
        script
    ), f"{script} does not exist.\n\n Use examples in {os.path.dirname(script)} as a template."

    return script


def verify_model(args, step_num):
    output_dir = get_output_dir(args, step_num)
    model_size = get_model_size(args, step_num)
    model_file = os.path.join(output_dir, "pytorch_model.bin")
    if not os.path.isfile(model_file):
        error_str = f"Step {step_num} model has not been trained. Train it with:\n"
        error_str += f"python3 train.py --step {step_num}"
        error_str += f" --{model_type[step_num]}-model {model_size}"
        raise RuntimeError(error_str)


def get_cmd(args, step_num):
    output_dir = get_output_dir(args, step_num)
    script = get_script(args, step_num)

    zero_stage = get_zero_stage(args, step_num)
    cmd = f"bash {script} {output_dir} {zero_stage}"

    return cmd


def launch_cmd(args, step_num, cmd):
    working_dir = step_dirs[step_num]
    print(f"Running:\n{cmd}")
    p = subprocess.Popen(cmd, cwd=working_dir, shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('\n\n'.join((
            f"Step {step_num} exited with non-zero status {p.returncode}",
            f"Launch command: {cmd}",
            f"Log output: {os.path.join(get_output_dir(args, step_num), 'training.log')}",
            f"Please see our tutorial at {dse_url}{step_dirs[step_num]}",
            "Please check that you have installed our requirements: `pip install -r requirements.txt`",
            f"If you are seeing an OOM error, try modifying {get_script(args, step_num)}:",
            "  - Reduce `--per_device_*_batch_size`",
            "  - Increase `--zero_stage {0,1,2,3}` on multi-gpu setups",
            "  - Enable `--gradient_checkpointing` or `--only_optimizer_lora`"
        )))


def main(args):
    start_time = time.time()
    args.step = 1
    step_num = 1

    print(f"---=== Running Step Instruction Tuning ===---")
    step_start_time = time.time()

    cmd = get_cmd(args, step_num)
    print(cmd)
    launch_cmd(args, step_num, cmd)

    step_time = int(time.time() - start_time)
    time_str = str(datetime.timedelta(seconds=step_time))
    print(f"---=== Finished Step Instruction Tuning in {time_str} ===---")

    total_time = int(time.time() - start_time)
    time_str = str(datetime.timedelta(seconds=total_time))


if __name__ == "__main__":
    args = parse_args()
    main(args)
