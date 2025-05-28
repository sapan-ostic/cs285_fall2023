"""
Usage:
    python cs285/scripts/run_tasks.py --mode run [--do_dagger]
        Runs behavioral cloning or DAgger experiments for all environments.

    python cs285/scripts/run_tasks.py --mode report [--sort_by name|percent]
        Prints summary tables and plots Eval_AverageReturn vs Iteration for experiments in data/q1 and data/q2.

    --do_dagger: If set with --mode run, runs DAgger experiments (for q2).
    --sort_by: Sorts report tables by name or Eval/Train % (default: name).
    --log_video: logs videos during training.
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from tabulate import tabulate
import argparse
import subprocess
import itertools

scalars = {
    'Loss/train': 1,
    'Eval_BatchSize': 0,
    'Eval_AverageReturn': 1,
    'Eval_StdReturn': 1,
    'Eval_MaxReturn': 0,
    'Eval_MinReturn': 0,
    'Eval_AverageEpLen': 0,
    'Train_BatchSize': 0,
    'Train_AverageReturn': 1,
    'Train_StdReturn': 1,
    'Train_MaxReturn': 0,
    'Train_MinReturn': 0,
    'Train_AverageEpLen': 0,
    'Training_Loss': 0,
    'Train_EnvstepsSoFar': 0,
    'TimeSinceStart': 0,
    'Initial_DataCollection_AverageReturn': 0
}


def run_tasks(do_dagger=False, log_video=False):
    n_iter = 10 if do_dagger else 1
    log_dir = "data/q1/" if not do_dagger else "data/q2/"

    # Associate env_name with its expert policy and expert data
    envs = [
        {
            "env_name": "Ant-v4",
            "expert_policy_file": "cs285/policies/experts/Ant.pkl",
            "expert_data": "cs285/expert_data/expert_data_Ant-v4.pkl",
        },
        {
            "env_name": "Walker2d-v4",
            "expert_policy_file": "cs285/policies/experts/Walker2d.pkl",
            "expert_data": "cs285/expert_data/expert_data_Walker2d-v4.pkl",
        },
        {
            "env_name": "HalfCheetah-v4",
            "expert_policy_file": "cs285/policies/experts/HalfCheetah.pkl",
            "expert_data": "cs285/expert_data/expert_data_HalfCheetah-v4.pkl",
        },
        {
            "env_name": "Hopper-v4",
            "expert_policy_file": "cs285/policies/experts/Hopper.pkl",
            "expert_data": "cs285/expert_data/expert_data_Hopper-v4.pkl",
        },
    ]

    param_grid = {
        "learning_rate": [5e-3],
        "n_iter": [n_iter],
        "steps": [1000, 5000],
        "n_layers": [2],
        "size": [64],
        "eval_batch_size": [5000],
    }

    # Generate all combinations for hyperparameters (excluding envs)
    keys, values = zip(*param_grid.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for env in envs:
        for params in hp_combinations:
            exp_prefix = "dagger_" if do_dagger else "bc_"
            exp_name = (
                f"{exp_prefix}{env['env_name']}_"
                f"{params['steps']}_steps"
            )
            cmd = [
                "python", "cs285/scripts/run_hw1.py",
                "--env_name", env["env_name"],
                "--learning_rate", str(params["learning_rate"]),
                "--n_iter", str(params["n_iter"]),
                "--num_agent_train_steps_per_iter", str(params["steps"]),
                "--n_layers", str(params["n_layers"]),
                "--size", str(params["size"]),
                "--eval_batch_size", str(params["eval_batch_size"]),
                "--expert_policy_file", env["expert_policy_file"],
                "--expert_data", env["expert_data"],
                "--exp_name", exp_name,
                "--log_dir", log_dir + exp_name,
            ]

            # Set video log frequency based on log_video flag
            if log_video:
                cmd.extend(["--video_log_freq", "1"])
            else:
                cmd.extend(["--video_log_freq", "-1"])

            if do_dagger:
                cmd.append("--do_dagger")

            print("Running:", " ".join(cmd))
            subprocess.run(cmd)


def plot_table(log_base_dir="data/q1", sort_by="name"):
    # Path to your TensorBoard log directory
    log_dirs = [os.path.join(log_base_dir, d) for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d))]

    # Collect results for all log dirs
    results = {}
    for log_dir in log_dirs:
        print(f"Processing log directory: {log_dir}")

        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        values = {}
        for tag, enabled in scalars.items():
            if enabled:
                try:
                    scalars_list = event_acc.Scalars(tag)
                    value = scalars_list[-1].value if scalars_list else 'No data'
                except KeyError:
                    value = 'Not found'
                values[tag] = value
        results[os.path.basename(log_dir)] = values

    # Prepare transposed table
    tags = [tag for tag, enabled in scalars.items() if enabled]
    headers = ["Log Directory"] + tags + ["Eval/Train %"]
    table = []
    for log_name in results.keys():
        tag_values = results[log_name]
        row = [log_name]
        for tag in tags:
            row.append(tag_values.get(tag, 'N/A'))
        # Calculate Eval/Train %
        eval_ret = tag_values.get('Eval_AverageReturn', None)
        train_ret = tag_values.get('Train_AverageReturn', None)
        try:
            percent = (float(eval_ret) / float(train_ret)) * 100 if train_ret not in [None, 0, 'N/A', 'No data', 'Not found'] else 'N/A'
            percent_val = percent if isinstance(percent, float) else None
            percent = f"{percent:.1f}%" if isinstance(percent, float) else percent
        except Exception:
            percent = 'N/A'
            percent_val = None
        row.append(percent)
        # Store percent_val for sorting
        table.append((row, log_name, percent_val))

    # Sort table
    if sort_by == "percent":
        # Sort by percent_val descending, then name
        table.sort(key=lambda x: (x[2] is None, -(x[2] or 0), x[1]))
    else:
        # Sort by log_name
        table.sort(key=lambda x: x[1])

    # Extract only the row for tabulate
    table_rows = [row for row, _, _ in table]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


def homework1_q_4_2(sort_by="name"):
    # Path to your TensorBoard log directory
    log_base_dir = "data/q2/"
    log_dirs = [os.path.join(log_base_dir, d) for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d))]

    # Collect results for all log dirs
    results = {}
    for log_dir in log_dirs:
        print(f"Processing log directory: {log_dir}")

        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        values = {}
        for tag, enabled in scalars.items():
            if enabled:
                try:
                    scalars_list = event_acc.Scalars(tag)
                    # Convert list of ScalarEvent to list of values
                    value = [s.value for s in scalars_list] if scalars_list else 'No data'
                except KeyError:
                    value = 'Not found'
                values[tag] = value
        results[os.path.basename(log_dir)] = values
    
    # Plot iterations vs Eval_AverageReturn for each task, with std error bars
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 5))
    for task, vals in results.items():
        eval_avg = vals.get('Eval_AverageReturn', [])
        eval_std = vals.get('Eval_StdReturn', [])
        if (
            isinstance(eval_avg, list) and eval_avg and eval_avg != 'No data'
            and isinstance(eval_std, list) and eval_std and eval_std != 'No data'
        ):
            x = np.arange(1, len(eval_avg) + 1)
            plt.plot(x, eval_avg, label=task)
            plt.fill_between(x, np.array(eval_avg) - np.array(eval_std), np.array(eval_avg) + np.array(eval_std), alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Eval_AverageReturn')
    plt.title('Eval_AverageReturn vs Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort_by", choices=["name", "percent"], default="name", help="Sort by log directory name or Eval/Train %")
    parser.add_argument("--mode", choices=["run", "report"], default="report", help="Choose to run tasks or generate report/plots")
    parser.add_argument("--do_dagger", action="store_true", help="If running tasks, use DAgger (for q2)")
    parser.add_argument("--log_video", action="store_true", help="If set, log videos during training")
    args = parser.parse_args()

    if args.mode == "run":
        run_tasks(do_dagger=args.do_dagger, log_video=args.log_video)
    else:
        plot_table(log_base_dir="data/q1", sort_by=args.sort_by)
        plot_table(log_base_dir="data/q2", sort_by=args.sort_by)
        homework1_q_4_2(sort_by=args.sort_by)