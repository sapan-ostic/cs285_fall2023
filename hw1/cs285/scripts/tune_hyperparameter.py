"""
Usage:
    python cs285/scripts/tune_hyperparameter.py --mode tune
        Runs hyperparameter tuning for behavioral cloning on Walker2d-v4.
        This will launch multiple experiments with different hyperparameter combinations.

    python cs285/scripts/tune_hyperparameter.py --mode visualize
        Visualizes the results of the hyperparameter tuning using parallel coordinates plots.
        Make sure the log directories are present in data/bc_hyperparameter_tuning/.
"""

import subprocess
import itertools
import argparse


def tune_hyperparameter():
    # Define hyperparameter grid
    param_grid = {
        "env_name": ["Walker2d-v4"],
        "learning_rate": [5e-2, 5e-3, 5e-4],
        "n_iter": [1],
        "steps": [1000, 2000, 3000, 5000, 10000],
        "n_layers": [2, 3, 4],
        "size": [64, 128, 256],
        "eval_batch_size": [1000],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in combinations:
        exp_name = (
            f"bc_{params['env_name']}_"
            f"lr{params['learning_rate']}_"
            f"n_iter{params['n_iter']}_"
            f"steps{params['steps']}"
            f"n_layers{params['n_layers']}_"
            f"size{params['size']}_"
            f"eval_batch_size{params['eval_batch_size']}_"
        )
        cmd = [
            "python", "cs285/scripts/run_hw1.py",
            "--env_name", str(params["env_name"]),
            "--learning_rate", str(params["learning_rate"]),
            "--n_iter", str(params["n_iter"]),
            "--num_agent_train_steps_per_iter", str(params["steps"]),
            "--n_layers", str(params["n_layers"]),
            "--size", str(params["size"]),
            "--eval_batch_size", str(params["eval_batch_size"]),
            "--expert_policy_file", "cs285/policies/experts/Walker2d.pkl",
            "--expert_data", "cs285/expert_data/expert_data_Walker2d-v4.pkl",
            "--video_log_freq", "-1",
            "--exp_name", exp_name,
            "--log_dir", "data/bc_hyperparameter_tuning/" + exp_name,
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd)


def visualize_results():
    import os
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd
    import plotly.express as px
    import re

    log_base_dir = "data/bc_hyperparameter_tuning/"
    log_dirs = [
        os.path.join(log_base_dir, d)
        for d in os.listdir(log_base_dir)
        if os.path.isdir(os.path.join(log_base_dir, d))
    ]

    tags = [
        'Eval_AverageReturn',
        'Eval_StdReturn',
        'Train_AverageReturn',
        'Loss/train'
    ]

    def extract_metrics_from_logdir(log_dir, tags):
        event_acc = EventAccumulator(log_dir)
        try:
            event_acc.Reload()
        except Exception:
            return None
        result = {}
        for tag in tags:
            try:
                scalars = event_acc.Scalars(tag)
                value = scalars[-1].value if scalars else None
            except KeyError:
                value = None
            result[tag] = value
        # Add %Eval/Train
        eval_ret = result.get('Eval_AverageReturn', None)
        train_ret = result.get('Train_AverageReturn', None)
        try:
            percent = (float(eval_ret) / float(train_ret)) * 100 if train_ret not in [None, 0] else None
            percent = round(percent, 1) if percent is not None else None
        except Exception:
            percent = None
        result['%Eval/Train'] = percent
        return result

    def parse_exp_name(exp_name):
        hp = {}
        hp['env_name'] = re.search(r'bc_(.*?)_', exp_name).group(1)
        hp['learning_rate'] = float(re.search(r'lr([0-9.e-]+)_', exp_name).group(1))
        hp['n_iter'] = int(re.search(r'n_iter([0-9]+)_', exp_name).group(1))
        hp['steps'] = int(re.search(r'steps([0-9]+)', exp_name).group(1))
        hp['n_layers'] = int(re.search(r'n_layers([0-9]+)_', exp_name).group(1))
        hp['size'] = int(re.search(r'size([0-9]+)_', exp_name).group(1))
        hp['eval_batch_size'] = int(re.search(r'eval_batch_size([0-9]+)_', exp_name).group(1))
        return hp

    records = []
    for log_dir in log_dirs:
        exp_name = os.path.basename(log_dir)
        try:
            hp = parse_exp_name(exp_name)
        except Exception:
            continue
        metrics = extract_metrics_from_logdir(log_dir, tags)
        if metrics is not None:
            record = {**hp, **metrics}
            records.append(record)

    if records:
        import plotly.io as pio
        pio.renderers.default = "browser"
        df = pd.DataFrame(records)
        fig = px.parallel_coordinates(
            df,
            dimensions=['learning_rate', 'steps', 'n_layers', 'size', '%Eval/Train'],
            color='%Eval/Train',
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Hyperparameter Tuning Results"
        )
        fig.show()
    else:
        print("No valid logs found for plotting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "visualize"], default="tune", help="Choose to tune hyperparameters or visualize results")
    args = parser.parse_args()
    if args.mode == "tune":
        tune_hyperparameter()
    else:
        visualize_results()
