import os
import glob
import yaml
import numpy as np

OUTPUT_DIR = "../output"
ENVS = ["Hopper-v4", "HumanoidStandup-v4"]
ALGS = ["ppo", "sac"]


for env in ENVS:
    for alg in ALGS:
        pattern = os.path.join(OUTPUT_DIR, env, alg, "*", "config.yaml")
        runs = sorted(glob.glob(pattern))

        if not runs:
            continue

        print(f"\n{'*'*60}")
        print(f"  {alg.upper()} on {env}")
        print(f"{'*'*60}")

        results = []
        for cfg_path in runs:
            run_dir = os.path.dirname(cfg_path)
            returns_path = os.path.join(run_dir, "returns.npy")

            # final_run folders for Problem 4)
            if "final_run" in run_dir:
                continue

            if not os.path.exists(returns_path):
                continue

            cfg = yaml.safe_load(open(cfg_path))
            data = np.load(returns_path)

            # Area under the curve using trapezoidal integration
            auc = np.trapz(data[:, 1], data[:, 0])

            if alg == "ppo":
                lr = cfg.get("learning_rate", "?")
                gamma = cfg.get("gamma", "?")
                results.append((auc, lr, gamma, run_dir))
            else:
                lr = cfg.get("policy_lr", "?")
                gamma = cfg.get("gamma", "?")
                results.append((auc, lr, gamma, run_dir))

        results.sort(key=lambda x: x[0], reverse=True)

        lr_label = "learning_rate" if alg == "ppo" else "policy_lr"
        print(f"  {'Rank':<6}{lr_label:<18}{'gamma':<12}{'AUC':<25}")
        print(f"  {'-'*55}")
        for i, (auc, lr, gamma, run_dir) in enumerate(results, 1):
            print(f"  {i:<6}{lr:<18}{gamma:<12}{auc:<25.2f}")
