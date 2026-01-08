import os
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

FILENAMES = [
    "results/outputs_eedi_test_student-turn_eedi-stud-sft-pnone-3b_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-dpo-pnone-3b-ss.2-eft5_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-sft-pnone-8b_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-dpo-pnone-8b-ss.2-eft5_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_zs-eth-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_persona-ocean-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_icl-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_reasoning-gpt-5-mini_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_persona-ff-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
]
METHODS = ["SFT (3B)", "DPO (3B)", "SFT (8B)", "DPO (8B)", "Zero-Shot", "OCEAN Persona", "ICL", "Reasoning", "Oracle"]

COLS = ["acts-sim", "correctness-sim", "error-sim", "ks-sim", "cos-sim", "rouge", "tutor-ppl"]
METRIC_NAMES = ["Acts", "Correctness", "Errors", "Knowledge Acquisition", "Cosine Similarity", "ROUGE-L", "Tutor Response"]

def main():
    max_turns = 15

    # Collect metric values across methods and turn idxs
    dfs = [pd.read_csv(fn) for fn in FILENAMES]
    col_to_method_to_turn_to_vals = {col: [{} for _ in range(len(dfs))] for col in COLS}
    for i in range(len(dfs[0])):
        for col in COLS:
            for df, turn_to_vals in zip(dfs, col_to_method_to_turn_to_vals[col]):
                val = df.iloc[i][col]
                if not pd.isna(val):
                    turn_pair_idx = (df.iloc[i]["turn_idx"] + 1) // 2
                    if turn_pair_idx <= max_turns:
                        turn_to_vals.setdefault(turn_pair_idx, []).append(val)

    # Get turn-level stats
    col_to_method_to_turn_to_stats = {
        col: [
            {
                turn_idx: norm.fit(turn_vals)
                for turn_idx, turn_vals in turn_to_vals.items()
            }
            for turn_to_vals in method_to_turn_to_vals
        ]
        for col, method_to_turn_to_vals in col_to_method_to_turn_to_vals.items()
    }

    os.makedirs("figures", exist_ok=True)
    plt.rcParams['font.size'] = 14

    # Create a 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (metric_name, col) in enumerate(zip(METRIC_NAMES, COLS)):
        if idx == len(METRIC_NAMES) - 1: # Put last method in the middle of the last row
            idx += 1
        ax = axes[idx]
        for method_idx, turn_to_stats in enumerate(col_to_method_to_turn_to_stats[col]):
            turns = sorted(turn_to_stats.keys())
            means = [turn_to_stats[turn][0] for turn in turns]
            ax.plot(turns, means, marker='o', label=METHODS[method_idx])
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xlabel("Turn Pair Index")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} by Turn")
        ax.grid(True)

    axes[-3].set_visible(False)

    # Create legend in the last subplot
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='center', fontsize=12)

    plt.tight_layout()
    plt.savefig("figures/results_by_turn_grid.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
