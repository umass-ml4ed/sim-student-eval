import os
import json
import argparse
import matplotlib.pyplot as plt

from sim_student.data_loading import load_test_data, read_csv
from sim_student.acts import eval_acts_turn_level, ACTS
from sim_student.correctness import eval_correctness_turn_level
from sim_student.prompting import correct_to_str

CORR = ["correct", "incorrect", "na"]

METHODS = ["Ground-Truth", "SFT (3B)", "DPO (3B)", "SFT (8B)", "DPO (8B)", "Zero-Shot", "OCEAN Persona", "ICL", "Reasoning", "Oracle"]
FILENAMES = [
    "results/outputs_eedi_test_student-turn_eedi-stud-sft-pnone-3b.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-dpo-pnone-3b-ss.2-eft5.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-sft-pnone-8b.csv",
    "results/outputs_eedi_test_student-turn_eedi-stud-dpo-pnone-8b-ss.2-eft5.csv",
    "results/outputs_eedi_test_student-turn_zs-eth-gpt-4.1.csv",
    "results/outputs_eedi_test_student-turn_persona-ocean-gpt-4.1.csv",
    "results/outputs_eedi_test_student-turn_icl-gpt-4.1.csv",
    "results/outputs_eedi_test_student-turn_reasoning-gpt-5-mini.csv",
    "results/outputs_eedi_test_student-turn_persona-ff-gpt-4.1.csv",
]

def main():
    os.makedirs("figures", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", type=int)
    args = parser.parse_args()

    if os.path.exists("temp_plot_label_distros.json"):
        with open("temp_plot_label_distros.json") as f:
            temp = json.load(f)
            method_to_acts = [[act.lower() for act in method_acts] for method_acts in temp["acts"]]
            method_to_corr = temp["corr"]
    else:
        # Load results
        eval_results = [read_csv(fn).to_dict("records") for fn in FILENAMES]
        if args.truncate:
            for idx in range(len(eval_results)):
                eval_results[idx] = eval_results[idx][:args.truncate]
        data_concatted = [sample for method_results in eval_results for sample in method_results]
        data_size = len(eval_results[0])
        assert len(data_concatted) == data_size * len(FILENAMES)

        # Get act and correctness for simulated turns
        acts_args = {"base_model": "8b", "model_name": "acts-8b"}
        act_predictions = eval_acts_turn_level(data_concatted, acts_args, "pred_turn")
        correctness_args = {"base_model": "8b", "model_name": "correctness-8b"}
        correctness_predictions = eval_correctness_turn_level(data_concatted, correctness_args, "pred_turn")

        # Accumulate acts and correctness per method
        method_to_acts = [[] for _ in range(len(eval_results))]
        method_to_corr = [[] for _ in range(len(eval_results))]
        for idx, (act, corr) in enumerate(zip(act_predictions, correctness_predictions)):
            method_idx = idx // data_size
            method_to_acts[method_idx].append(act.lower())
            method_to_corr[method_idx].append(corr.lower())

        with open("temp_plot_label_distros.json", "w") as f:
            json.dump({"acts": method_to_acts, "corr": method_to_corr}, f)

    # Get acts and correctness for ground-truth
    method_to_acts.insert(0, [])
    method_to_corr.insert(0, [])
    gt_data = load_test_data("eedi")
    for dialogue in gt_data:
        acts = dialogue["acts"]
        if acts:
            for act_data in acts.values():
                method_to_acts[0].append(act_data["act"].lower())
        corr = dialogue["correctness"]
        if corr and "error" not in corr:
            for corr_data in corr.values():
                method_to_corr[0].append(correct_to_str(corr_data["correct"]))

    plt.rcParams['font.size'] = 14

    # Plot acts distribution
    acts_counts = [{act: method_to_acts[i].count(act) / len(method_to_acts[i]) for act in ACTS} for i in range(len(METHODS))]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(ACTS))
    width = 0.08
    for i, method in enumerate(METHODS):
        counts = [acts_counts[i].get(act, 0) for act in ACTS]
        ax.bar([xi + i * width for xi in x], counts, width, label=method)
    ax.set_xlabel("Act")
    ax.set_ylabel("Frequency")
    ax.set_title("Act Distribution by Method")
    ax.set_xticks([xi + width * (len(METHODS) - 1) / 2 for xi in x])
    ax.set_xticklabels(ACTS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/acts_distribution.png", dpi=200)
    plt.close()

    # Plot correctness distribution
    corr_counts = [{corr: method_to_corr[i].count(corr) / len(method_to_corr[i]) for corr in CORR} for i in range(len(METHODS))]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(CORR))
    for i, method in enumerate(METHODS):
        counts = [corr_counts[i].get(corr, 0) for corr in CORR]
        ax.bar([xi + i * width for xi in x], counts, width, label=method)
    ax.set_xlabel("Correctness")
    ax.set_ylabel("Frequency")
    ax.set_title("Correctness Distribution by Method")
    ax.set_xticks([xi + width * (len(METHODS) - 1) / 2 for xi in x])
    ax.set_xticklabels(CORR)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/correctness_distribution.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
