"""
Create files:
python -m sim_student.scripts.human_eval create a --test_on test
python -m sim_student.scripts.human_eval create a --second_run --test_on test

Analyze results:
python -m sim_student.scripts.human_eval analyze a
"""

from typing import List, Dict, Union, Literal
import argparse
import json
import os
from ast import literal_eval
import re
from collections import Counter
import random
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

from sim_student.data_loading import load_train_val_data, load_test_data, read_csv
from sim_student.data_utils import Dialogue
from sim_student.prompting import LETTERS, END_OF_DIALOGUE

TASK_A_METHOD_NAMES = ["DPO", "Zero-Shot", "Oracle"]
TASK_A_EVAL_FILES = [
    "results/outputs_eedi_test_student-turn_eedi-stud-dpo-pnone-8b-ss.2-eft5_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_zs-eth-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv",
    "results/outputs_eedi_test_student-turn_persona-ff-gpt-4.1_eval_sim-text,sim-acts,sim-ks,sim-corr-errs,tutor-ppl.csv"
]

#### Create files ####

def format_key(dialogue: Dialogue):
    return re.sub(r"np\.int64\((\d+)\)", r"\g<1>", dialogue["key"])

def format_question(dialogue: Dialogue):
    anno = dialogue["question_annotation"]
    return f"{dialogue['question']}\nCorrect Answer: {LETTERS[anno['correct_option'] - 1]}"

def format_solution(dialogue: Dialogue):
    anno = dialogue["question_annotation"]
    solution = f"Solution: {anno['solution']}\n"
    solution += "\n".join([f"Answer {LETTERS[i - 1]} Explanation: " + anno[f"option_{i}_explanation"] for i in range(1, 5)])
    return solution

def format_dialogue(dialogue: Dialogue, ending_turn: int = None):
    turns = []
    ending_turn = ending_turn or len(dialogue["turns"])
    for turn in dialogue["turns"][:ending_turn]:
        role = "STUDENT" if turn["role"] == "student" else "TUTOR"
        turns.append(f"[{role}]: {turn['content'].replace(END_OF_DIALOGUE, '')}")
    return "\n".join(turns)

def create_task_a(src_data: List[Dialogue], second_run: bool):
    num_dialogues = 24
    starting_turn = 5
    turns_per_dialogue = 5

    exclude_keys = set()
    if second_run:
        for idx in range(0, 2):
            df = pd.read_csv(f"human_eval/task_A_{idx + 1}.csv")
            for _, row in df.iterrows():
                exclude_keys.add(row.key)

    rand = random.Random(221)
    selected_dialogue_idxs = set(rand.sample(range(len(src_data)), len(src_data) if second_run else num_dialogues * 2))
    selected_dialogue_idxs = [
        dialogue_idx for dialogue_idx in selected_dialogue_idxs
        if len(src_data[dialogue_idx]["turns"]) >= starting_turn + 2 * turns_per_dialogue and format_key(src_data[dialogue_idx]) not in exclude_keys
    ]
    selected_dialogue_idxs = selected_dialogue_idxs[:num_dialogues]

    all_outputs = []
    dfs = [read_csv(fn) for fn in TASK_A_EVAL_FILES]
    for i in range(len(dfs[0])):
        dialogue_idx = dfs[0].iloc[i].dialogue_idx
        if dialogue_idx not in selected_dialogue_idxs:
            continue
        turn_idx = dfs[0].iloc[i].turn_idx
        if turn_idx < starting_turn or turn_idx >= starting_turn + 2 * turns_per_dialogue:
            continue
        dialogue = src_data[dialogue_idx]
        order = rand.sample(range(len(dfs)), len(dfs))
        all_outputs.append({
            "key": format_key(dialogue),
            "turn_id": turn_idx,
            "question": format_question(dialogue),
            "solution": format_solution(dialogue),
            "context": format_dialogue(dialogue, ending_turn=turn_idx),
            "gt": dialogue["turns"][turn_idx]["content"],
            "methods": [dfs[df_idx].iloc[i].pred_turn for df_idx in order],
            "order": order
        })

    outputs_per_participant = [
        all_outputs[:70],   # 50 (0-49) unique, 20 (50-69) shared
        all_outputs[50:120] # 20 (50-69) shared, 50 (70-119) unique
    ]

    for idx, outputs in enumerate(outputs_per_participant):
        if second_run:
            idx += 2
        filename = f"task_A_{idx + 1}.csv"
        os.makedirs("human_eval", exist_ok=True)
        pd.DataFrame(outputs).to_csv(f"human_eval/{filename}", index=False)


#### Analyze results ####

BEHAVIOR_MAP = {
    "Math Answer": 1,
    "Not Understanding": 2,
    "Seek Information": 3,
    "Off-Topic": 4,
    "Acknowledge": 5,
}

def norm_label(value: Union[str, int, None], label: Literal["act", "corr", "likert"], src: Literal["auto", "hum"]):
    assert label in ("act", "corr", "likert") and src in ("auto", "hum")
    if label == "act":
        if src == "auto":
            return BEHAVIOR_MAP[value]
        return value
    if label == "corr":
        if src == "auto":
            return 1 if value is True else 0 if value is False else -1
        return -1 if value is None else value
    if label == "likert":
        if src == "hum":
            return (value - 1) / 4

def handle_na(value):
    return -1 if pd.isna(value) else value

def analyze_task_a(results: Dict[str, dict], src_data: List[Dialogue]):
    num_participants = 3
    all_participant_ids = [f"A{idx + 1}" for idx in range(num_participants)]

    key_to_src_dialogue = {
        format_key(dialogue): dialogue for dialogue in src_data
    }

    participant_dfs = [
        pd.read_csv(f"human_eval/task_A_{idx + 1}.csv", converters={"order": literal_eval})
        for idx in range(num_participants)
    ]
    participant_df_key_turn_to_row = [
        {(row["key"], row["turn_id"]): row for _, row in df.iterrows()}
        for df in participant_dfs
    ]

    eval_dfs = [
        pd.read_csv(filename, converters={"turns": literal_eval})
        for filename in TASK_A_EVAL_FILES
    ]
    eval_df_key_turn_to_row = [
        {(format_key(row), row["turn_idx"]): row for _, row in df.iterrows()}
        for df in eval_dfs
    ]

    method_to_act_score = {idx: [] for idx in range(len(TASK_A_METHOD_NAMES))}
    method_to_corr_score = {idx: [] for idx in range(len(TASK_A_METHOD_NAMES))}
    method_to_err_score = {idx: [] for idx in range(len(TASK_A_METHOD_NAMES))}
    method_to_ling_score = {idx: [] for idx in range(len(TASK_A_METHOD_NAMES))}

    all_act_scores_human = []
    all_act_scores_auto = []
    all_corr_scores_human = []
    all_corr_scores_auto = []
    all_err_scores_human = []
    all_err_scores_auto = []
    all_ling_scores_human = []
    all_cos_sim_scores_auto = []
    all_rouge_scores_auto = []

    all_gt_acts_human = []
    all_gt_acts_label = []
    all_gt_corr_human = []
    all_gt_corr_label = []

    key_turn_method_to_acts = {}
    key_turn_method_to_corr = {}
    key_turn_method_to_err = {}
    key_turn_method_to_ling = {}

    method_to_corr_evals_detailed = [Counter() for _ in range(len(TASK_A_METHOD_NAMES))]
    method_to_corr_evals = [Counter() for _ in range(len(TASK_A_METHOD_NAMES))]

    # Collect data
    for participant_idx, participant_id in enumerate(all_participant_ids):
        participant_data = results[participant_id]
        for key, turns in sorted(participant_data.items()):
            if key == "id":
                continue
            dialogue = key_to_src_dialogue[key]
            for turn_id, turn in sorted(turns.items(), key=lambda kv: int(kv[0])):
                turn_idx = int(turn_id)
                key_turn = (key, turn_idx)
                acts_anno_exists = dialogue["acts"] is not None
                corr_anno_exists = dialogue["correctness"] is not None and "error" not in dialogue["correctness"]
                gt_act = norm_label(turn["gt_evaluation"]["behavior"], "act", "hum")
                gt_corr = norm_label(turn["gt_evaluation"]["correctness"], "corr", "hum")
                if acts_anno_exists:
                    all_gt_acts_human.append(gt_act)
                    all_gt_acts_label.append(norm_label(dialogue["acts"][f"turn {turn_idx + 1}"]["act"], "act", "auto"))
                if corr_anno_exists:
                    all_gt_corr_human.append(gt_corr)
                    all_gt_corr_label.append(norm_label(dialogue["correctness"][f"turn {turn_idx + 1}"]["correct"], "corr", "auto"))
                key_turn_method = (*key_turn, "gt")
                key_turn_method_to_acts.setdefault(key_turn_method, []).append(gt_act)
                key_turn_method_to_corr.setdefault(key_turn_method, []).append(gt_corr)
                order = participant_df_key_turn_to_row[participant_idx][key_turn]["order"]
                for cand_eval in sorted(turn["candidate_evaluations"], key=lambda cand_eval: cand_eval["method_index"]):
                    method = order[cand_eval["method_index"]]
                    pred_act = norm_label(cand_eval["behavior"], "act", "hum")
                    act_score = 1 if gt_act == pred_act else 0
                    pred_corr = norm_label(cand_eval["correctness"], "corr", "hum")
                    corr_score = -1 if gt_corr == -1 else 1 if gt_corr == pred_corr else 0
                    err_score = -1 if gt_corr != 0 else 1 if cand_eval["sameError"] == "Yes" else 0
                    ling_score = norm_label(cand_eval["linguistic"], "likert", "hum")
                    method_to_act_score[method].append(act_score)
                    if gt_corr != -1:
                        method_to_corr_score[method].append(corr_score)
                    if err_score != -1:
                        method_to_err_score[method].append(err_score)
                    method_to_ling_score[method].append(ling_score)
                    auto_eval_row = eval_df_key_turn_to_row[method][key_turn]
                    if acts_anno_exists:
                        all_act_scores_human.append(act_score)
                        all_act_scores_auto.append(auto_eval_row["acts-sim"])
                    if corr_anno_exists:
                        all_corr_scores_human.append(corr_score)
                        all_corr_scores_auto.append(handle_na(auto_eval_row["correctness-sim"]))
                        method_to_corr_evals_detailed[method][(cand_eval['correctness'], turn['gt_evaluation']['correctness'], corr_score, handle_na(auto_eval_row["correctness-sim"]))] += 1
                        method_to_corr_evals[method][(corr_score, handle_na(auto_eval_row["correctness-sim"]))] += 1
                    all_err_scores_human.append(err_score)
                    all_err_scores_auto.append(handle_na(auto_eval_row["error-sim"]))
                    all_ling_scores_human.append(ling_score)
                    all_cos_sim_scores_auto.append(auto_eval_row["cos-sim"])
                    all_rouge_scores_auto.append(auto_eval_row["rouge"])
                    key_turn_method = (*key_turn, method)
                    key_turn_method_to_acts.setdefault(key_turn_method, []).append(pred_act)
                    key_turn_method_to_corr.setdefault(key_turn_method, []).append(pred_corr)
                    key_turn_method_to_err.setdefault(key_turn_method, []).append(err_score)
                    key_turn_method_to_ling.setdefault(key_turn_method, []).append(ling_score)

    # Report statistics
    print("Performance:")
    for idx, method in enumerate(TASK_A_METHOD_NAMES):
        print(f"{method}:")
        print(f"Acts: {np.mean(method_to_act_score[idx]):.4f}")
        print(f"Correctness: {np.mean(method_to_corr_score[idx]):.4f}")
        print(f"Errors: {np.mean(method_to_err_score[idx]):.4f}")
        print(f"Linguistic: {np.mean(method_to_ling_score[idx]):.4f}")
    print("\nHuman-Metric Agreement:")
    print(f"Acts: {cohen_kappa_score(all_act_scores_human, all_act_scores_auto):.4f}")
    print(f"Correctness: {cohen_kappa_score(all_corr_scores_human, all_corr_scores_auto):.4f}")
    print(f"Errors: {cohen_kappa_score(all_err_scores_human, all_err_scores_auto):.4f}")
    print(f"Linguistic (Cos. Sim): {pearsonr(all_ling_scores_human, all_cos_sim_scores_auto).statistic:.4f}")
    print(f"Linguistic (ROUGE): {pearsonr(all_ling_scores_human, all_rouge_scores_auto).statistic:.4f}")
    print("\nHuman-Label Agreement:")
    print(f"Acts: {cohen_kappa_score(all_gt_acts_human, all_gt_acts_label):.4f}")
    print(f"Correctness: {cohen_kappa_score(all_gt_corr_human, all_gt_corr_label):.4f}")
    print("\nHuman-Human Agreement:")
    for label, data in [("Acts", key_turn_method_to_acts), ("Correctness", key_turn_method_to_corr), ("Errors", key_turn_method_to_err), ("Linguistic", key_turn_method_to_ling)]:
        first_vals = []
        second_vals = []
        for vals in data.values():
            if len(vals) > 1:
                first_vals.append(vals[0])
                second_vals.append(vals[1])
        if label == "Linguistic":
            print(f"{label}: {pearsonr(first_vals, second_vals).statistic:.4f}")
        else:
            print(f"{label}: {cohen_kappa_score(first_vals, second_vals):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "analyze"])
    parser.add_argument("task", choices=["a"])
    parser.add_argument("--test_on", choices=["val", "test"], default="test")
    parser.add_argument("--second_run", action="store_true")
    args = parser.parse_args()

    if args.test_on == "val":
        _, src_data = load_train_val_data("eedi")
    else:
        src_data = load_test_data("eedi")

    if args.mode == "create":
        if args.task == "a":
            create_task_a(src_data, args.second_run)
    else:
        with open("human_eval/task_ratings.json") as f:
            results = json.load(f)
        results = {participant_data["id"]: participant_data for participant_data in results}
        if args.task == "a":
            analyze_task_a(results, src_data)

if __name__ == "__main__":
    main()
