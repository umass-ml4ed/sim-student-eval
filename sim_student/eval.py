from typing import List, Dict, Tuple, Union
import argparse
import evaluate
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score

from sim_student.data_loading import read_csv
from sim_student.data_utils import DatasetBase
from sim_student.prompting import get_llm_prompt, get_prompting_function, extract_result, correct_to_str, get_local_prompt
from sim_student.model import get_base_model, get_model
from sim_student.training_utils import TrainingCollator
from sim_student.llmkt import llmkt_eval, get_llmkt_model_for_eval, LLMKTDatasetTurnLevel
from sim_student.acts import eval_acts_turn_level
from sim_student.utils import run_gc, merge_defaults

EMB_MODEL_BATCH_SIZE = 16

def eval_similarity_text(data: List[dict], _args: dict = None, sbert_model: SentenceTransformer = None):
    print("Evaluating turn-level text similarity...")
    gt_turns = [sample["gt_turn"] for sample in data]
    pred_turns = [sample["pred_turn"] for sample in data]

    print("Computing ROUGE...")
    rouge_module = evaluate.load("rouge")
    rouge_list = rouge_module.compute(
        predictions=pred_turns,
        references=gt_turns,
        rouge_types=["rougeL"],
        use_aggregator=False
    )["rougeL"]

    print("Computing Cosine Sim...")
    _sbert_model = sbert_model or SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    gt_encodings = _sbert_model.encode(gt_turns, batch_size=EMB_MODEL_BATCH_SIZE, show_progress_bar=True)
    pred_encodings = _sbert_model.encode(pred_turns, batch_size=EMB_MODEL_BATCH_SIZE, show_progress_bar=True)
    cos_sim_list = np.einsum("ij,ij->i", gt_encodings, pred_encodings).tolist()
    if not sbert_model:
        del _sbert_model
    del gt_encodings, pred_encodings
    run_gc()

    return {
        "rouge": rouge_list,
        "cos-sim": cos_sim_list
    }

def eval_similarity_correctness_and_errors(data: List[dict], args: dict):
    print("Evaluating turn-level correctness and error similarity...")

    system_prompt = """You are a math education expert. You will observe a tutoring dialogue where a student is attempting to solve a math problem. You will see two versions of the next student turn: a ground-truth turn and a candidate turn. Your job is to evaluate the correctness and errors of the candidate turn.
- The correctness of the ground-truth turn is given. You must evaluate the correctness of the candidate turn.
- Correctness can be "correct", "incorrect", or "na". It is "correct" if the student correctly responds to the previous tutor turn. It is "incorrect" if the student incorrectly responds to the previous tutor turn, or indicates they do not know the answer. It is "na" in all other cases, such as when the tutor does not ask a question or only asks a conversational question, or if the student response is purely conversational. A turn is conversational when it does not address a mathematical task posed by the tutor.
- If both the ground-truth AND candidate turns are "incorrect", evaluate if they have the same error. They have the same error if the two turns are mathematically EQUIVALENT. If they are mathematically inequivalent, they do NOT have the same error.

After reasoning, please return the correctness of the **candidate** turn as "correct", "incorrect", or "na". If both the ground-truth and candidate turns are incorrect, add "same error" or "different error" to your response (ex: "incorrect, same error"). Do not include any other text in your response."""
    correctness_rewards: List[Union[None, int]] = [None] * len(data)
    error_rewards: List[Union[None, int]] = [None] * len(data)
    gt_correctness: List[Union[None, int]] = [None] * len(data)
    pred_correctness: List[Union[None, int]] = [None] * len(data)
    prompts = []
    to_prompt_idx = []
    gt_corr = []
    for idx, sample in enumerate(data):
        # Handle annotation errors
        if not sample["correctness"] or "error" in sample["correctness"]:
            continue
        turn_idx = sample["turn_idx"]
        # No reward for either function when gt is na
        corr = correct_to_str(sample["correctness"][f"turn {turn_idx + 1}"]["correct"])
        if corr == "na":
            continue
        prompt = get_llm_prompt(sample, ending_turn=turn_idx)
        prompt += f"\n\nGround-truth next student turn:\n{sample['gt_turn']}\nCorrectness: {corr}\n\nCandidate next student turn:\n{sample['pred_turn']}"
        prompts.append(prompt)
        to_prompt_idx.append(idx)
        gt_corr.append(corr)
    prompting_fn = get_prompting_function({"engine": "openai", "annotation_model": "gpt-5-mini", "reasoning_effort": "low", "batch_api": True})
    raw_results = prompting_fn(prompts, system_prompt)
    results = [extract_result(result, "text") for result in raw_results]
    for idx, result, corr in zip(to_prompt_idx, results, gt_corr):
        result = result.strip().lower()
        gt_correctness[idx] = 1 if corr == "correct" else 0 if corr == "incorrect" else None
        pred_correctness[idx] = 1 if result == "correct" else 0 if "incorrect" in result else None
        if corr == "correct":
            correctness_rewards[idx] = 1 if result == "correct" else 0
        elif corr == "incorrect":
            correctness_rewards[idx] = 1 if "incorrect" in result else 0
            error_rewards[idx] = 1 if "same error" in result else 0
    return {"correctness-sim": correctness_rewards, "correctness-gt": gt_correctness, "correctness-pred": pred_correctness, "error-sim": error_rewards}

def eval_similarity_acts(data: List[dict], args: dict):
    print("Evaluating turn-level dialogue acts similarity...")
    acts_class_args = {**args, "base_model": args["rm_base_model"], "model_name": args["acts_model_name"]}
    all_pred_acts = eval_acts_turn_level(data, acts_class_args, "pred_turn")
    run_gc()

    rewards = []
    gt_acts = []
    for turn, pred_act in zip(data, all_pred_acts):
        if turn["acts"]:
            gt_act = turn["acts"][f"turn {turn['turn_idx'] + 1}"]["act"]
            rewards.append(1 if gt_act == pred_act else 0)
            gt_acts.append(gt_act)
        else:
            rewards.append(None) # In case of annotation errors
            gt_acts.append(None)
    return {"acts-sim": rewards, "acts-gt": gt_acts, "acts-pred": all_pred_acts}

def get_quantile(val: float, cutoffs: List[float]):
    if val is None:
        return None
    for idx, cutoff in enumerate(cutoffs):
        if val < cutoff:
            return idx
    return len(cutoffs)

def ks_delta_reward(gt_quantile: float, pred_quantile: float, num_quantiles: int):
    return 1 - (abs(gt_quantile - pred_quantile) / (num_quantiles - 1))

def get_ks_deltas(dia_turn_to_kc_probs: Dict[Tuple[int, int], List[float]], results: List[dict]):
    deltas = []
    for turn in results:
        prev_probs = dia_turn_to_kc_probs.get((turn["dialogue_idx"], turn["turn_idx"] - 2))
        if prev_probs:
            deltas.append([cur_p - prev_p for cur_p, prev_p in zip(turn["kc_probs"], prev_probs)])
        else:
            # For the first student turn in the dialogue
            deltas.append([None] * len(turn["kc_probs"]))
    return deltas

def eval_knowledge_state(data: List[dict], args: dict):
    print("Evaluating turn-level knowledge state similarity...")
    num_quantiles = 5

    # Get KS predictions for ground-truth and predicted turns via LLMKT
    llmkt_args = {**args, "base_model": args["rm_base_model"], "model_name": args["llmkt_model_name"], "batch_size": 8}
    model, tokenizer = get_llmkt_model_for_eval(llmkt_args)
    gt_dataset = LLMKTDatasetTurnLevel(data, tokenizer, "gt_turn", args["kcs_src"], persona_type="ocean")
    gt_results, _ = llmkt_eval(model, tokenizer, gt_dataset, llmkt_args)
    pred_dataset = LLMKTDatasetTurnLevel(data, tokenizer, "pred_turn", args["kcs_src"], persona_type="ocean")
    pred_results, _ = llmkt_eval(model, tokenizer, pred_dataset, llmkt_args)
    assert all([gt["dialogue_idx"] == pred["dialogue_idx"] and gt["turn_idx"] == pred["turn_idx"] for gt, pred in zip(gt_results, pred_results)])

    # Compute KS deltas for both sets of turns
    dia_turn_to_kc_probs = {(turn["dialogue_idx"], turn["turn_idx"]): turn["kc_probs"] for turn in gt_results}
    gt_deltas = get_ks_deltas(dia_turn_to_kc_probs, gt_results)
    pred_deltas = get_ks_deltas(dia_turn_to_kc_probs, pred_results)

    # Categorize KS deltas by quantile (cutoffs computed with ground-truth)
    delta_quantile_cutoffs = np.quantile(
        [prob for gt_probs in gt_deltas for prob in gt_probs if prob is not None],
        np.arange(1, num_quantiles) / num_quantiles
    )
    print(f"KS Delta Quantile Cutoffs: {delta_quantile_cutoffs}")
    gt_quantiles = [[get_quantile(delta, delta_quantile_cutoffs) for delta in deltas] for deltas in gt_deltas]
    pred_quantiles = [[get_quantile(delta, delta_quantile_cutoffs) for delta in deltas] for deltas in pred_deltas]

    # Sample-level similarity relative to average distance of quantiles
    ks_sim = [
        None if gt[0] is None else
        float(np.mean([
            ks_delta_reward(gt_kc_quant, pred_kc_quant, num_quantiles)
            for gt_kc_quant, pred_kc_quant in zip(gt, pred)
        ]))
        for gt, pred in zip(gt_quantiles, pred_quantiles)
    ]

    # Use QWK on quantiles for data-level similarity
    gt_quantiles_unrolled = [kc_q for q in gt_quantiles for kc_q in q if kc_q is not None]
    pred_quantiles_unrolled = [kc_q for q in pred_quantiles for kc_q in q if kc_q is not None]
    qwk = cohen_kappa_score(gt_quantiles_unrolled, pred_quantiles_unrolled, weights="quadratic")
    del model, tokenizer, gt_dataset, pred_dataset, gt_results, pred_results
    run_gc()
    return {"ks-sim": ks_sim, "ks-delta-quant-qwk": qwk, "ks-delta-gt": gt_deltas, "ks-delta-pred": pred_deltas, "ks-quant-gt": gt_quantiles, "ks-quant-pred": pred_quantiles}

class TutorPPLDataset(DatasetBase):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer):
        self.data = []
        for sample_idx, turn in enumerate(data):
            if turn["turn_idx"] == len(turn["turns"]) - 1:
                continue
            next_turn = turn["turns"][turn["turn_idx"] + 1]
            self.data.append({
                "sample_idx": sample_idx,
                "prompt": get_local_prompt(turn, "tutor", tokenizer, ending_turn=turn["turn_idx"] + 1, last_turn_sub=turn["pred_turn"]),
                "label": next_turn["content"]
            })

def eval_tutor_ppl(data: List[dict], args: dict):
    # Load tutor model for evaluation
    base_model, tokenizer = get_base_model(args["rm_base_model"], args["quantize"])
    model = get_model(base_model, True, model_name=args["tutor_model_name"])
    model.eval()

    cel = torch.nn.CrossEntropyLoss(reduction="mean")
    dataset = TutorPPLDataset(data, tokenizer)
    collator = TrainingCollator(tokenizer)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    results = [None] * len(data)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # Get 1 / ppl(tutor turn|history) for batch
            for idx in range(len(batch["input_ids"])):
                inv_ppl = 1 / torch.exp(cel(outputs.logits[idx, :-1, :], batch["labels"][idx, 1:])).item()
                results[batch["meta_data"][idx]["sample_idx"]] = inv_ppl

    # Free up memory
    del dataset, base_model, model, tokenizer
    run_gc()
    return {"tutor-ppl": results}

EVAL_TO_FN = {
    "sim-text": eval_similarity_text,
    "sim-acts": eval_similarity_acts,
    "sim-ks": eval_knowledge_state,
    "tutor-ppl": eval_tutor_ppl,
    "sim-corr-errs": eval_similarity_correctness_and_errors,
}

def apply_defaults(args: dict):
    defaults = {
        "rm_base_model": "8b",
        "quantize": False,
        "tutor_model_name": "eedi-tutor-sft-8b",
        "llmkt_model_name": "llmkt-8b-ocean",
        "kcs_src": "eedi",
        "acts_model_name": "acts-8b",
        "correctness_model_name": "correctness-8b"
    }
    if args["level"] == "turn":
        if args["role"] == "student":
            defaults["evaluations"] = ["sim-text", "sim-acts", "sim-ks", "sim-corr-errs", "tutor-ppl"]
    return merge_defaults(args, defaults)

METRICS_TO_AVERAGE = {"rouge", "cos-sim", "correctness-sim", "error-sim", "acts-sim", "ks-sim", "tutor-ppl"}

def eval(df: pd.DataFrame, args: dict):
    data = df.to_dict("records")
    args = apply_defaults(args)
    if args["level"] == "turn":
        cols_to_save = ["key", "prompt", "dialogue_idx", "turn_idx", "gt_turn", "pred_turn"]
    result_str = ""
    for eval_type in args["evaluations"]:
        cur_eval_results = EVAL_TO_FN[eval_type](data, args)
        for result_name, result in cur_eval_results.items():
            if isinstance(result, list):
                if len(result) == len(df):
                    df[result_name] = result
                    cols_to_save.append(result_name)
                if result_name in METRICS_TO_AVERAGE:
                    non_null = [r for r in result if r is not None]
                    result_str += f"{result_name}: {np.mean(non_null):.4f} ± {np.std(non_null):.4f}\n"
            elif isinstance(result, (int, float)):
                result_str += f"{result_name}: {result:.4f}\n"
            else:
                result_str += f"{result_name}: {result}\n"

    # Print and save results
    print(result_str)
    suffix = args["input_file"].replace("results/outputs_", "").replace(".csv", "")
    suffix += f"_eval_{','.join(args['evaluations'])}"
    df[cols_to_save].to_csv(f"results/outputs_{suffix}.csv", index=False)
    with open(f"results/metrics_{suffix}.txt", "w") as f:
        f.write(result_str)

def eval_from_file(args: dict):
    df = read_csv(args["input_file"])
    if args["truncate"]:
        df = df[:args["truncate"]]
    eval(df, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--evaluations")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate the number of samples to evaluate")
    parser.add_argument("--level", choices=["turn", "full"], default="turn")
    parser.add_argument("--role", choices=["student", "tutor"], default="student")
    parser.add_argument("--rm_base_model", default="8b", help="Base model for fine-tuned reward models")
    parser.add_argument("--quantize", action="store_true", help="If base_model should be quantized")
    parser.add_argument("--tutor_model_name", help="Model to use for tutor ppl eval")
    parser.add_argument("--llmkt_model_name", help="Model to use for LLMKT eval")
    parser.add_argument("--kcs_src", choices=["eedi"], default="eedi", help="Source of KCs for KS evaluation")
    parser.add_argument("--acts_model_name", help="Model to use for acts eval")
    parser.add_argument("--correctness_model_name", help="Model to use for correctness eval")
    args = parser.parse_args().__dict__
    if args["evaluations"]:
        args["evaluations"] = args["evaluations"].split(",")

    eval_from_file(args)

if __name__ == "__main__":
    main()
