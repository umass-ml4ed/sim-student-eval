import argparse
from typing import List
import random
from itertools import combinations
from scipy.stats import norm
from transformers import PreTrainedTokenizer
import datasets
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig

from sim_student.data_loading import load_train_val_data, save_overgenerated_data, load_overgenerated_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.model import get_base_model, get_model, generate_vllm, get_tokenizer
from sim_student.prompting import get_local_prompt
from sim_student.training_utils import MAX_LEN
from sim_student.testing import test, TestingDataset
from sim_student.eval import eval_similarity_text, eval_similarity_acts, eval_knowledge_state, eval_similarity_correctness_and_errors, eval_tutor_ppl
from sim_student.utils import get_checkpoint_path, run_gc, initialize_seeds

def get_overgen_descriptor(args: dict):
    role = "" if args["role"] == "student" else "tutor_"
    desc = f"{role}{args['base_model']}_{args['pt_model_name']}_temp{args['temperature']}_ns{args['num_samples']}"
    if args["subsample"]:
        desc += f"_ss{args['subsample']}"
    return desc

def unroll_data(data: List[Dialogue]):
    unrolled = []
    for dialogue_idx, dialogue in enumerate(data):
        for turn_idx, turn in enumerate(dialogue["turns"]):
            if not turn.get("candidates"):
                continue
            for cand_idx, cand in enumerate(turn["candidates"]):
                unrolled.append({
                    **dialogue,
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn_idx,
                    "cand_idx": cand_idx,
                    "gt_turn": turn["content"],
                    "pred_turn": cand
                })
    return unrolled

def apply_scores(data: List[Dialogue], unrolled_data: List[dict], scores: List[float]):
    results = [{} for _ in range(len(data))]
    for sample, score in zip(unrolled_data, scores):
        dialogue_idx = sample["dialogue_idx"]
        turn_idx = sample["turn_idx"]
        cand_idx = sample["cand_idx"]
        cand_scores = results[dialogue_idx].setdefault(turn_idx, [None] * len(data[dialogue_idx]["turns"][turn_idx]["candidates"]))
        cand_scores[cand_idx] = score
    return results

def score_tutor_ppl(unrolled_data: List[Dialogue], args: dict):
    assert args["tutor_model"]
    results = eval_tutor_ppl(unrolled_data, {**args, "tutor_model_name": args["tutor_model"]})
    return results["tutor-ppl"]

def score_acts_sim(unrolled_data: List[Dialogue], args: dict):
    assert args["acts_model"]
    acts_sim = eval_similarity_acts(unrolled_data, {**args, "acts_model_name": args["acts_model"]})
    return acts_sim["acts-sim"]

def score_kt(unrolled_data: List[Dialogue], args: dict):
    assert args["llmkt_model"]
    ks_sim = eval_knowledge_state(unrolled_data, {**args, "llmkt_model_name": args["llmkt_model"]})
    return ks_sim["ks-sim"]

def score_cos_sim(unrolled_data: List[Dialogue], _args: dict):
    text_sim = eval_similarity_text(unrolled_data)
    return text_sim["cos-sim"]

def score_correctness_and_error_sim(unrolled_data: List[Dialogue]):
    return eval_similarity_correctness_and_errors(unrolled_data, {"batch_api": True})

def generate_candidate_turns(data: List[Dialogue], args: dict):
    tokenizer = get_tokenizer(args["base_model"])
    dataset = TestingDataset(data, args["role"], tokenizer, args)
    prompts = [sample["prompt"] for sample in dataset]
    outputs = generate_vllm(args["base_model"], prompts, args["pt_model_name"],
                            {"n": args["num_samples"], "temperature": args["temperature"], "top_p": 0.95, "max_tokens": args["max_gen_tokens"]})
    for idx, sample in enumerate(dataset):
        candidates = outputs[idx * args["num_samples"] : (idx + 1) * args["num_samples"]]
        data[sample["dialogue_idx"]]["turns"][sample["turn_idx"]]["candidates"] = list(set(candidates))
    run_gc()

ROLE_TO_REWARDS = {
    "student": {"cos", "acts", "kt", "corr", "err", "tut"}
}

REWARD_TO_FN = {
    "acts": score_acts_sim,
    "kt": score_kt,
    "tut": score_tutor_ppl,
    "cos": score_cos_sim
}

def get_reward_cols(args: dict):
    cols = []
    for reward in args["rewards"]:
        col = f"reward_{reward}"
        if reward == "tut":
            col += f"_{args['tutor_model']}"
        elif reward == "so":
            col += f"_{args['student_model']}"
        cols.append(col)
    return cols

def overgen_dpo_data(args: dict):
    reward_cols = get_reward_cols(args)

    # For train and val data, perform 1) candidate turn generation, and 2) reward scoring
    train_data, val_data = load_train_val_data(args["dataset"])
    for split, data in [("train", train_data), ("val", val_data)]:
        should_save = False

        # Generate candidate turns
        descriptor = get_overgen_descriptor(args)
        if args["force_regen"]:
            overgenerated_data = None
        else:
            overgenerated_data = load_overgenerated_data(args["dataset"], split, descriptor, reward_cols)
        if overgenerated_data is None:
            print(f"Overgenerated data not found for {args['dataset']} {split} with descriptor {descriptor}, generating...")
            should_save = True
            if args["subsample"]:
                data = data[:int(len(data) * args["subsample"])]
            generate_candidate_turns(data, args)
        else:
            print(f"Overgenerated data found for {args['dataset']} {split} with descriptor {descriptor}, skipping generation")
            data = overgenerated_data

        # Score candidate turns
        unrolled_data = unroll_data(data)
        if any([col in reward_cols and col not in data[0] for col in ["reward_corr", "reward_err"]]):
            # Correctness and errors computed together so only do once
            correctness_and_error_scores = score_correctness_and_error_sim(unrolled_data)
        for reward, col in zip(args["rewards"], reward_cols):
            if col in data[0]:
                print(f"Reward {col} already scored for {args['dataset']} {split}, skipping scoring")
                continue
            print(f"Scoring {col} for {args['dataset']} {split}")
            should_save = True
            if reward == "corr":
                unrolled_scores = correctness_and_error_scores["correctness-sim"]
            elif reward == "err":
                unrolled_scores = correctness_and_error_scores["error-sim"]
            else:
                unrolled_scores = REWARD_TO_FN[reward](unrolled_data, args)
            scores = apply_scores(data, unrolled_data, unrolled_scores)
            for dialogue, score in zip(data, scores):
                dialogue[col] = score

        if should_save:
            print(f"Saving overgenerated data for {args['dataset']} {split} with descriptor {descriptor}")
            save_overgenerated_data(data, args["dataset"], split, descriptor)

        if split == "train":
            train_data = data
        elif split == "val":
            val_data = data

    return train_data, val_data

class DPODataset(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, args: dict):
        self.data = []
        num_turns = 0
        turns_without_pairs = 0
        excluded = 0
        reward_cols = get_reward_cols(args)
        reward_to_all_scores = {reward: [] for reward in reward_cols}

        # Do first pass to compute reward statistics and verify data
        for dialogue in data:
            for turn_idx, turn in enumerate(dialogue["turns"]):
                if turn["role"] != args["role"]:
                    continue
                for col in reward_cols:
                    if turn_idx in dialogue[col]:
                        reward_to_all_scores[col].extend(dialogue[col][turn_idx])
                    else:
                        # Should have value for all rewards, except tutor ppl for the last turn in the dialogue
                        assert col.startswith("reward_tut") and turn_idx == len(dialogue["turns"]) - 1

        # Map reward to (mean, std)
        reward_to_stats = {reward: norm.fit([s for s in scores if s is not None]) for reward, scores in reward_to_all_scores.items()}

        # Create preference pairs from data
        for dialogue in data:
            for turn_idx, turn in enumerate(dialogue["turns"]):
                if turn["role"] != args["role"]:
                    continue
                if args["exclude_first_turns"] and turn_idx < args["exclude_first_turns"]:
                    excluded += 1
                    continue
                prompt = get_local_prompt(dialogue, args["role"], tokenizer, ending_turn=turn_idx, persona_type=args["persona"])
                if len(prompt) >= MAX_LEN:
                    excluded += 1
                    continue
                num_turns += 1
                cands = turn["candidates"]

                # Get score from each reward function for each candidate
                reward_to_scores = [
                    dialogue[col][turn_idx]
                    for col in reward_cols
                    if turn_idx in dialogue[col]
                ] # num_rewards x num_cands

                # Normalize rewards if specified
                if args["reward_norm_level"] == "group":
                    for idx, scores in enumerate(reward_to_scores):
                        non_null_scores = [s for s in scores if s is not None]
                        if non_null_scores:
                            mean, std = norm.fit(non_null_scores)
                        else:
                            std = 0
                        if std:
                            reward_to_scores[idx] = [float((score - mean) / std) if score is not None else None for score in scores]
                        else:
                            reward_to_scores[idx] = [0] * len(scores) # Don't use this reward if all scores the same
                elif args["reward_norm_level"] == "dataset":
                    for idx, (scores, col) in enumerate(zip(reward_to_scores, reward_cols)):
                        mean, std = reward_to_stats[col]
                        reward_to_scores[idx] = [float((score - mean) / std) if score is not None else None for score in scores]

                # Aggregate scores across reward functions
                if args["square_rewards"]:
                    reward_to_scores = [[s and s ** 2 for s in scores] for scores in reward_to_scores]
                final_scores = []
                for cand_idx in range(len(cands)):
                    cand_scores = [scores[cand_idx] for scores in reward_to_scores if scores[cand_idx] is not None]
                    if cand_scores:
                        final_scores.append(sum(cand_scores) / len(cand_scores))
                    else:
                        final_scores.append(None)

                if args["pair_strategy"] == "max_delta":
                    # Get all preference pair candidates, exclude when scores within epsilon
                    # Take pairs with maximal reward deltas (https://arxiv.org/html/2502.14560v1)
                    pair_cands = []
                    for idx0, idx1 in combinations(range(len(cands)), 2):
                        if cands[idx0] == cands[idx1] or final_scores[idx0] is None or final_scores[idx1] is None:
                            continue
                        if final_scores[idx0] - final_scores[idx1] > args["epsilon"]:
                            pair_cands.append((final_scores[idx0] - final_scores[idx1], idx0, idx1))
                        elif final_scores[idx1] - final_scores[idx0] > args["epsilon"]:
                            pair_cands.append((final_scores[idx1] - final_scores[idx0], idx1, idx0))
                    pairs = sorted(pair_cands, reverse=True)[:args["num_pairs"]]
                elif args["pair_strategy"] == "each_wins":
                    # Allow each sample to be preferred once, negative is randomly selected from others
                    # Doing this in linear time would be a fun leetcode problem :)
                    pairs = []
                    for chosen_idx in range(len(cands)):
                        chosen_score = final_scores[chosen_idx]
                        neg_cands = []
                        for rejected_idx in range(len(cands)):
                            rejected_score = final_scores[rejected_idx]
                            if chosen_idx == rejected_idx or chosen_score is None or rejected_score is None:
                                continue
                            diff = chosen_score - rejected_score
                            if diff > args["epsilon"]:
                                neg_cands.append((diff, rejected_idx))
                        if neg_cands:
                            diff, rejected_idx = random.sample(neg_cands, 1)[0]
                            pairs.append((diff, chosen_idx, rejected_idx))

                if not pairs:
                    turns_without_pairs += 1
                    continue

                # Construct final inputs and preference pairs
                for _, chosen_idx, rejected_idx in pairs:
                    self.data.append({
                        "prompt": prompt,
                        "chosen": cands[chosen_idx],
                        "rejected": cands[rejected_idx]
                    })

        print(f"Num dialogues: {len(data)}, num turns: {num_turns} ({excluded} excluded), num pref pairs: {len(self.data)}, num turns without pairs: {turns_without_pairs}")
        for reward, stats in reward_to_stats.items():
            print(f"{reward}: mean - {stats[0]:.4f}, std - {stats[1]:.4f}")

def dpo(args):
    datasets.logging.set_verbosity_error() # Disable hashing warning from calling .map in DPOTrainer

    # Load or generate overgenerated data
    train_data, val_data = overgen_dpo_data(args)

    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])
    if not args["pt_model_name"]:
        print("Using base model as reference model")

    # Train
    train_dataset = DPODataset(train_data, tokenizer, args)
    val_dataset = DPODataset(val_data, tokenizer, args)
    config = DPOConfig(
        output_dir=get_checkpoint_path(args["model_name"]),
        num_train_epochs=args["epochs"],
        learning_rate=args["lr"],
        weight_decay=args["wd"],
        max_grad_norm=args["gc"],
        warmup_ratio=0.1,
        gradient_accumulation_steps=args["grad_accum_steps"],
        per_device_train_batch_size=args["train_batch_size"],
        per_device_eval_batch_size=args["val_batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        # report_to="wandb" if args["wandb"] else "none",
        report_to="none",
        # DPO-specific arguments
        model_adapter_name="default",
        ref_adapter_name="lora_ref" if args["pt_model_name"] else None,
        generate_during_eval=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args["val_batch_size"],
        beta=args["beta"]
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=HFDataset.from_list(train_dataset.data),
        eval_dataset=HFDataset.from_list(val_dataset.data),
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model()

    # Free up memory
    del trainer, base_model, model
    run_gc()

    # Test
    test({
        **args,
        **({"student_model": args["model_name"]} if args["role"] == "student" else {"tutor_model": args["model_name"]}),
        "temperature": None # Test-time temperature should be different from overgeneration temperature
    })

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--role", choices=["student", "tutor"], default="student")
    parser.add_argument("--subsample", type=float, default=0.2, help="Subsample dataset (0 for no subsampling), take from beginning of shuffle")
    parser.add_argument("--exclude_first_turns", type=int, default=5, help="Exclude first n turns of each dialogue")
    parser.add_argument("--test_on", choices=["val", "test"], default="val", help="Set to test on after training")
    parser.add_argument("--test_subsample", type=float, help="Subsample from test set")
    # Settings
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
    parser.add_argument("--rewards", default="cos,acts,kt,corr,err,tut", help="Comma-separated reward types for training")
    parser.add_argument("--rm_base_model", default="8b", help="Base model for fine-tuned reward models")
    parser.add_argument("--llmkt_model", default="llmkt-8b-ocean", help="Model to use for KT reward")
    parser.add_argument("--kcs_src", choices=["eedi"], default="eedi", help="Source of KCs for KT reward")
    parser.add_argument("--acts_model", default="acts-8b", help="Model to use for dialogue acts reward")
    parser.add_argument("--tutor_model", default="eedi-tutor-sft-8b", help="Model to use for tutor PPL reward")
    parser.add_argument("--student_model", help="Model to use for student outcomes reward")
    # Model
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, default=64, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank, only used if initializing from base model")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha, only used if initializing from base model")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter")
    parser.add_argument("--pair_strategy", choices=["max_delta", "each_wins"], default="max_delta", help="Strategy for forming preference pairs from candidates")
    parser.add_argument("--reward_norm_level", choices=["none", "group", "dataset"], default="none", help="Level to normalize rewards at")
    parser.add_argument("--square_rewards", action="store_true", help="Square each reward before aggregating")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Score threshold for DPO pair forming")
    parser.add_argument("--num_pairs", type=int, default=6, help="Maximum number of pairs to form per turn")
    # Overgeneration hyperparameters
    parser.add_argument("--force_regen", action="store_true", help="Regenerate data regardless of existing cache")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_gen_tokens", type=int, default=400)

    args = parser.parse_args().__dict__
    args["rewards"] = args["rewards"].split(",")
    assert all([r in ROLE_TO_REWARDS[args["role"]] for r in args["rewards"]])

    dpo(args)

if __name__ == "__main__":
    main()
