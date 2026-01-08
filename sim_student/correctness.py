from typing import List
import argparse
import os
import random
from transformers import Trainer, PreTrainedTokenizer

from sim_student.model import get_base_model, get_model, get_tokenizer, generate_vllm
from sim_student.prompting import get_llm_prompt, correct_to_str
from sim_student.data_loading import load_train_val_data, load_test_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.training_utils import TrainingCollator, get_training_args, MAX_LEN
from sim_student.utils import run_gc, initialize_seeds

CLASSIFY_CORRECTNESS_SYSTEM_PROMPT = """Your task is to classify whether the last student turn in the given dialogue is one of: "correct", "incorrect", or "na"."""

def get_correctness_classifier_prompt(tokenizer: PreTrainedTokenizer, dialogue: Dialogue, ending_turn: int, last_turn_sub: str = None):
    prompt = get_llm_prompt(dialogue, ending_turn=ending_turn, last_turn_sub=last_turn_sub)
    chat = [
        {"role": "system", "content": CLASSIFY_CORRECTNESS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

class CorrectnessClassifierDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, drop_long_seqs: bool = False, include_labels: bool = True):
        self.data = []
        num_errors = 0
        excluded = 0
        for dialogue_idx, dialogue in enumerate(data):
            # Skip if annotation error
            if include_labels and (not dialogue["correctness"] or "error" in dialogue["correctness"]):
                num_errors += 1
                continue
            for turn_idx, turn in enumerate(dialogue["turns"]):
                if turn["role"] != "student":
                    continue
                prompt = get_correctness_classifier_prompt(tokenizer, dialogue, turn_idx + 1)
                if drop_long_seqs and len(prompt) > MAX_LEN:
                    excluded += 1
                    continue
                self.data.append({
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn_idx,
                    "dialogue": dialogue,
                    "prompt": prompt,
                    "label": correct_to_str(dialogue["correctness"][f"turn {turn_idx + 1}"]["correct"]) if include_labels else None
                })
        print(f"Num dialogues: {len(data)}, Num turns: {len(self.data)} ({excluded} excluded), Num dialogues with annotation error: {num_errors}")

class CorrectnessClassifierDatasetTurnLevel(DatasetBase):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, last_turn_sub_key: str):
        self.data = []
        for turn in data: # Unrolled dialogue data from TestingDataset, all turns are student turns
            self.data.append({
                "dialogue_idx": turn["dialogue_idx"],
                "turn_idx": turn["turn_idx"],
                "prompt": get_correctness_classifier_prompt(tokenizer, turn, turn["turn_idx"] + 1, last_turn_sub=turn[last_turn_sub_key])
            })
        print(f"Number of data points: {len(self.data)}")

def train_correctness_classifier(args: dict):
    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])

    # Load data
    train_data, val_data = load_train_val_data(args["dataset"])
    train_dataset = CorrectnessClassifierDataset(train_data, tokenizer, drop_long_seqs=True)
    val_dataset = CorrectnessClassifierDataset(val_data, tokenizer, drop_long_seqs=True)
    if args["subsample"]:
        for dataset in [train_dataset, val_dataset]:
            dataset.data = random.sample(dataset.data, int(len(dataset) * args["subsample"]))
            print(f"New dataset size: {len(dataset)}")
    collator = TrainingCollator(tokenizer)

    # Train
    trainer = Trainer(
        model=model,
        args=get_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model()
    del trainer, base_model, model
    run_gc()

    # Test
    test_correctness_classifier(args)

def test_correctness_classifier(args: dict):
    # Load data
    test_data = load_test_data(args["dataset"])
    test_dataset = CorrectnessClassifierDataset(test_data, get_tokenizer(args["base_model"]))

    # Generate predictions
    prompts = [sample["prompt"] for sample in test_dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 10})

    # Compute performance
    total = len(test_dataset)
    correct_count = 0
    class_counts = {"correct": 0, "incorrect": 0, "na": 0}
    correct_by_class = {"correct": 0, "incorrect": 0, "na": 0}

    for sample, pred in zip(test_dataset, predictions):
        label = sample["label"]
        class_counts[label] += 1
        if pred == label:
            correct_count += 1
            correct_by_class[label] += 1

    overall_acc = correct_count / total if total > 0 else 0.0
    per_class_strs = []
    for cls in ("correct", "incorrect", "na"):
        cnt = class_counts[cls]
        acc = (correct_by_class[cls] / cnt) if cnt > 0 else None
        per_class_strs.append(f"{cls}: count={cnt}, acc={acc:.4f}" if acc is not None else f"{cls}: count=0, acc=N/A")
    result_str = f"Overall accuracy: {overall_acc:.4f}. " + " ".join(per_class_strs)

    print(result_str)
    os.makedirs("results/correctness_classification", exist_ok=True)
    with open(f"results/correctness_classification/metrics_{args['dataset']}_{args.get('model_name','model')}.txt", "w") as file:
        file.write(result_str + "\n")

def eval_correctness_turn_level(data: List[dict], args: dict, last_turn_sub_key: str):
    assert args["model_name"]
    dataset = CorrectnessClassifierDatasetTurnLevel(data, get_tokenizer(args["base_model"]), last_turn_sub_key)
    prompts = [sample["prompt"] for sample in dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 10})
    return predictions

def eval_correctness_dia_level(data: List[Dialogue], args: dict):
    assert args["model_name"]
    dataset = CorrectnessClassifierDataset(data, get_tokenizer(args["base_model"]), include_labels=False)
    prompts = [sample["prompt"] for sample in dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 10})
    results = [
        {"dialogue_idx": sample["dialogue_idx"], "turn_idx": sample["turn_idx"], "correct": correct}
        for sample, correct in zip(dataset, predictions)
    ]
    return results

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    # Data and settings
    parser.add_argument("--dataset", default="eedi")
    # Model
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training settings
    parser.add_argument("--subsample", type=float, help="Portion of train/val data to keep")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, default=64, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    # Testing settings
    parser.add_argument("--test_batch_size", type=int, default=32)
    args = parser.parse_args().__dict__

    if args["mode"] == "train":
        train_correctness_classifier(args)
    elif args["mode"] == "test":
        test_correctness_classifier(args)

if __name__ == "__main__":
    main()
