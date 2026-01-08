from typing import List
import argparse
import os
import random
from transformers import Trainer, PreTrainedTokenizer

from sim_student.model import get_base_model, get_model, get_tokenizer, generate_vllm
from sim_student.prompting import get_llm_prompt
from sim_student.data_loading import load_train_val_data, load_test_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.training_utils import TrainingCollator, get_training_args, MAX_LEN
from sim_student.utils import run_gc, initialize_seeds
from sklearn.metrics import precision_recall_fscore_support

ACTS = ["math answer", "seek information", "not understanding", "acknowledge", "off-topic"]

CLASSIFY_ACTS_SYSTEM_PROMPT = """Your task is to classify the dialogue acts for the last turn in the given dialogue."""

def get_acts_classifier_prompt(tokenizer: PreTrainedTokenizer, dialogue: Dialogue, ending_turn: int, last_turn_sub: str = None):
    prompt = get_llm_prompt(dialogue, ending_turn=ending_turn, last_turn_sub=last_turn_sub)
    chat = [
        {"role": "system", "content": CLASSIFY_ACTS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

class ActsClassifierDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, drop_long_seqs: bool = False, include_labels: bool = True):
        self.data = []
        num_errors = 0
        excluded = 0
        for dialogue_idx, dialogue in enumerate(data):
            # Skip if annotation error
            if include_labels and not dialogue["acts"]:
                num_errors += 1
                continue
            for turn_idx, turn in enumerate(dialogue["turns"]):
                if turn["role"] != "student":
                    continue
                prompt = get_acts_classifier_prompt(tokenizer, dialogue, turn_idx + 1)
                if drop_long_seqs and len(prompt) > MAX_LEN:
                    excluded += 1
                    continue
                self.data.append({
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn_idx,
                    "dialogue": dialogue,
                    "prompt": prompt,
                    "label": dialogue["acts"][f"turn {turn_idx + 1}"]["act"] if include_labels else None
                })
        print(f"Num dialogues: {len(data)}, Num turns: {len(self.data)} ({excluded} excluded), Num dialogues with annotation error: {num_errors}")

class ActsClassifierDatasetTurnLevel(DatasetBase):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, last_turn_sub_key: str):
        self.data = []
        for turn in data: # Unrolled dialogue data from TestingDataset, all turns are student turns
            self.data.append({
                "dialogue_idx": turn["dialogue_idx"],
                "turn_idx": turn["turn_idx"],
                "prompt": get_acts_classifier_prompt(tokenizer, turn, turn["turn_idx"] + 1, last_turn_sub=turn[last_turn_sub_key])
            })
        print(f"Number of data points: {len(self.data)}")

def train_acts_classifier(args: dict):
    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])

    # Load data
    train_data, val_data = load_train_val_data(args["dataset"])
    train_dataset = ActsClassifierDataset(train_data, tokenizer, drop_long_seqs=True)
    val_dataset = ActsClassifierDataset(val_data, tokenizer, drop_long_seqs=True)
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
    test_acts_classifier(args)

def test_acts_classifier(args: dict):
    # Load data
    test_data = load_test_data(args["dataset"])
    test_dataset = ActsClassifierDataset(test_data, get_tokenizer(args["base_model"]))

    # Generate predictions
    prompts = [sample["prompt"] for sample in test_dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 100})

    # Compute performance
    all_labels = []
    all_predictions = []
    correct = 0
    for sample, prediction in zip(test_dataset, predictions):
        label = sample["label"]
        prediction = prediction.strip()
        all_labels.append(label)
        all_predictions.append(prediction)
        if label == prediction:
            correct += 1
    result_str = f"Overall accuracy: {correct / len(test_dataset):.4f}\n"

    # Get per-act performance
    for act in sorted(set(all_labels)):
        act_labels_binary = [1 if l == act else 0 for l in all_labels]
        act_predictions_binary = [1 if p == act else 0 for p in all_predictions]
        p, r, f1, _ = precision_recall_fscore_support(act_labels_binary, act_predictions_binary, average="binary")
        result_str += f"{act} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n"

    # Save results
    print(result_str)
    os.makedirs("results/acts_classification", exist_ok=True)
    with open(f"results/acts_classification/metrics_{args['dataset']}_{args['model_name']}.txt", "w") as file:
        file.write(result_str + "\n")

def eval_acts_turn_level(data: List[dict], args: dict, last_turn_sub_key: str):
    assert args["model_name"]
    dataset = ActsClassifierDatasetTurnLevel(data, get_tokenizer(args["base_model"]), last_turn_sub_key)
    prompts = [sample["prompt"] for sample in dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 100})
    return predictions

def eval_acts_dia_level(data: List[Dialogue], args: dict):
    assert args["model_name"]
    dataset = ActsClassifierDataset(data, get_tokenizer(args["base_model"]), include_labels=False)
    prompts = [sample["prompt"] for sample in dataset]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 100})
    results = [
        {"dialogue_idx": sample["dialogue_idx"], "turn_idx": sample["turn_idx"], "act": act}
        for sample, act in zip(dataset, predictions)
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
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    # Testing settings
    parser.add_argument("--test_batch_size", type=int, default=32)
    args = parser.parse_args().__dict__

    if args["mode"] == "train":
        train_acts_classifier(args)
    elif args["mode"] == "test":
        test_acts_classifier(args)

if __name__ == "__main__":
    main()
