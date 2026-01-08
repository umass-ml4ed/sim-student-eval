import argparse
import os
from dataclasses import dataclass
from typing import List, Union, Optional
from peft import PeftModel
import torch
from tqdm import tqdm
import numpy as np
from transformers import PreTrainedTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from sim_student.prompting import get_llm_prompt
from sim_student.data_loading import load_train_val_data, load_test_data
from sim_student.data_utils import Dialogue
from sim_student.model import BASE_MODEL_NAME_MAP, get_base_model, get_model
from sim_student.training_utils import MAX_LEN
from sim_student.utils import run_gc

from dialogue_kt.training import get_lmkt_loss_packed
from dialogue_kt.kt_data_loading import LMKTCollatorPacked, get_dataloader, DatasetBase
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.utils import get_checkpoint_path, initialize_seeds


LLMKT_SYSTEM_PROMPT = """You are an experienced math teacher. You are given a dialogue between a student and teacher where a student is trying to solve a math problem. Your job is to predict if the student has a particular knowledge component (KC) at the current point in the dialogue. Please follow these instructions carefully when making your prediction:
- The student will need to possess this KC in order to respond correctly to the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this KC or not.
- Only respond with a single word, "True" or "False"."""

def get_llmkt_prompt(tokenizer: PreTrainedTokenizer, dialogue: Dialogue, ending_turn: int, kcs: List[str], kcs_src: str, persona_type: str, last_turn_sub: Optional[Union[str, List[str]]] = None):
    # Based on dialogue_kt/prompting/kt_user_prompt and dialogue_kt/kt_data_loading/LMKTDatasetPacked
    user_prompt = get_llm_prompt(dialogue, kcs_src=kcs_src, ending_turn=ending_turn, last_turn_sub=last_turn_sub, persona_type=persona_type) + "\n\nKC:"
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": LLMKT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], tokenize=False)
    # Create all possible knowledge component continuations
    kc_conts = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": kc},
            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
        ], tokenize=False)
        for kc in kcs
    ]
    kc_conts = [" " + cont.split("user<|end_header_id|>\n\n")[1] for cont in kc_conts]
    prompt += "".join(kc_conts)
    return prompt

class LLMKTDatasetFull(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, args: dict):
        self.data = []
        failed_dialogues = 0
        excluded = 0
        for dialogue_idx, dialogue in enumerate(data):
            dia_corr = dialogue["correctness"]
            if args["kcs_src"] == "eedi":
                dia_kcs = dialogue["eedi_kcs"]
            if not dia_corr or "error" in dia_corr or not dia_kcs or "error" in dia_kcs:
                failed_dialogues += 1
                continue
            for turn_idx, turn in enumerate(dialogue["turns"]):
                if turn["role"] != "student" or turn_idx == 0: # If first turn is student then won't have kcs from previous tutor turn
                    continue
                correct = dia_corr[f"turn {turn_idx + 1}"]["correct"]
                kcs = dia_kcs[f"turn {turn_idx}"]["kcs"]
                if correct is None or not kcs:
                    continue
                cur_ending_turns = []
                if args["condition"] in ("s", "both"):
                    cur_ending_turns.append(turn_idx - 1) # Prompt up to and including previous student turn
                if args["condition"] in ("t", "both"):
                    cur_ending_turns.append(turn_idx) # Prompt up to and including current tutor turn
                for ending_turn in cur_ending_turns:
                    prompt = get_llmkt_prompt(tokenizer, dialogue, ending_turn, kcs, args["kcs_src"], args["persona"])
                    if len(prompt) < MAX_LEN:
                        self.data.append({
                            "dialogue_idx": dialogue_idx,
                            "turn_idx": turn_idx,
                            "prompt": prompt,
                            "label": correct,
                            "kcs": kcs
                        })
                    else:
                        excluded += 1
        print(f"Number of dialogues: {len(data)}, number of data points: {len(self.data)} ({excluded} excluded), number of failed dialogues: {failed_dialogues}")

class LLMKTDatasetTurnLevel(DatasetBase):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, last_turn_sub_key: Union[str, List[str]], kcs_src: str, persona_type: str):
        if isinstance(last_turn_sub_key, str):
            last_turn_sub_key = [last_turn_sub_key]
        self.data = []
        for turn in data: # Unrolled dialogue data from TestingDataset, all turns are student turns
            # Use all relevant KCs for the question
            if kcs_src == "eedi":
                kcs = [kc for kc, level in turn["subjects"] if level == 3] + ["Default"]
            # Prompt up to and including current student turn
            # Use ground truth (last_turn_sub_key=gt_turn) or predicted (last_turn_sub_key=pred_turn) turn text
            prompt = get_llmkt_prompt(tokenizer, turn, turn["turn_idx"] + 1, kcs, kcs_src, persona_type,
                                      last_turn_sub=[turn[sub_key] for sub_key in last_turn_sub_key])
            self.data.append({
                "dialogue_idx": turn["dialogue_idx"],
                "turn_idx": turn["turn_idx"],
                "prompt": prompt,
                "label": False, # Dummy value, only used when training/testing model
                "kcs": kcs
            })
        print(f"Number of data points: {len(self.data)}")

@dataclass
class DialogueKTArgs:
    agg: str = "mean-ar"

def get_loss(model, batch, true_token, false_token):
    batch["labels"] = batch["labels"].type(model.dtype) # Ensure labels are correct dtype
    return get_lmkt_loss_packed(model, batch, true_token, false_token, DialogueKTArgs())

def train_llmkt(args: dict):
    # Load language model with trainable LoRA adapters
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])
    model.print_trainable_parameters()

    # Load and split dataset, annotated with correctness and KCs
    train_data, val_data = load_train_val_data(args["dataset"])
    train_dataset = LLMKTDatasetFull(train_data, tokenizer, args)
    val_dataset = LLMKTDatasetFull(val_data, tokenizer, args)
    collator = LMKTCollatorPacked(tokenizer)
    train_dataloader = get_dataloader(train_dataset, collator, args["batch_size"], True)
    val_dataloader = get_dataloader(val_dataset, collator, args["batch_size"], False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Do training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
    best_val_loss = None
    for epoch in range(args["epochs"]):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _, _ = get_loss(model, batch, true_token, false_token)
            total_train_loss += loss.item()
            loss = loss / args["grad_accum_steps"]
            loss.backward()
            if (batch_idx + 1) % args["grad_accum_steps"] == 0 or batch_idx == len(train_dataloader) - 1:
                if args["gc"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["gc"])
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _, _ = get_loss(model, batch, true_token, false_token)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model.save_pretrained(get_checkpoint_path(args["model_name"]))
            best_val_loss = avg_val_loss

    del base_model, model
    run_gc()

    return test_llmkt(args)

def get_llmkt_model_for_eval(args: dict):
    assert args["model_name"]
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, True, model_name=args["model_name"], quantize=args["quantize"])
    model.eval()
    return model, tokenizer

def llmkt_eval(model: PeftModel, tokenizer: PreTrainedTokenizer, dataset: Union[LLMKTDatasetFull, LLMKTDatasetTurnLevel], args: dict):
    collator = LMKTCollatorPacked(tokenizer)
    test_dataloader = get_dataloader(dataset, collator, args["batch_size"], False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Collect meta data and predicted KC/correctness probabilities for test set
    total_loss = 0
    results = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            loss, kc_probs, corr_probs = get_loss(model, batch, true_token, false_token)
        total_loss += loss.item()
        for sample_idx, sample in enumerate(batch["meta_data"]):
            results.append({
                "dialogue_idx": sample["dialogue_idx"],
                "turn_idx": sample["turn_idx"],
                "label": sample.get("label"),
                "pred": corr_probs[sample_idx].item(),
                "kc_probs": kc_probs[sample_idx]
            })
    avg_loss = total_loss / len(test_dataloader)
    return results, avg_loss

def test_llmkt(args: dict):
    if args["test_on"] == "val":
        _, test_data = load_train_val_data(args["dataset"])
    else:
        test_data = load_test_data(args["dataset"])
    model, tokenizer = get_llmkt_model_for_eval(args)
    test_dataset = LLMKTDatasetFull(test_data, tokenizer, args)
    results, loss = llmkt_eval(model, tokenizer, test_dataset, args)
    all_labels = [turn["label"] for turn in results]
    all_preds = [turn["pred"] for turn in results]
    all_preds_rounded = np.round(all_preds)
    accuracy = accuracy_score(all_labels, all_preds_rounded)
    auc = roc_auc_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds_rounded, average="binary")
    result_str = f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    print(result_str)
    os.makedirs("results/llmkt", exist_ok=True)
    with open(f"results/llmkt/metrics_{args['dataset']}_{args['model_name']}_{args['test_on']}.txt", "w") as f:
        f.write(result_str + "\n")

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run the script in")
    parser.add_argument("--dataset", type=str, choices=["eedi"], default="eedi", help="Which dataset to use")
    parser.add_argument("--test_on", choices=["val", "test"], default="val", help="Which split to test on")
    parser.add_argument("--condition", choices=["s", "t", "both"], default="s", help="Last turn in dialogue history to condition on: last student turn (s), current tutor turn (t), or both")
    parser.add_argument("--kcs_src", choices=["eedi"], default="eedi", help="Data source for KCs")
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="ocean")
    parser.add_argument("--base_model", type=str, default="8b", help="HuggingFace base model for LLMKT")
    parser.add_argument("--model_name", type=str, help="Name of model to save for LLMKT")
    parser.add_argument("--batch_size", type=int, default=1, help="Model batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--grad_accum_steps", type=int, default=64, help="Steps to accumulate gradients")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model")
    args = parser.parse_args()
    args = args.__dict__
    args["base_model"] = BASE_MODEL_NAME_MAP.get(args["base_model"], args["base_model"])

    if args["mode"] == "train":
        train_llmkt(args)
    elif args["mode"] == "test":
        test_llmkt(args)

if __name__ == "__main__":
    main()
