import argparse
from typing import List
import pandas as pd
from transformers import PreTrainedTokenizer, Trainer

from sim_student.model import get_base_model, get_model
from sim_student.data_loading import load_train_val_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.prompting import get_local_prompt
from sim_student.testing import test
from sim_student.training_utils import get_training_args, MAX_LEN
from sim_student.utils import device, run_gc, merge_defaults

# For Llama 3, consts since no tokenizer variables point to these
BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007

class SFTCombinedDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], role: str, tokenizer: PreTrainedTokenizer, args: dict):
        self.data = []
        excluded = 0
        num_turns = 0
        for dialogue in data:
            num_turns += round(len(dialogue["turns"]) / 2)
            prompt = get_local_prompt(dialogue, role, tokenizer, persona_type=args["persona"])
            if len(prompt) < MAX_LEN:
                self.data.append({**dialogue, "prompt": prompt})
            else:
                excluded += 1
        print(f"Num dialogues: {len(self.data)} ({excluded} excluded), num turns: {num_turns}")

class SFTCombinedCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer([sample["prompt"] for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # Create labels - mask out all but assistant turns
        labels = tokens.input_ids.clone()
        labels[tokens.attention_mask == 0] = -100 # Mask padding region
        for idx in range(len(labels)):
            boh_idxs = (labels[idx] == BOH_TOKEN_ID).nonzero()
            eoh_idxs = (labels[idx] == EOH_TOKEN_ID).nonzero()
            labels[idx, :eoh_idxs[2] + 1] = -100 # Mask labels up to end of first assistant header
            for header_ct in range(3, len(boh_idxs), 2):
                # Mask labels between start of each user header to end of subsequent assistant header
                end_idx = eoh_idxs[header_ct + 1] if header_ct + 1 < len(eoh_idxs) else labels.shape[1] # In case dialogue ends with a user turn
                labels[idx, boh_idxs[header_ct] : end_idx + 1] = -100
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels
        }

def sft(args):
    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])

    # Load data
    train_data, val_data = load_train_val_data(args["dataset"])
    if args["subsample"]:
        train_data = train_data[int(len(train_data) * (1 - args["subsample"])):]
        val_data = val_data[int(len(val_data) * (1 - args["subsample"])):]
    train_dataset = SFTCombinedDataset(train_data, args["role"], tokenizer, args)
    val_dataset = SFTCombinedDataset(val_data, args["role"], tokenizer, args)
    collator = SFTCombinedCollator(tokenizer)

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
    test({
        **args,
        **({"student_model": args["model_name"]} if args["role"] == "student" else {"tutor_model": args["model_name"]})
    })

def get_default_args(args: dict):
    if args["base_model"] == "3b":
        return {
            "train_batch_size": 2,
            "val_batch_size": 4,
            "grad_accum_steps": 32
        }
    return {
        "train_batch_size": 1,
        "val_batch_size": 2,
        "grad_accum_steps": 64
    }

def main():
    parser = argparse.ArgumentParser()
    # Data and settings
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--role", choices=["student", "tutor"], default="student")
    parser.add_argument("--subsample", type=float, default=0, help="Subsample dataset (0 for no subsampling), take from end of shuffle")
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
    # Model
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training settings
    parser.add_argument("--train_batch_size", type=int, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args().__dict__
    args = merge_defaults(args, get_default_args(args))

    sft(args)

if __name__ == "__main__":
    main()
