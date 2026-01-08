import torch
from transformers import PreTrainedTokenizer, TrainingArguments

from sim_student.utils import device, get_checkpoint_path

MAX_LEN = 6_000 # Exclude overly long prompts to avoid OOM

def get_training_args(args: dict):
    return TrainingArguments(
        output_dir=get_checkpoint_path(args["model_name"]),
        num_train_epochs=args["epochs"],
        learning_rate=args["lr"],
        weight_decay=args["wd"],
        max_grad_norm=args["gc"] or None,
        per_device_train_batch_size=args["train_batch_size"],
        gradient_accumulation_steps=args["grad_accum_steps"],
        per_device_eval_batch_size=args["val_batch_size"],
        eval_accumulation_steps=4,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # report_to="wandb" if args["wandb"] else "none"
        report_to="none"
    )

class TrainingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(
            [sample["prompt"] + sample["label"] + self.tokenizer.eos_token for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = tokens.input_ids
        attn_mask = tokens.attention_mask
        prompt_tokens = self.tokenizer(
            [sample["prompt"] for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_lens = prompt_tokens.attention_mask.sum(dim=1)
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        label_mask = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1) < prompt_lens.unsqueeze(1)
        labels[label_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "meta_data": batch
        }

class TestingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "left"
        tokens = self.tokenizer([sample["prompt"] for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "meta_data": batch
        }
