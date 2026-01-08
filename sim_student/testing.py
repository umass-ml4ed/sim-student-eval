import argparse
from typing import List, Optional
import os
import random
import pandas as pd
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from vllm import LLM
from sentence_transformers import SentenceTransformer

from sim_student.model import get_base_model, get_model, get_tokenizer, BASE_MODEL_NAME_MAP, generate_vllm
from sim_student.data_loading import load_test_data, load_train_val_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.prompting import get_local_prompt, get_llm_prompt, END_OF_DIALOGUE, get_prompting_function, BASELINE_TO_SYS_PROMPT
from sim_student.eval import eval
from sim_student.training_utils import TestingCollator
from sim_student.utils import get_suffix, run_gc, merge_defaults, initialize_seeds, bool_type

MAX_RESPONSE_LEN = 500

class TestingDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], role: str, tokenizer: Optional[PreTrainedTokenizer], args: dict, include_dialogue: bool = True):
        self.data = []
        for dialogue_idx, dialogue in enumerate(data):
            starting_turn = 0 if role == dialogue["turns"][0]["role"] else 1
            for turn_idx in range(starting_turn, len(dialogue["turns"]), 2):
                if tokenizer is not None:
                    prompt = get_local_prompt(dialogue, role, tokenizer, ending_turn=turn_idx, persona_type=args["persona"])
                else:
                    prompt = get_llm_prompt(dialogue, ending_turn=turn_idx, persona_type=args["persona"])
                self.data.append({
                    **(dialogue if include_dialogue else {}),
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn_idx,
                    "gt_turn": dialogue["turns"][turn_idx]["content"],
                    "prompt": prompt
                })
        print(f"Num dialogues: {len(data)}, num turns: {len(self.data)}")

def test_hf(dataset: TestingDataset, args: dict):
    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    if args["role"] == "student":
        model = get_model(base_model, True, model_name=args["student_model"], quantize=args["quantize"])
    else:
        model = get_model(base_model, True, model_name=args["tutor_model"], quantize=args["quantize"])

    # Set generation config based on arguments
    if args["temperature"]:
        model.generation_config.update(do_sample=True, temperature=args["temperature"], top_p=0.95)
    else:
        model.generation_config.update(do_sample=False, temperature=None, top_p=None)

    # Generate tutor turns
    collator = TestingCollator(tokenizer)
    data_loader = DataLoader(dataset, args["test_batch_size"], collate_fn=collator, shuffle=False)
    results: List[str] = []
    for batch in tqdm(data_loader):
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args["max_gen_tokens"]
        )
        pred_turns = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        results.extend(pred_turns)

    # Free up memory
    del base_model, model
    run_gc()

    return results

def test_vllm(dataset: TestingDataset, args: dict):
    prompts = [sample["prompt"] for sample in dataset]
    if args["role"] == "student":
        adapter = ("student", 1, args["student_model"])
    else:
        adapter = ("tutor", 2, args["tutor_model"])
    return generate_vllm(args["base_model"], prompts, adapter, {"temperature": args["temperature"], "top_p": 0.95, "max_tokens": args["max_gen_tokens"]})

def get_icl_examples(test_data: List[Dialogue], args: dict):
    train_data, _ = load_train_val_data(args["dataset"])
    emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    test_personas = [dialogue["freeform_persona"] for dialogue in test_data]
    train_personas = [dialogue["freeform_persona"] for dialogue in train_data]
    test_persona_embs = emb_model.encode(test_personas, batch_size=16, show_progress_bar=True)
    train_persona_embs = emb_model.encode(train_personas, batch_size=16, show_progress_bar=True)
    cos_sim = np.matmul(test_persona_embs, train_persona_embs.T)
    example_idxs = np.argmax(cos_sim, axis=1)
    del emb_model, test_persona_embs, train_persona_embs
    run_gc()
    return [get_llm_prompt(train_data[idx]) for idx in example_idxs]

def add_examples_to_prompts(prompts: List[str], examples: List[str], dialogue_idxs: List[int]):
    for idx, dialogue_idx in enumerate(dialogue_idxs):
        example = examples[dialogue_idx]
        prompts[idx] = f"#### Begin Example ####\n\n{example}\n\n#### End Example ####\n\n" + prompts[idx]

def test_turn_level(data: List[Dialogue], args: dict):
    tokenizer = None if args["baseline"] else get_tokenizer(args["base_model"])
    dataset = TestingDataset(data, args["role"], tokenizer, args)

    # Generate outputs
    if args["baseline"]:
        prompting_fn = get_prompting_function({"engine": "openai", "annotation_model": args["baseline_model"],
                                               "batch_api": args["batch_api"], "temperature": args["temperature"]})
        prompts = [sample["prompt"] for sample in dataset]
        # Add examples for ICL baseline
        if args["baseline"] == "icl":
            examples = get_icl_examples(data, args)
            add_examples_to_prompts(prompts, examples, [sample["dialogue_idx"] for sample in dataset])
        outputs = prompting_fn(prompts, BASELINE_TO_SYS_PROMPT[args["baseline"]])
    else:
        if args["vllm"]:
            outputs = test_vllm(dataset, args)
        else:
            outputs = test_hf(dataset, args)

    # Merge with dataset
    results = []
    for sample, pred_turn in zip(dataset, outputs):
        results.append({**sample, "pred_turn": pred_turn.replace(END_OF_DIALOGUE, "")[:MAX_RESPONSE_LEN]})

    return results

def test_full(data: List[Dialogue], args: dict):
    # Setup for baseline
    if args["baseline"]:
        baseline_fn = get_prompting_function({"engine": "openai", "use_azure": False, "temperature": args["temperature"],
                                              "annotation_model": args["baseline_model"], "batch_api": args["batch_api"]})
        if args["baseline"] == "icl":
            examples = get_icl_examples(data, args)

    # Base model for student/tutor (if baseline only used for tutor)
    tokenizer = get_tokenizer(args["base_model"])
    student_base_model = LLM(BASE_MODEL_NAME_MAP.get(args["base_model"], args["base_model"]), enable_lora=True, max_lora_rank=256)
    if args["tutor_base_model"] and args["tutor_base_model"] != args["base_model"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        tutor_base_model = LLM(BASE_MODEL_NAME_MAP.get(args["tutor_base_model"], args["tutor_base_model"]), enable_lora=True, max_lora_rank=256)
    else:
        tutor_base_model = student_base_model

    # Use given data distribution but start dialogue from scratch
    gen_data = [{**dialogue, "gt_turns": dialogue["turns"], "turns": [], "done": False} for dialogue in data]

    # Generate dialogues one turn at a time
    for turn_idx in tqdm(range(args["max_turns"])):
        unfinished_dia_idxs = [dia_idx for dia_idx, dialogue in enumerate(gen_data) if not dialogue["done"]]
        if not unfinished_dia_idxs:
            break

        for role in ["student", "tutor"]:
            print(f"Generating {role} utterances for turn {turn_idx + 1}...")
            # Get dialogues to generate current role for
            if turn_idx == 0:
                # For first turn, use ground truth data to determine if student/tutor goes first
                cur_role_dia_idxs = [dia_idx for dia_idx, dialogue in enumerate(gen_data) if dialogue["gt_turns"][0]["role"] == role]
            else:
                # For later turns, alternate roles
                cur_role_dia_idxs = [dia_idx for dia_idx in unfinished_dia_idxs if gen_data[dia_idx]["turns"][turn_idx - 1]["role"] != role]

            if turn_idx == 0 and role == "tutor":
                # For dialogues that start with tutor turns, use the ground-truth turn to start the dialogue
                outputs = [gen_data[dia_idx]["gt_turns"][0]["content"] for dia_idx in cur_role_dia_idxs]
            elif cur_role_dia_idxs:
                # Simulate utterances for this turn
                if args["baseline"] and role == "student":
                    # Use LLM client for baselines
                    prompts = [
                        get_llm_prompt(gen_data[dia_idx], ending_turn=turn_idx, persona_type=args["persona"])
                        for dia_idx in cur_role_dia_idxs
                    ]
                    if args["baseline"] == "icl":
                        add_examples_to_prompts(prompts, examples, cur_role_dia_idxs)
                    outputs = baseline_fn(prompts, BASELINE_TO_SYS_PROMPT[args["baseline"]])
                else:
                    # Use vllm for local models
                    prompts = [
                        get_local_prompt(gen_data[dia_idx], role, tokenizer, ending_turn=turn_idx, persona_type=args["persona"])
                        for dia_idx in cur_role_dia_idxs
                    ]
                    if role == "student":
                        adapter = args["student_model"] and ("student", 1, args["student_model"])
                        base_model = student_base_model
                    else:
                        adapter = args["tutor_model"] and ("tutor", 2, args["tutor_model"])
                        base_model = tutor_base_model
                    outputs = generate_vllm(base_model, prompts, adapter, {"temperature": args["temperature"], "top_p": 0.95, "max_tokens": args["max_gen_tokens"]})
            else:
                outputs = []

            for dia_idx, output in zip(cur_role_dia_idxs, outputs):
                if END_OF_DIALOGUE in output:
                    gen_data[dia_idx]["done"] = True
                    output = output.replace(END_OF_DIALOGUE, "")[:MAX_RESPONSE_LEN]
                gen_data[dia_idx]["turns"].append({"role": role, "content": output})

    del student_base_model, tutor_base_model
    return gen_data

def test(args: dict):
    defaults = DEFAULT_ARGS
    if args.get("level") == "full":
        defaults["temperature"] = 0.6
    else:
        defaults["temperature"] = 0.0
    args = merge_defaults(args, defaults)
    print(args)

    # Load data
    if args["test_on"] == "test":
        data = load_test_data(args["dataset"])
    else:
        _, data = load_train_val_data(args["dataset"])
    if args["truncate"]:
        data = data[:args["truncate"]]
    if args["test_subsample"]:
        data = random.Random(221).sample(data, int(len(data) * args["test_subsample"]))

    if args["level"] == "full":
        results = test_full(data, args)
    else:
        results = test_turn_level(data, args)
    run_gc()

    # Save results and run evals
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_filename = f"results/outputs_{get_suffix(args)}.csv"
    results_df.to_csv(results_filename, index=False)
    eval(results_df, {**args, "input_file": results_filename})

DEFAULT_ARGS = {
    "dataset": "eedi",
    "truncate": 0,
    "level": "turn",
    "role": "student",
    "test_on": "val",
    "base_model": "8b",
    "baseline": None,
    "max_turns": 40,
    "max_gen_tokens": 400,
    "test_batch_size": 32,
    "vllm": True,
    "batch_api": True,
    # Default temperature set based on level
}

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--test_on", choices=["val", "test"])
    parser.add_argument("--truncate", type=int, help="Number of dialogue to truncate to, keep all if unset")
    parser.add_argument("--test_subsample", type=float, help="Subsample from test set")
    parser.add_argument("--level", choices=["turn", "full"])
    parser.add_argument("--role", choices=["student", "tutor"])
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
    parser.add_argument("--base_model")
    parser.add_argument("--tutor_base_model", help="If tutor has different base model than student")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--student_model")
    parser.add_argument("--tutor_model")
    parser.add_argument("--baseline", choices=["zs-eth", "persona-ff", "persona-ocean", "icl", "reasoning"])
    parser.add_argument("--baseline_model", default="gpt-4.1")
    # Generation args
    parser.add_argument("--max_turns", type=int, help="Maximum number of turns for full dialogue generation")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max_gen_tokens", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--vllm", type=bool_type, default=True, help="Use vllm for decoding")
    parser.add_argument("--batch_api", type=bool_type, default=True, help="Use batch api for baseline")
    args = parser.parse_args().__dict__

    # Ensure persona type aligns with baseline prompt
    if args["baseline"] and args["baseline"].startswith("persona"):
        persona_type = args["baseline"].split("-")[1]
        if persona_type == "ff":
            args["persona"] = "freeform"
        if persona_type == "ocean":
            args["persona"] = "ocean"

    test(args)

if __name__ == "__main__":
    main()
