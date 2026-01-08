import os
import gc
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

def run_gc():
    gc.collect()
    torch.cuda.empty_cache()

def bool_type(x: str):
    return x != "0"

def get_checkpoint_path(model_name: str):
    os.makedirs("saved_models", exist_ok=True)
    return f"saved_models/{model_name}"

def get_suffix(args: dict):
    suffix = f"{args['dataset']}_{args['test_on']}"

    base_model = args["base_model"].replace('/', '-')
    if args["baseline"]:
        student_model = f"{args['baseline']}-{args['baseline_model']}"
    else:
        student_model = args.get("student_model") or base_model
    tutor_model = args.get("tutor_model") or base_model

    if args["level"] == "turn":
        model_name = student_model if args["role"] == "student" else tutor_model
        suffix += f"_{args['role']}-turn_{model_name}"
    else:
        suffix += f"_full_{student_model}_{tutor_model}"

    if args["test_subsample"]:
        suffix += f"_tss{args['test_subsample']}"
    return suffix

def merge_defaults(args: dict, defaults: dict):
    for k, v in defaults.items():
        if args.get(k) is None:
            args[k] = v
    return args
