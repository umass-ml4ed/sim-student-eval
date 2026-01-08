import random
import json
import os
from typing import List, Optional
import pandas as pd
from ast import literal_eval

from sim_student.data_utils import Dialogue, Turn

###### Generic ######

def parse_obj(obj: str):
    try:
        return literal_eval(obj)
    except (ValueError, SyntaxError):
        return None

def read_csv(path: str, extra_obj_cols: Optional[List[str]] = None):
    obj_cols = ["turns", "subjects", "gt_turns", "correctness", "ocean_persona", "question_annotation", "eedi_kcs", "acts"]
    if extra_obj_cols:
        obj_cols.extend(extra_obj_cols)
    df = pd.read_csv(path, converters={col: parse_obj for col in obj_cols})
    return df

def load_csv_as_dict(path: str, obj_cols: Optional[List[str]] = None):
    if not os.path.exists(path):
        return None
    df = read_csv(path, obj_cols)
    return df.to_dict("records")

def save_annotated_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, dataset: str, suffix: str = ""):
    os.makedirs(f"data/annotated/{dataset}", exist_ok=True)
    for data, split in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        data.to_csv(f"data/annotated/{dataset}/{split}{suffix}.csv", index=False)

def load_annotated_data(dataset: str, split: str, suffix: str = "", drop_unsolvable: bool = True) -> Optional[List[Dialogue]]:
    data = load_csv_as_dict(f"data/annotated/{dataset}/{split}{suffix}.csv")
    # Drop unsolvable questions
    if drop_unsolvable and data and "question_annotation" in data[0]:
        data = [dialogue for dialogue in data if dialogue["question_annotation"] and dialogue["question_annotation"].get("solvable")]
    return data

def load_train_val_data(dataset: str, annotation_model: str = "gpt-4.1", drop_unsolvable: bool = True):
    train_data = load_annotated_data(dataset, "train", f"_{annotation_model}", drop_unsolvable)
    val_data = load_annotated_data(dataset, "val", f"_{annotation_model}", drop_unsolvable)
    if train_data is not None and val_data is not None:
        return train_data, val_data
    if dataset == "eedi":
        return load_eedi_train_val_data()
    raise Exception(f"Invalid dataset {dataset}")

def load_test_data(dataset: str, annotation_model: str = "gpt-4.1", drop_unsolvable: bool = True):
    test_data = load_annotated_data(dataset, "test", f"_{annotation_model}", drop_unsolvable)
    if test_data is not None:
        return test_data
    if dataset == "eedi":
        return load_eedi_test_data()
    raise Exception(f"Invalid dataset {dataset}")

def load_overgenerated_data(dataset: str, split: str, descriptor: str, obj_cols: List[str]) -> Optional[List[Dialogue]]:
    return load_csv_as_dict(f"data/overgenerated/{dataset}/{split}_{descriptor}.csv", obj_cols)

def save_overgenerated_data(data: List[Dialogue], dataset: str, split: str, descriptor: str):
    os.makedirs(f"data/overgenerated/{dataset}", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(f"data/overgenerated/{dataset}/{split}_{descriptor}.csv", index=False)


###### EEDI ######

def load_eedi_metadata():
    metadata = {}
    question_df = pd.read_csv("data/Question-Anchored-Tutoring-Dialogues-2k/dq-question-metadata.csv")
    subjects_df = pd.read_csv("data/Question-Anchored-Tutoring-Dialogues-2k/dialogue-subjects.csv")
    for name, group in question_df.groupby(["QuestionId_DQ", "InterventionId"]):
        question_text = []
        for _, row in group.iterrows():
            row_text = row["Text"].replace("\n", " ").strip()
            question_text.append(f"{row['Label']}: {row_text}")
        subjects = subjects_df[subjects_df["InterventionId"] == group.iloc[0]["InterventionId"]]
        metadata[name] = {
            "question": "\n".join(question_text),
            "subjects": [(row["SubjectName"], row["SubjectLevel"]) for _, row in subjects.iterrows()]
        }
    return metadata

def process_eedi_dialogues(df: pd.DataFrame):
    # Extra individual dialgoues from turn-level table
    data: List[Dialogue] = []
    metadata = load_eedi_metadata()
    for name, group in df.groupby(["QuestionId_DQ", "InterventionId"]):
        turns: List[Turn] = []
        for _, row in group.iterrows():
            role = "tutor" if row["IsTutor"] else "student"
            turn_text = str(row["MessageString"]).replace("\n", " ").strip()
            if not turns or turns[-1]["role"] != role:
                turns.append({"role": role, "content": turn_text})
            else:
                # For multiple turns in a row from same role, append to previous string
                if not turns[-1]["content"].endswith((".", "!", "?")):
                    turns[-1]["content"] += "."
                turns[-1]["content"] += " " + turn_text
        data.append({
            "key": (int(name[0]), int(name[1])),
            "question": metadata[name]["question"],
            "subjects": metadata[name]["subjects"],
            "turns": turns
        })
    return data

def load_eedi_train_val_data():
    # Read and process dialogue data
    df = pd.read_csv("data/Question-Anchored-Tutoring-Dialogues-2k/anchored-dialogues/train.csv")
    data = process_eedi_dialogues(df)

    # Shuffle and split data
    random.Random(221).shuffle(data)
    split_point = int(len(data) * .75)
    return data[:split_point], data[split_point:]

def load_eedi_test_data():
    # Read and process dialogue data
    df = pd.read_csv("data/Question-Anchored-Tutoring-Dialogues-2k/anchored-dialogues/test.csv")
    return process_eedi_dialogues(df)
