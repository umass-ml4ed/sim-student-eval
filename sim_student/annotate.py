import argparse
from typing import List, Optional
import re
import pandas as pd

from sim_student.data_loading import load_train_val_data, load_test_data, save_annotated_data
from sim_student.data_utils import Dialogue
from sim_student.prompting import get_llm_prompt, format_question, get_prompting_function, PromptingFnSig, extract_result


###### System Prompts ######

ANNO_KCS_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to list the knowledge components (KCs) that can be used to classify the learning objectives at each turn in this dialogue. Please follow these instructions carefully when making your prediction:
- Tutor turns are often phrased as questions or tasks. In these cases, choose KCs that the student will need in order to respond correctly to the tutor's question. If the tutor turn does not pose a question or task, then you do not need to assign KCs to it.
- You will be given a list of KCs to choose from. When choosing them, write them exactly as they appear.
- If the tutor posed a task but none of the given KCs apply, assign "Default".
- Write a short summary of each tutor turn in the dialogue, including the intended learning objectives.
- Along with each summary, list ALL candidate KCs that can be used to describe each tutor turn in the dialogue.
- Your final response should be a JSON object using the template: {"turn n": {"summary": "...", "kcs": ["kc 1 id", "kc 2 id", ...]}, "turn n+2": ...}
- Use the turn index from the conversation history as the key in your result. There should be exactly one entry for each tutor turn in the dialogue."""

ANNO_CORRECTNESS_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to identify when the student responds correctly to the tutor. Please follow these instructions carefully when making your prediction:
- For each student turn, identify the correctness of the student's response to the previous tutor turn.
- Correctness can be true, false, or null. It is true when the student correctly responds to the previous tutor turn. It is false if the student incorrectly responds to the previous tutor turn, or indicates they do not know the answer. It is null in all other cases, such as when the tutor does not ask a question or only asks a conversational question, or if the student response is purely conversational. A turn is conversational when it does not address a mathematical task posed by the tutor.
- Before making each correctness prediction, write a short summary of each student turn in the dialogue. The summary should include the task previously posed by the tutor, and explain why the student's response is correct, incorrect, or conversational.
- Your final prediction should be a JSON object using the template: {"turn n": {"summary": ..., "correct": true/false/null}, "turn n+2": ...}.
- Use the turn index from the conversation history as the key in your result. There should be exactly one entry for each student turn in the dialogue."""

ANNO_PERSONAS_OCEAN_SYSTEM_PROMPT = """You are analyzing a dialogue between a student and a math tutor. Your task is to assess the student's personality based on the OCEAN model, also known as the Big Five Traits. 

**OCEAN Traits Description:**
- **Openness to Experience:** Reflects the student's curiosity, creativity, willingness to try new things, and openness to new ideas and experiences.  
- **Conscientiousness:** Indicates the student's level of organization, diligence, responsibility, and reliability in approaching tasks.  
- **Extraversion:** Represents how outgoing, energetic, and socially confident the student appears.  
- **Agreeableness:** Measures the student's friendliness, cooperativeness, compassion, and willingness to collaborate.  
- **Neuroticism:** Assesses the student's emotional stability, tendency to experience negative emotions such as anxiety, moodiness, or vulnerability to stress.

First provide reasoning about the student's behavior with respect to the OCEAN model. Then, determine if the student's expression of each trait is **high**, **neutral**, or **low**. Base your reasoning only on the dialogue provided. In your final answer, output your results as a JSON object with the following template:
{
  "reasoning": "...",
  "Openness": "low/neutral/high",
  "Conscientiousness": "low/neutral/high",
  "Extraversion": "low/neutral/high",
  "Agreeableness": "low/neutral/high",
  "Neuroticism": "low/neutral/high"
}"""

ANNO_PERSONAS_FREEFORM_SYSTEM_PROMPT = """You are analyzing a dialogue between a student and a math tutor. Your task is to summarize the student's persona based on their interactions in the dialogue. Focus on the following aspects:
- How well the student acquires knowledge during the dialogue.
- The types of mathematical errors the student makes.
- Any notable behavioral patterns, such as frequent question asking, immediately jumping to the answer, distracting from the task at hand, etc.
- The student's personality traits, such as openness, conscientiousness, extraversion, agreeableness, and neuroticism.
- Notable linguistic patterns in the student's responses.

Your response should be a single paragraph summarizing the student's persona."""

ANNO_QUESTIONS_SYSTEM_PROMPT = """You are a math education expert. Your task is to analyze the options of math multiple choice questions. Follow these instructions carefully:
- First attempt to solve the problem. If it is not possible to solve the problem because it is poorly defined, then say the problem is not solvable.
- Then write an explanation for each option. If the option is the correct answer, write the correct solution to reach that answer. If the option is an incorrect answer, explain the error a student might make to reach that answer.
- Give your final response as a JSON object with the following template:
{
  "solution": ...,
  "solvable": true/false,
  "correct_option": 1-4,
  "option_1_explanation": ...,
  "option_2_explanation": ...,
  "option_3_explanation": ...,
  "option_4_explanation": ...
}"""

ANNO_ACTS_SYSTEM_PROMPT = """You are a math education expert. Your job is to label the **dialogue acts** for student turns in a given dialogue.

These are the available dialogue act labels:
- Math Answer: When the tutor asks a math content-related question, the student attempts to answer that question
- Seek Information: The student seeks more information regarding the math problem or topic, for example, by asking a clarifying or conceptual question
- Not Understanding: The student simply indicates that they do not know the answer to a question or do not understand a concept
- Acknowledge: The student simply acknowledges what the tutor said in the previous turn
- Off-Topic: The student utterance is unrelated to the problem or math topic, including greetings, goodbyes, and other casual converstation

For each **student turn** in the dialogue, choose the dialogue act that best describes the turn. Pick exactly one act for each turn from the list above, and write the dialogue act name exactly as it appears. Before writing the acts for a turn, provide reasoning about what the best act should be.

Please provide your answer as a JSON object with the following format:
{
    "turn n": {
        "reasoning": "...",
        "act": "..."
    },
    "turn n+2": {
        "reasoning": "...",
        "act": "..."
    },
    ...
}"""


###### Helper Functions ######

def get_turn_idx_notice(dialogue: Dialogue, role: Optional[str]):
    return "\n\nImportant: Please ensure your response has an entry for each of the following turns: " + ', '.join([str(idx + 1) for idx, turn in enumerate(dialogue["turns"]) if not role or turn["role"] == role]) + "."

def process_result(annotation: str, anno_type: str):
    assert anno_type in ("json_turn_keys", "json", "text")
    result = extract_result(annotation, "text" if anno_type == "text" else "json")
    if anno_type in ("json", "text"):
        return result
    if anno_type == "json_turn_keys" and result is not None:
        anno_json_proc = {}
        for k, v in result.items():
            if re.match(r"\d+", k): # Prepend "turn" if model only uses integer as key
                k = "turn " + k
            anno_json_proc[k] = v
        return anno_json_proc
    return None

def process_correctness(dialogue: Dialogue, correctness: dict):
    if not correctness:
        return None
    gt_idxs = {idx for idx, turn in enumerate(dialogue["turns"]) if turn["role"] == "student"}
    anno_idxs = {int(k.split()[1]) - 1 for k in correctness.keys() if k.startswith("turn ")}
    if gt_idxs != anno_idxs:
        return {"error": f"GT student turn indices {gt_idxs} do not match annotation indices {anno_idxs}."}
    for turn_key, turn in correctness.items():
        if not isinstance(turn, dict) or "correct" not in turn:
            return {"error": f"Annotation for turn {turn_key} does not contain 'correct' key."}
        if isinstance(turn["correct"], str):
            if turn["correct"].lower() == "true":
                turn["correct"] = True
            elif turn["correct"].lower() == "false":
                turn["correct"] = False
            elif turn["correct"].lower() == "null":
                turn["correct"] = None
            else:
                return {"error": f"Invalid correctness value: {turn['correct']}. Must be true, false, or null."}
        if not (isinstance(turn["correct"], bool) or turn["correct"] is None):
            return {"error": f"Invalid correctness value: {turn['correct']}. Must be true, false, or null."}
    return correctness

def fill_over_idxs(idx_lists: List[List[int]], prompts: List[str], results: List[str], annotations: list, data_size: int):
    prompts_exp = [None] * data_size
    results_exp = [None] * data_size
    annotations_exp = [None] * data_size
    for outer_idx, indices in enumerate(idx_lists):
        for idx in indices:
            prompts_exp[idx] = prompts[outer_idx]
            results_exp[idx] = results[outer_idx]
            annotations_exp[idx] = annotations[outer_idx]
    return prompts_exp, results_exp, annotations_exp


###### Annotation Functions ######

def annotate_corr(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) + get_turn_idx_notice(row, "student") for _, row in data.iterrows()]
    results = prompting_fn(prompts, ANNO_CORRECTNESS_SYSTEM_PROMPT)
    correctness = [
        process_correctness(dialogue, process_result(result, "json_turn_keys"))
        for (_, dialogue), result in zip(data.iterrows(), results)
    ]
    data["correctness_prompt"] = prompts
    data["correctness_annotation_raw"] = results
    data["correctness"] = correctness # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in data['correctness'] if a is not None and "error" not in a])} / {len(data)}")
    return data

def annotate_ocean_personas(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) for _, row in data.iterrows()]
    results = prompting_fn(prompts, ANNO_PERSONAS_OCEAN_SYSTEM_PROMPT)
    personas = [process_result(result, "json") for result in results]
    data["ocean_persona_prompt"] = prompts
    data["ocean_persona_annotation_raw"] = results
    data["ocean_persona"] = personas # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([p for p in personas if p is not None])} / {len(data)}")
    return data

def annotate_freeform_personas(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) for _, row in data.iterrows()]
    results = prompting_fn(prompts, ANNO_PERSONAS_FREEFORM_SYSTEM_PROMPT)
    personas = [process_result(result, "text") for result in results]
    data["freeform_persona_prompt"] = prompts
    data["freeform_persona_annotation_raw"] = results
    data["freeform_persona"] = personas
    print(f"Succeeded: {len([p for p in personas if p is not None])} / {len(data)}")
    return data

def annotate_questions(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    # Get unique question texts
    question_to_indices = {}
    for idx, row in data.iterrows():
        question_to_indices.setdefault(row["question"], []).append(idx)
    idx_lists = list(question_to_indices.values())

    # Annotate questions
    prompts = [question for question in question_to_indices.keys()]
    results = prompting_fn(prompts, ANNO_QUESTIONS_SYSTEM_PROMPT)
    annotations = [process_result(result, "json") for result in results]

    # Clean annotations - occasionally have correct_option=null even when solvable=true
    for annotation in annotations:
        if annotation and annotation["correct_option"] is None:
            annotation["solvable"] = False

    # Expand over dialogues and save
    prompts_exp, results_exp, annotations_exp = fill_over_idxs(idx_lists, prompts, results, annotations, len(data))
    data["question_prompt"] = prompts_exp
    data["question_annotation_raw"] = results_exp
    data["question_annotation"] = annotations_exp # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in annotations_exp if a is not None])} / {len(data)}")
    return data

def annotate_eedi_kcs(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [
        get_llm_prompt(dialogue, kcs_src="eedi") + get_turn_idx_notice(dialogue, "tutor")
        for _, dialogue in data.iterrows()
    ]
    results = prompting_fn(prompts, ANNO_KCS_SYSTEM_PROMPT)
    kcs = [process_result(result, "json_turn_keys") for result in results]
    data["eedi_kcs_prompt"] = prompts
    data["eedi_kcs_annotation_raw"] = results
    data["eedi_kcs"] = kcs # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([kc for kc in kcs if kc is not None])} / {len(data)}")
    return data

def annotate_acts(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [
        get_llm_prompt(dialogue) + get_turn_idx_notice(dialogue, "student")
        for _, dialogue in data.iterrows()
    ]
    results = prompting_fn(prompts, ANNO_ACTS_SYSTEM_PROMPT)
    acts = [process_result(result, "json_turn_keys") for result in results]
    data["acts_prompt"] = prompts
    data["acts_annotation_raw"] = results
    data["acts"] = acts # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in acts if a])} / {len(data)}")
    return data


###### Main ######

LABEL_TO_ANNO_FN = {
    "corr": annotate_corr,
    "ocean_personas": annotate_ocean_personas,
    "freeform_personas": annotate_freeform_personas,
    "questions": annotate_questions,
    "eedi_kcs": annotate_eedi_kcs,
    "acts": annotate_acts,
}

LABELS_WITH_TEXT_OUTPUT = {"freeform_personas"}

def annotate_split(data_list: List[Dialogue], prompting_fn: PromptingFnSig, args: dict):
    data = pd.DataFrame(data_list)
    if args["truncate"]:
        data = data[:args["truncate"]]
    return LABEL_TO_ANNO_FN[args["label"]](data, prompting_fn, args)

def annotate(args: dict):
    # Load data
    annotation_model = args["annotation_model"].replace("/", "-")
    train_data, val_data = load_train_val_data(args["dataset"], annotation_model=annotation_model, drop_unsolvable=False)
    test_data = load_test_data(args["dataset"], annotation_model=annotation_model, drop_unsolvable=False)

    # Set up LLM for prompting
    response_format = "text" if args["label"] in LABELS_WITH_TEXT_OUTPUT else "json_object"
    prompting_fn = get_prompting_function({**args, "response_format": response_format})

    # Annotate all data splits
    print("Annotating train split...")
    train_data = annotate_split(train_data, prompting_fn, args)
    print("Annotating val split...")
    val_data = annotate_split(val_data, prompting_fn, args)
    print("Annotating test split...")
    test_data = annotate_split(test_data, prompting_fn, args)
    suffix = f"_{annotation_model}"
    if args["truncate"]:
        suffix += f"_truncate_{args['truncate']}"
    save_annotated_data(train_data, val_data, test_data, args["dataset"], suffix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["corr", "ocean_personas", "freeform_personas", "questions", "eedi_kcs", "acts"])
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--truncate", type=int)
    parser.add_argument("--engine", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--annotation_model", default="gpt-4.1")
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--batch_api", action="store_true")
    args = parser.parse_args().__dict__

    annotate(args)

if __name__ == "__main__":
    main()
