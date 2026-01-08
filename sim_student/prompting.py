from typing import Optional, List, Callable, Union
import json
from transformers import PreTrainedTokenizer
from vllm import LLM

from sim_student.model import generate_vllm, tokenize_prompts
from sim_student.openai_api import OpenAIClient
from sim_student.data_utils import Dialogue

END_OF_DIALOGUE = "<end_of_dialogue>"

STUDENT_SYS_PROMPT_ZS_ETH = f"""You will act as a student in a conversation with a teacher in training. You will need to act as much like a student as possible. If possible do not respond with overly long messages. The conversation with the teacher will be about the following math problem. You may or may not know how to solve it already, let the teacher guide you to the correct understanding. You will be tested at the end and scored thus it is best if you collaborate with the teacher as it has more experience in math than you. If you believe you have figured out the problem and don't need any more help, put {END_OF_DIALOGUE} after your response."""

STUDENT_SYS_PROMPT_PERSONA_FF = f"""{STUDENT_SYS_PROMPT_ZS_ETH}\n\nYou will be given a persona that describes how you should act in the dialogue. Follow this persona as closely as possible."""

STUDENT_SYS_PROMPT_PERSONA_OCEAN = f"""{STUDENT_SYS_PROMPT_ZS_ETH}\n\nYou will be given a Big Five persona that describes how you should act in the dialogue. Follow this persona as closely as possible."""

STUDENT_SYS_PROMPT_ICL = f"""{STUDENT_SYS_PROMPT_ZS_ETH}\n\nYou will also be given an example of a previous dialogue. Your responses should be similar to the ones in this example."""

STUDENT_SYS_PROMPT_REASONING = f"""{STUDENT_SYS_PROMPT_ZS_ETH}\n\nYour response will be judged on how well it matches what the actual student said next in the dialogue (unseen). The following criteria will be used to evaluate your response:
- Acts: Does your response make the same dialogue act as the real student response
- Correctness: Does your response have the same correctness as the real student response
- Errors: If your response is an incorrect math answer, does it have the same underlying error as the real student response
- Knowledge: Does your response represent the same mastery of knowledge concepts as the real student response
- Linguistic: Does your response have the same linguistic features as the real student response

These are the available dialogue acts:
- Math Answer: When the tutor asks a math content-related question, the student attempts to answer that question
- Seek Information: The student seeks more information regarding the math problem or topic, for example, by asking a clarifying or conceptual question
- Not Understanding: The student simply indicates that they do not know the answer to a question or do not understand a concept
- Acknowledge: The student simply acknowledges what the tutor said in the previous turn
- Off-Topic: The student utterance is unrelated to the problem or math topic, including greetings, goodbyes, and other casual conversation

These are the available correctness states:
- Correct: The student correctly responds to the mathematical task posed in the previous tutor turn
- Incorrect: The student incorrectly responds to the mathematical task posed in the previous tutor turn or indicates they don't know the answer
- NA: The tutor doesn't pose a task that has a clear correct/incorrect answer OR the student doesn't indicate correctness in their response

Reason about how to respond in order to maximize the evaluation criteria. Your final response should only contain the predicted student utterance."""

BASELINE_TO_SYS_PROMPT = {
    "zs-eth": STUDENT_SYS_PROMPT_ZS_ETH,
    "persona-ff": STUDENT_SYS_PROMPT_PERSONA_FF,
    "persona-ocean": STUDENT_SYS_PROMPT_PERSONA_OCEAN,
    "icl": STUDENT_SYS_PROMPT_ICL,
    "reasoning": STUDENT_SYS_PROMPT_REASONING
}

STUDENT_SYS_PROMPT_SHORT = "You are a student attempting to solve a math problem, seeking help from a tutor."

TUTOR_SYS_PROMPT_SHORT = "You are a tutor guiding a student through a math problem."

def correct_to_str(correct: Optional[bool]):
    return "correct" if correct else "incorrect" if correct is False else "na"

def str_to_correct(correct_str: str):
    correct_str = correct_str.strip().lower()
    if correct_str == "correct":
        return True
    if correct_str == "incorrect":
        return False
    return None

def format_persona(dialogue: Dialogue, persona_type: str):
    if persona_type == "ocean":
        # Annotation could have failed
        if not dialogue["ocean_persona"]:
            return None
        # Get all traits (exclude reasoning from response)
        return {k: v for k, v  in dialogue["ocean_persona"].items() if k != "reasoning"}
    if persona_type == "freeform":
        # Get freeform persona (pure text)
        return dialogue["freeform_persona"]
    raise Exception(f"Invalid persona type: {persona_type}")

def get_formatted_persona(dialogue: Dialogue, persona_type: Optional[str], turn_idx: int = None):
    assert persona_type in (None, "none", "ocean", "freeform")
    if not persona_type or persona_type == "none":
        return ""
    persona = format_persona(dialogue, persona_type)
    if not persona:
        return ""
    return f"Student Persona:\n{persona}\n\n"

LETTERS = ["A", "B", "C", "D"]

def format_question(dialogue: Dialogue, kcs_src: str = None):
    question = dialogue["question"]
    if (dialogue.get("question_annotation") or {}).get("solvable"):
        anno = dialogue["question_annotation"]
        question += f"\nCorrect Answer: {LETTERS[anno['correct_option'] - 1]}\nSolution: {anno['solution']}\n"
        question += "\n".join([f"Answer {LETTERS[i - 1]} Explanation: " + anno[f"option_{i}_explanation"] for i in range(1, 5)])
    if kcs_src == "eedi":
        question += f"\n\nRelevant Knowledge Components:\n"
        question += "\n".join([f"- {kc}" for kc, level in dialogue["subjects"] if level == 3] + ["- Default"])
    return question

def get_local_prompt(dialogue: Dialogue, role: str, tokenizer: PreTrainedTokenizer, kcs_src: str = None, ending_turn: int = None, last_turn_sub: str = None, persona_type: str = None):
    assert role in ("student", "tutor")
    turns = [*dialogue["turns"]] # Copy to not modify original dialogue
    # Add special end of dialogue tag to end of last turn (except when doing full dialogue generation, i.e. done flag is present)
    if "done" not in dialogue:
        turns[-1] = {**turns[-1], "content": turns[-1]["content"] + END_OF_DIALOGUE}
    _ending_turn = ending_turn if ending_turn is not None else len(turns)
    if last_turn_sub: # Sub in alternative text for last turn, copy to not modify original dialogue
        turns[_ending_turn - 1] = {**turns[_ending_turn - 1], "content": last_turn_sub}

    # Create prompt context
    context = ""
    if role == "student":
        context += get_formatted_persona(dialogue, persona_type, turn_idx=_ending_turn - 1)
    question = format_question(dialogue, kcs_src=kcs_src)
    context += f"Question:\n{question}\n\n"
    # Include first turn if other role starts
    alt_role_title = "Tutor" if role == "student" else "Student"
    if not turns or turns[0]["role"] == role:
        starting_turn = 0
        context += f"(No First {alt_role_title} Turn)"
    else:
        starting_turn = 1
        context += f"First {alt_role_title} Turn: {turns[0]['content']}"

    # Add remaining turns and construct prompt
    turns = turns[starting_turn : _ending_turn]
    system_prompt = STUDENT_SYS_PROMPT_SHORT if role == "student" else TUTOR_SYS_PROMPT_SHORT
    return tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
        *[
            {"role": "assistant" if turn["role"] == role else "user", "content": turn["content"]}
            for turn in turns
        ]
    ], tokenize=False, add_generation_prompt=ending_turn is not None)

def get_llm_prompt(dialogue: Dialogue, kcs_src: str = None, standards: dict = None, ending_turn: int = None,
                   last_turn_sub: Union[str, List[str]] = None, persona_type: str = None):
    turns = [*dialogue["turns"]] # Copy to not modify original dialogue
    _ending_turn = ending_turn if ending_turn is not None else len(turns)
    if last_turn_sub: # Sub in alternative text for last turn(s), copy to not modify original dialogue
        if not isinstance(last_turn_sub, list):
            last_turn_sub = [last_turn_sub]
        for idx, sub in enumerate(reversed(last_turn_sub)):
            turns[_ending_turn - (1 + idx)] = {**turns[_ending_turn - (1 + idx)], "content": sub}

    # Build context
    context = get_formatted_persona(dialogue, persona_type, turn_idx=_ending_turn - 1)
    question = format_question(dialogue, kcs_src=kcs_src)
    context += f"Question:\n{question}\n\n"
    prompt = context + "Conversation History:"

    # Add turns
    for idx, turn in enumerate(turns[:_ending_turn]):
        turn_idx = idx + 1
        role_title = "Student" if turn["role"] == "student" else "Tutor"
        prompt += f"\nTurn {turn_idx} ({role_title}): {turn['content']}"
        if standards and turn["role"] == "tutor":
            cur_skills = standards[f"turn {turn_idx}"]["standards"]
            prompt += f"\nSkills for Turn {turn_idx}: {cur_skills}"
    return prompt

PromptingFnSig = Callable[[List[str], str], List[str]]

def get_prompting_function(args: dict) -> PromptingFnSig:
    if args["engine"] == "openai":
        client = OpenAIClient(args.get("use_azure", False))
        reasoning_model = args["annotation_model"].startswith("gpt-5")
        generation_args = {"response_format": {"type": args.get("response_format", "text")}}
        if reasoning_model:
            generation_args["max_completion_tokens"] = 15000
            generation_args["reasoning_effort"] = args.get("reasoning_effort", "medium")
            generation_args["temperature"] = 1.0
        else:
            generation_args["max_completion_tokens"] = 4000
            generation_args["temperature"] = args.get("temperature", 0.0)
        def prompt_model(prompts: List[str], system_prompt: str):
            return client.get_responses(prompts, args["annotation_model"], system_prompt, generation_args, args.get("batch_api", False))
    elif args["engine"] == "vllm":
        llm = LLM(args["annotation_model"])
        def prompt_model(prompts: List[str], system_prompt: str):
            prompts_tokenized = tokenize_prompts(args["annotation_model"], prompts, system_prompt)
            return generate_vllm(llm, prompts_tokenized, None, {"temperature": 0.2, "top_p": 0.95, "max_tokens": 32000})
    else:
        raise Exception(f"Invalid engine: {args['engine']}")
    return prompt_model

def extract_result(annotation: str, anno_type: str):
    assert anno_type in ("json", "text")
    for reasoning_term in ["</think>", "assistantfinal"]: # Terminators for Qwen, gpt-oss
        if reasoning_term in annotation:
            annotation = annotation.split(reasoning_term)[-1].strip() # Remove think tag if present
            break
    if anno_type == "text":
        return annotation.strip()
    try:
        annotation = annotation.replace("\\", "\\\\") # Make sure latex backslashes are escaped properly
        return json.loads(annotation)
    except json.JSONDecodeError:
        return None
