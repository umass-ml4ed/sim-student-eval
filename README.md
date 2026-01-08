# Simulated Student Eval

With the increasing use of simulated students in educational AI, we need to be able to evaluate how realisitically these simulations behave. In this repo, we release 1) a set of automated evaluation metrics for simulated student realism grounded in learning sciences research, and 2) a set of benchmark simulated student methods, including prompting, fine-tuning, and reinforcement learning. This code is associated with the paper <a href="https://arxiv.org/abs/2601.04025">Simulated Students in Tutoring Dialogues: Substance or Illusion?</a>.

If you find this code useful, please cite us!
```
@misc{scarlatos2026simulatedstudentstutoringdialogues,
      title={Simulated Students in Tutoring Dialogues: Substance or Illusion?}, 
      author={Alexander Scarlatos and Jaewook Lee and Simon Woodhead and Andrew Lan},
      year={2026},
      eprint={2601.04025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04025}, 
}
```

## Setup

### Python Environment

We used Python 3.12.3 in our experiments.

Create venv and install dependencies:
```
python -m venv sse
source sse/bin/activate
pip install -r requirements.txt
```

Install vllm (run on a node with access to a GPU and Cuda installed):
```
uv pip install vllm==0.10.1.1 --torch-backend=auto
```

Set environment variables:
```
export PYTHONPATH=$PYTHONPATH:./dialogue-kt # Allow Python to access dialogue-kt code
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # For enabling deterministic operations
export VLLM_LOGGING_LEVEL=WARN # Reduce vLLM logging

# OpenAI credentials, for correctness/errors eval and/or data annotation
export OPENAI_API_KEY="your key here" # If using OpenAI API
export AZURE_OPENAI_API_KEY="your key here" # If using Azure API
export AZURE_OPENAI_ENDPOINT="your endpoint here"
```

### Dialogue-KT Code

Clone the dialouge-kt repo (outside this repo):
```
git clone https://github.com/umass-ml4ed/dialogue-kt
```

Create a symlink to the dialogue-kt code (run at the top level of this repo):
```
ln -s "path to dialogue-kt repo"/dialogue_kt dialogue_kt
```

Fix external dependencies:
1. In `site-packages/pykt/models/qdkt.py`, remove line 2 (`from turtle import forward`).
2. In `site-packages/pyBKT/models/Model.py`, on line 32, replace `1e8` with `int(1e8)`.



## Eedi Dataset

### License and Terms

We release LLM-generated annotations on the <a href="https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k">Question-Anchored Tutoring Dialogues 2k dataset</a> from Eedi. 

We release our annotations under the same license as the original dataset: <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)</a>. Our terms of use follow those on the original dataset: the data is intended for non-commercial research purposes. See https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k#uses for more details.

### Recreating Annotations

We include an annotated version of the Eedi dataset in `data/annotated/eedi`, so you don't need to run the following; it is just here for documentation/replication purposes.

Clone the Eedi dataset (outside this repo):
```
git clone https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k
```

Create a symlink to the Eedi dataset (run at the top level of this repo):
```
ln -s "path to Question-Anchored-Tutoring-Dialogues-2k repo" data/Question-Anchored-Tutoring-Dialogues-2k
```

Annotate the data using OpenAI:
```
python -m sim_student.annotate --label questions          # Question solutions (for all metrics and student models)
python -m sim_student.annotate --label acts               # Dialogue acts (for Acts metric)
python -m sim_student.annotate --label corr               # Correctness (for Correctness, Errors, and Knowledge Acquistion metrics)
python -m sim_student.annotate --label eedi_kcs           # Turn-level KCs (for Knowledge Acquistion metric)
python -m sim_student.annotate --label ocean_personas     # OCEAN personas (for Knowledge Acquistion metric and OCEAN prompting method)
python -m sim_student.annotate --label freeform_personas  # Oracle summaries/personas (for Oracle and ICL prompting methods)
```

## Training Models for Automated Metrics

Many of our automated metrics rely on fine-tuned models that will make predictions on simulated student turns. Run the following to train these models.

### Dialouge Act Classifier (Acts Metric)
```
python -m sim_student.acts train --model_name acts-8b
```

### LLMKT (Knowledge Acquisition Metric)
```
python -m sim_student.llmkt train --model_name llmkt-8b-ocean
```

### Tutor Model (Inducing Tutor Response Metric)
```
python -m sim_student.sft --model_name eedi-tutor-sft-8b --role tutor
```

### Correctness Classifier (Optional, only used for distribution analysis)
```
python -m sim_student.correctness train --model_name correctness-8b
```

## Student Simulation

The following trains/tests/evaluates the student models implemented in this repo.

### Fine-tuning methods

Train SFT and test/evaluate on validation set:
```
python -m sim_student.sft --model_name eedi-stud-sft-8b
```

Train DPO and test/evaluate on validation set:
```
python -m sim_student.dpo --pt_model_name eedi-stud-sft-8b --model_name eedi-stud-dpo-8b
```

Test and evaluate on test set:
```
python -m sim_student.testing --test_on test --student_model eedi-stud-dpo-8b
```

Standalone evaluation (after testing):
```
python -m sim_student.eval --input_file results/outputs_eedi_test_student-turn_eedi-stud-dpo-8b.csv
```

### Prompting methods

Test and evaluate on test set:
```
python -m sim_student.testing --test_on test --baseline zs-eth                                 # Zero-Shot
python -m sim_student.testing --test_on test --baseline persona-ocean                          # OCEAN persona
python -m sim_student.testing --test_on test --baseline icl                                    # ICL
python -m sim_student.testing --test_on test --baseline reasoning --baseline_model gpt-5-mini  # Reasoning
python -m sim_student.testing --test_on test --baseline persona-ff                             # Oracle
```
