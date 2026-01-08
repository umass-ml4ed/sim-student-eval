import numpy as np
from collections import Counter

from sim_student.data_loading import load_train_val_data, load_test_data

PERSONA_DIMS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def num_turns(data):
    return sum(len(dialogue["turns"]) for dialogue in data)

def main():
    # Load data
    train_data, val_data = load_train_val_data("eedi")
    test_data = load_test_data("eedi")
    all_data = train_data + val_data + test_data

    # Stats to collect
    num_stud_turns = 0
    num_tutor_turns = 0
    tutor_starts = 0
    dia_lens = []
    stud_turn_lens = []
    tutor_turn_length = 0
    num_correct = 0
    num_incorrect = 0
    num_na = 0
    num_eedi_kcs_per_dia = []
    acts_counter = Counter()
    eedi_kcs_counter = Counter()
    cc_kcs_counter = Counter()
    persona_counter = Counter()
    persona_dim_to_counter = {dim: Counter() for dim in PERSONA_DIMS}

    # Collect stats from data
    for dialogue in all_data:
        dia_lens.append(len(dialogue["turns"]))
        persona = dialogue["ocean_persona"]
        if persona:
            persona_counter[str({k: v for k, v in persona.items() if k != "reasoning"})] += 1
            for dim, counter in persona_dim_to_counter.items():
                counter[persona[dim]] += 1
        if dialogue["turns"][0]["role"] == "tutor":
            tutor_starts += 1
        num_eedi_kcs_per_dia.append(len(dialogue["subjects"]))
        for turn_idx, turn in enumerate(dialogue["turns"]):
            turn_len = len(turn["content"].split(" "))
            if turn["role"] == "student":
                num_stud_turns += 1
                stud_turn_lens.append(turn_len)
                if dialogue["correctness"] and "error" not in dialogue["correctness"]:
                    corr = dialogue["correctness"][f"turn {turn_idx + 1}"]["correct"]
                    if corr is True:
                        num_correct += 1
                    elif corr is False:
                        num_incorrect += 1
                    else:
                        num_na += 1
                if dialogue["acts"]:
                    act = dialogue["acts"][f"turn {turn_idx + 1}"]["act"]
                    acts_counter[act] += 1
            else:
                num_tutor_turns += 1
                tutor_turn_length += turn_len
                kcs = dialogue["eedi_kcs"][f"turn {turn_idx + 1}"]["kcs"]
                for kc in kcs:
                    eedi_kcs_counter[kc] += 1
    min_dia_len = min(dia_lens)
    max_dia_len = max(dia_lens)
    dia_len_hist, _dia_len_bin_edges = np.histogram(dia_lens, max_dia_len - min_dia_len)
    dia_len_cdf = np.cumsum(dia_len_hist)

    # Report stats
    print(f"Num dialogues: {len(all_data)}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Num turns - Train: {num_turns(train_data)}, Val: {num_turns(val_data)}, Test: {num_turns(test_data)}, Student Turn Total: {num_stud_turns}")
    print(f"Dialogues where tutor starts: {tutor_starts}")
    print(f"Average dialogue length (turns): {np.mean(dia_lens):.2f} (min={min_dia_len}, max={max_dia_len})")
    print(f"Dialogue length CDF: {dia_len_cdf}")
    print(f"Average student turn length (words): {np.mean(stud_turn_lens):.2f}")
    print(f"Average tutor turn length (words): {tutor_turn_length / num_tutor_turns:.2f}")
    print(f"Student turn length counts: {sorted(Counter(stud_turn_lens).items())}")
    print(f"Student turns - Correct: {num_correct}, Incorrect: {num_incorrect}, N/A: {num_na}")
    print("Acts distribution:")
    for act, count in acts_counter.most_common():
        print(f"  {act}: {count}")
    print(f"Avg num Eedi KCs per dialogue: {np.mean(num_eedi_kcs_per_dia):.4f}")
    print(f"Eedi KCs distribution (top 20 out of {len(eedi_kcs_counter)}):")
    for kc, count in eedi_kcs_counter.most_common(20):
        print(f"  {kc}: {count}")
    print(f"CC KCs distribution (top 20 out of {len(cc_kcs_counter)}):")
    for kc, count in cc_kcs_counter.most_common(20):
        print(f"  {kc}: {count}")
    print(f"Persona distribution (top 10 out of {len(persona_counter)}):")
    for persona, count in persona_counter.most_common(10):
        print(f"  {persona}: {count}")
    print("Persona dimension distributions:")
    for dim, counter in persona_dim_to_counter.items():
        print(f"  {dim}:")
        for score, count in sorted(counter.items()):
            print(f"    {score}: {count}")

if __name__ == "__main__":
    main()
