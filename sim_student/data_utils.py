from typing import TypedDict, List, Optional
from torch.utils.data import Dataset

class Turn(TypedDict):
    role: str
    content: str
    candidates: Optional[List[str]]
    cand_scores: Optional[List[float]]

class Dialogue(TypedDict):
    # Ground truth attributes
    key: str
    question: str
    turns: List[Turn]
    subjects: List[tuple]
    # Annotations
    question_annotation: Optional[dict]
    correctness: Optional[dict]
    eedi_kcs: Optional[dict]
    acts: Optional[dict]
    ocean_persona: Optional[dict]
    freeform_persona: Optional[str]
    # Attributes for full dialogue generation
    gt_turns: Optional[List[Turn]] # Ground truth turns for reference
    done: bool # Whether the dialogue is complete

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)
