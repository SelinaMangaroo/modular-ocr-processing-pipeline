from typing import List, Optional
from pydantic import BaseModel

class CorrectedText(BaseModel):
    corrected_text: str

class EntitiesOutput(BaseModel):
    People: List[str]
    Productions: List[str]
    Companies: List[str]
    Theaters: List[str]
    Dates: List[str]

class CombinedOutput(BaseModel):
    page_number: Optional[int]
    letters: List[str]
