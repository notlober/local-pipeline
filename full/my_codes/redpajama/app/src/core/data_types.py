from dataclasses import dataclass
from msgspec import Struct

from typing import List, Tuple, Optional, Dict
from typing_extensions import TypeAlias

ScoreType: TypeAlias = Tuple[int, int, Optional[float]]
SignalType: TypeAlias = List[ScoreType]


@dataclass
class TextSlice:
    text: str
    start: int
    end: int

    def __len__(self):
        return len(self.text)
    
class InputSpec(Struct):
    raw_content: str
    url: str
    nlines: str
    original_nlines: str
    source_domain: str
    length: str
    original_length: str
    language: str
    language_score: float
    perplexity: float
    bucket: str
    digest: str
    cc_segment: str
    date_download: str


class OutputSpec(Struct):
    id: str
    id_int: int
    metadata: Dict[str, str]
    quality_signals: Dict[str, List[Tuple[int, int, Optional[float]]]]



