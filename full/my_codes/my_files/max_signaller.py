import gzip
import json
import msgspec
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from redpajama.app.src.core.document import Document
from redpajama.app.src.core.quality_signals.lines import register_lines_callables
from redpajama.app.src.core.quality_signals.natural_language import (
    register_natural_language_callables,
)
from redpajama.app.src.core.quality_signals.repetitions import (
    register_repetitions_callables,
)
from redpajama.app.src.core.quality_signals.content import (
    register_content_callables,
)


class TestInputSpec(msgspec.Struct):
    raw_content: str
    url: str
    nlines: str
    source_domain: str
    length: str
    digest: str
    date_download: str


class Signaller:
    def __init__(self, lang: str):
        self.lang = lang
        self.line_signals = register_lines_callables()
        self.natlang_signals = register_natural_language_callables()
        self.repetition_signals = register_repetitions_callables()
        self.content_signals = register_content_callables(
            language=lang,
            bad_urls_dir="blacklists",
            bad_words_dir="badwords",
        )
        self.decoder = msgspec.json.Decoder(type=TestInputSpec)

    def extract_signals(self, file_path: Path) -> List[Dict]:
        results = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:  # Open in text mode
            for line in tqdm(f):
                record = self.decoder.decode(line)
                doc = Document(
                    content=record.raw_content, domain=record.source_domain
                )
                signals = {}
                for signal_fn in (
                    *self.line_signals,
                    *self.natlang_signals,
                    *self.repetition_signals,
                    *self.content_signals,  # Include content signals
                ):
                    signals[signal_fn.field_name] = signal_fn(doc)
                results.append(signals)
                print(results)
        return results

    def save_results(self, results: List[Dict], output_file: Path):
        with gzip.open(output_file, "wt", encoding="utf-8") as f:  # Open in text mode
            for signals in results:
                f.write(json.dumps(signals, ensure_ascii=False) + "\n")


# Example usage
signaller = Signaller(lang="tr")
signals = signaller.extract_signals(file_path=Path("tr_all_new_2.json.gz"))
signaller.save_results(signals, output_file=Path("signals.json.gz"))