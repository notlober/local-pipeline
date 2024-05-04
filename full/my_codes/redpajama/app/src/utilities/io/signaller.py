import argparse
from pathlib import Path
from typing import List

from redpajama.app.src.core.quality_signals.lines import (
    register_lines_callables,
)
from redpajama.app.src.core.quality_signals.natural_language import (
    register_natural_language_callables,
)
from redpajama.app.src.core.quality_signals.repetitions import (
    register_repetitions_callables,
)
from redpajama.app.src.core.document import Document
from redpajama.app.src.utilities.io import Reader


class Signaller:
    def __init__(self, file: Path):
        self.file = file

    def get_signals(self):
        reader = Reader(schema=[("raw_content", str)])

        # initialize signal functions
        quality_signals = []
        quality_signals += register_natural_language_callables()
        quality_signals += register_lines_callables()
        quality_signals += register_repetitions_callables()

        for record in reader.read(uri="file://" + str(self.file)):
            # initialize document
            document = Document(record.raw_content, domain=None)

            # compute signals
            signals = {}
            for func in quality_signals:
                signals[func.field_name] = func(document)  # noqa

            yield signals

def main(file: str):
    signaller = Signaller(file=Path(file))
    for signals in signaller.get_signals():
        print(signals)


if __name__ == "__main__":
    main(file="tr_all_new_2.json.gz")