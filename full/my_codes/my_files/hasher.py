from pathlib import Path

from ccnet.cc_net import dedup
from ccnet.cc_net import jsonql


class Hasher:
    """
    Calculate hashes for the "raw_content" field in a json.gz file.
    """

    def __init__(self, output: Path):
        self.output = output
        self.hashes = dedup.FlatHashSet()

    def process_file(self, input_file: Path):
        """
        Processes the input file and calculates hashes for the 'raw_content' field.

        Args:
            input_file: Path to the input JSON.gz file.
        """
        for doc in jsonql.read_jsons(input_file):
            self.do(doc)
        self.close()

    def do(self, doc: dict) -> None:
        content = doc.get("raw_content")
        if not content:
            return
        doc_hashes = dedup.compute_hashes(content)
        if doc_hashes is None:
            return
        self.hashes.add(doc_hashes)

    def close(self):
        if self.output and self.hashes:
            self.hashes.dump(self.output)
            print(f"Saved {len(self.hashes)} hashes to {self.output}")


# Example usage:
if __name__ == "__main__":
    input_file = Path("tr_all_new_2.json.gz")  # Replace with the actual path to your input file
    output_file = Path("hashes.bin")  # Replace with the desired path for the hashes file

    hasher = Hasher(output_file)
    hasher.process_file(input_file)