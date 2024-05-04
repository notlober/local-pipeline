import json
from pathlib import Path
from ccnet.cc_net import jsonql

class SmallRemover:
    """
    Removes documents with a 'raw_content' field shorter than the specified minimum length from a json.gz file.
    """

    def __init__(self, min_len: int = 300):
        self.min_len = min_len
        self.removed = 0

    def process_file(self, input_file: Path, output_file: Path):
        """
        Processes the input file, removes small documents, and writes the remaining documents to the output file.

        Args:
            input_file: Path to the input JSON.gz file.
            output_file: Path to the output JSON.gz file.
        """
        with jsonql.open_write(output_file) as outfile:
            for doc in jsonql.read_jsons(input_file):
                if self.do(doc):
                    print(json.dumps(doc, ensure_ascii=False), file=outfile)
        print(f"Removed {self.removed} small documents.")

    def do(self, doc: dict) -> bool:
        """
        Checks the length of the 'raw_content' field and removes the document if it's too short.

        Args:
            doc: The document to process.

        Returns:
            True if the document meets the minimum length requirement, otherwise False.
        """
        content = doc.get("raw_content")
        if not content or len(content) < self.min_len:
            self.removed += 1
            return False
        return True

# Example usage:
if __name__ == "__main__":
    input_file = Path("tr_all_new_2.json.gz")  # Replace with the actual path to your input file
    output_file = Path("output.json.gz")  # Replace with the desired path for the output file
    min_len = 500  # Set the minimum length threshold
    remover = SmallRemover(min_len)
    remover.process_file(input_file, output_file)