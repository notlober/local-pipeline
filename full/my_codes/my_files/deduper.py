from pathlib import Path

from ccnet.cc_net import dedup, jsonql


class Deduplicator(dedup.DuplicatesRemover):
    """
    Deduplicates a JSON.gz file based on the 'raw_content' field using an existing hashes.bin file.
    """

    def __init__(self, hashes_file: Path):
        super().__init__(field="raw_content", hashes_files=[hashes_file])

    def deduplicate_file(self, input_file: Path, output_file: Path):
        """
        Deduplicates the given input file and writes the results to the output file.

        Args:
            input_file: Path to the input JSON.gz file.
            output_file: Path to the output JSON.gz file.
        """
        jsonql.run_pipes(self, file=input_file, output=output_file)


if __name__ == "__main__":
    hashes_file = Path("hashes.bin")  # Replace with the actual path to your hashes.bin
    input_file = Path("tr_all_new_2.json.gz")  # Replace with the actual path to your input file
    output_file = Path("output.json.gz")  # Replace with the desired path for the deduplicated output

    deduplicator = Deduplicator(hashes_file)
    deduplicator.deduplicate_file(input_file, output_file)