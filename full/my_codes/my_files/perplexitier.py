import gzip
import json
from pathlib import Path

import kenlm  # Make sure you have 'kenlm' installed


class Perplexitier:
    """Calculates perplexity for the 'tokenized' field using a KenLM model."""

    def __init__(self, model_path: Path):
        self.model = kenlm.Model(str(model_path))

    def calculate_and_save_perplexity(self, input_file: Path, output_file: Path):
        """
        Calculates perplexity for each document and saves the results to a JSON.gz file.

        Args:
            input_file: Path to the input JSON.gz file.
            output_file: Path to the output JSON.gz file.
        """
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                tokenized_text = doc.get("tokenized")
                if tokenized_text:
                    doc["perplexity"] = self.model.perplexity(tokenized_text)
                json_str = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_str + '\n')

# Example usage:
if __name__ == "__main__":
    model_file = Path("my_codes/data/lm_sp/tr.arpa.bin")  # Replace with the actual path
    input_file = Path("tokenized_data.json.gz")  # Replace with the path 
    output_file = Path("perplexity_output.json.gz")  # Replace with desired path

    perplexitier = Perplexitier(model_file)
    perplexitier.calculate_and_save_perplexity(input_file, output_file)