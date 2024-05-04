import gzip
import json
from pathlib import Path
import pandas as pd

class bucketr:
    """Performs perplexity bucketing based on cutoff values for a single language."""

    def __init__(self, cutoff_file: Path, language: str):
        """
        Loads cutoff values for a specific language from a CSV file.

        Args:
            cutoff_file: Path to the CSV file containing cutoff values.
            language: The language code (e.g., "en"). 
        """
        cutoffs = pd.read_csv(cutoff_file, index_col=0)
        self.cutoffs = cutoffs[language]

    def get_bucket(self, doc: dict) -> str:
        """
        Determines the perplexity bucket for a document.

        Args:
            doc: The document as a dictionary.

        Returns:
            The perplexity bucket (head, middle, tail, or all).
        """
        perplexity = doc.get("perplexity", -1)
        if perplexity < 0:
            return "all"

        cutoff_head, cutoff_tail = self.cutoffs[1], self.cutoffs[2]
        if perplexity < cutoff_head:
            return "head"
        if perplexity < cutoff_tail:
            return "middle"
        return "tail"

    def bucket_and_save(self, input_file: Path, output_file: Path):
        """
        Adds the 'bucket' field to documents and saves the results.

        Args:
            input_file: Path to the input JSON.gz file.
            output_file: Path to the output JSON.gz file.
        """
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                doc["bucket"] = self.get_bucket(doc)
                json_str = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_str + '\n')

# Example usage:
if __name__ == "__main__":
    cutoff_file = Path("cutoff_all_tr.csv")  # Replace with actual path
    language = "tr"  # Replace with your language code
    input_file = Path("perplexity_output.json.gz")  # Replace with actual path
    output_file = Path("bucketed_output.json.gz")  # Replace with desired path
    bucketer = bucketr(cutoff_file, language)
    bucketer.bucket_and_save(input_file, output_file)