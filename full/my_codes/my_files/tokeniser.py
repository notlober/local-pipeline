from pathlib import Path
import sentencepiece as spm  # Make sure to have the 'sentencepiece' library installed
import gzip
import json
from ccnet.cc_net import text_normalizer


class Tokenizer:
    """Tokenizes the 'raw_content' field using a SentencePiece model and normalization."""

    def __init__(self, sp_model_path: Path):
        self.sp_model_path = sp_model_path
        self.sp_model: spm.SentencePieceProcessor = spm.SentencePieceProcessor()  # Directly initialize 
        self.sp_model.Load(str(self.sp_model_path))

    def tokenize_file(self, input_file: Path, output_file: Path):
        """
        Tokenizes the 'raw_content' field in the input file and saves the result to the output file.
        Args:
            input_file: Path to the input JSON.gz file.
            output_file: Path to the output JSON.gz file.
        """
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                tokenized_doc = self.tokenize_doc(doc)
                json_str = json.dumps(tokenized_doc, ensure_ascii=False)
                outfile.write(json_str + '\n')

    def tokenize_doc(self, doc: dict) -> dict:
        """
        Tokenizes the 'raw_content' field of a single document and applies normalization.
        Args:
            doc: The document as a dictionary.
        Returns:
            The document with the tokenized 'raw_content' field.
        """
        text = doc.get("raw_content")
        if not text:
            return doc

        # Normalize the text before tokenization (using text_normalizer from cc_net)
        normalized_text = text_normalizer.normalize(text) 
        tokenized_text = self.sp_model.EncodeAsPieces(normalized_text)
        doc["tokenized"] = " ".join(tokenized_text)  # Store the tokenized text in a new field
        return doc


# Example usage:
if __name__ == "__main__":
    sp_model_file = Path("my_codes/data/lm_sp/tr.sp.model")  # Replace with the actual path to your model
    input_file = Path("tr_all_new_2.json.gz")  # Replace with the actual path to your input file
    output_file = Path("tokenized_data.json.gz")  # Replace with the desired path for the output file
    tokenizer = Tokenizer(sp_model_file)
    tokenizer.tokenize_file(input_file, output_file)