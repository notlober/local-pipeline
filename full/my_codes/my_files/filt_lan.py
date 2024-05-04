import fasttext
import gzip
import json
from pathlib import Path

class filt_lan:
    def __init__(self, model_path: Path, target_language: str):
        self.model = fasttext.load_model(str(model_path))
        self.target_language = target_language

    def process_file(self, input_file: Path, output_file: Path):
        with gzip.open(input_file, 'rt') as fin, gzip.open(output_file, 'wt') as fout:
            for line in fin:
                doc = json.loads(line)
                processed_doc = self.process_document(doc)
                if processed_doc:
                    fout.write(json.dumps(processed_doc) + '\n')

    def process_document(self, doc):
        text = doc.get("raw_content")
        if not text:
            return None

        labels, scores = self.model.predict(text.replace("\n", ""), k=1)
        language = labels[0].replace("__label__", "")
        score = scores[0]

        if language == self.target_language:
            doc["language"] = language
            doc["language_score"] = score
            return doc
        else:
            return None

# Example usage:
if __name__ == "__main__":
    model_path = Path("lid.bin")  # Path to your lid.bin model
    input_file = Path("tr_all_new_2.json.gz")  # Path to your input file
    output_file = Path("filter.json.gz")  # Path to your output file
    target_language = "tr"  # Target language (Turkish in this case)

    processor = filt_lan(model_path, target_language)
    processor.process_file(input_file, output_file)