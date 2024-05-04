import argparse
import gzip
import json
import msgspec
from pathlib import Path
from typing import List, Dict

import fasttext
import kenlm
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

from redpajama.app.src.core.document import Document
from redpajama.app.src.core.quality_signals.content import register_content_callables
from redpajama.app.src.core.quality_signals.lines import register_lines_callables
from redpajama.app.src.core.quality_signals.natural_language import (
    register_natural_language_callables,
)
from redpajama.app.src.core.quality_signals.repetitions import (
    register_repetitions_callables,
)

from ccnet.cc_net import jsonql
from ccnet.cc_net import dedup
from ccnet.cc_net import text_normalizer


class TestInputSpec(msgspec.Struct):
    raw_content: str
    url: str
    nlines: str
    source_domain: str
    length: str
    digest: str
    date_download: str


class CustomPipeline:
    """
    A custom pipeline for processing Turkish language text data.

    This pipeline performs the following steps:

    1. Hashing: Calculates hashes for the 'raw_content' field in a JSON.gz file to identify duplicates.
    2. Deduplication: Removes duplicate documents based on the calculated hashes.
    3. Length Calculation: Calculates the number of lines and characters in the 'raw_content' field.
    4. Small Document Removal: Removes documents with a 'raw_content' field shorter than the specified minimum length.
    5. Language Filtering: Filters documents to keep only those in the specified target language.
    6. Tokenization: Tokenizes the 'raw_content' field using a SentencePiece model and normalization.
    7. Perplexity Calculation: Calculates perplexity for each document using a KenLM model.
    8. Perplexity Bucketing: Assigns documents to perplexity buckets (head, middle, tail, or all) based on cutoff values.
    9. Quality Signal Extraction: Extracts various quality signals from the processed documents.

    The pipeline outputs a single JSON.gz file containing the processed documents with extracted quality signals.
    """

    def __init__(
        self,
        sp_model_path: Path,
        lm_model_path: Path,
        lid_model_path: Path,
        cutoff_file: Path,
        target_language: str,
        min_len: int = 300,
        output_file: Path = Path("processed_data.json.gz"),
    ):
        self.sp_model_path = sp_model_path
        self.lm_model_path = lm_model_path
        self.lid_model_path = lid_model_path
        self.cutoff_file = cutoff_file
        self.target_language = target_language
        self.min_len = min_len
        self.output_file = output_file

        # Initialize models and components
        self.sp_model: spm.SentencePieceProcessor = spm.SentencePieceProcessor()
        self.sp_model.Load(str(self.sp_model_path))
        self.lm_model = kenlm.Model(str(self.lm_model_path))
        self.lid_model = fasttext.load_model(str(self.lid_model_path))
        self.bucketer = bucketr(self.cutoff_file, self.target_language)
        self.line_signals = register_lines_callables()
        self.natlang_signals = register_natural_language_callables()
        self.repetition_signals = register_repetitions_callables()
        self.content_signals = register_content_callables(
            language=self.target_language,
            bad_urls_dir="blacklists", 
            bad_words_dir="badwords", 
        )
        self.decoder = msgspec.json.Decoder(type=TestInputSpec)

    def process_file(self, input_file: Path):
        """
        Processes a JSON.gz file through the custom pipeline.

        Args:
            input_file: Path to the input JSON.gz file.
        """
        # 1. hashing and deduplication
        hashes_file = jsonql._tmp(Path("hashes.bin"))
        hasher = Hasher(hashes_file)
        hasher.process_file(input_file)
        deduplicator = Deduplicator(hashes_file)

        # 2. length Calculation, ssmall Document Removal, and Language Filtering
        calculator = calculate_len()
        
        def remove_small_docs(docs):
            remover = SmallRemover(self.min_len)
            for doc in docs:
                if doc is None:  # check for None
                    continue
                if remover.do(doc):
                    yield doc
            print(f"Removed {remover.removed} small documents.")

        def filter_language(docs):
            filterer = filt_lan(self.lid_model_path, self.target_language)
            for doc in docs:
                if doc is None:  # check for none
                    continue
                processed_doc = filterer.process_document(doc)
                if processed_doc:
                    yield processed_doc
        
        # 3. Tokenization and Perplexity Calculation
        tokenizer = Tokenizer(self.sp_model_path)

        def tokenize_docs(docs):
            for doc in docs:
                yield tokenizer.tokenize_doc(doc)

        perplexitier = Perplexitier(self.lm_model_path)

        def calculate_perplexity(docs):
            for doc in docs:
                tokenized_text = doc.get("tokenized")
                if tokenized_text:
                    doc["perplexity"] = perplexitier.model.perplexity(tokenized_text)
                yield doc

        # 4. Perplexity Bucketing and Quality Signal Extraction
        signaller = Signaller(self.target_language)

        def bucket_and_extract_signals(docs):
            for doc in docs:
                doc["bucket"] = self.bucketer.get_bucket(doc)  # Apply bucketing
                document_obj = Document(content=doc["raw_content"], domain=doc["source_domain"])  # Recreate Document object
                signals = {}
                for signal_fn in (
                    *self.line_signals,
                    *self.natlang_signals,
                    *self.repetition_signals,
                    *self.content_signals,
                ):
                    signals[signal_fn.field_name] = signal_fn(document_obj)  # Use the Document object for signal extraction
                doc.update(signals)
                yield doc

        # Create temporary output file
        temp_output_file = jsonql._tmp(Path("temp_output.json.gz"))

        # Build and run the pipeline
        pipeline = [
            jsonql.JsonReader(),
            deduplicator,
            calculator,
            remove_small_docs,
            filter_language,
            tokenize_docs,
            calculate_perplexity,
            bucket_and_extract_signals,  # Integrate bucketing and signal extraction
        ]
        inputs = jsonql.read_jsons(input_file)
        tqdm_inputs = tqdm(inputs, desc="Processing documents") # very hacky, by selahattin baki damar :)

        jsonql.run_pipes(
            *pipeline,
            inputs=tqdm_inputs,  # Read from input file
            output=temp_output_file,
        )

        # Save the final results
        signaller.save_results(jsonql.read_jsons(temp_output_file), self.output_file)

        # Remove temporary files
        hashes_file.unlink()
        temp_output_file.unlink()

# --- Classes from final_repo ---

class Signaller:
    """Extracts quality signals from documents."""

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
        """Extracts quality signals from a JSON.gz file."""
        results = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:  # Open in text mode
            for line in tqdm(f):
                record = self.decoder.decode(line)
                doc = Document(content=record.raw_content, domain=record.source_domain)
                signals = {}
                for signal_fn in (
                    *self.line_signals,
                    *self.natlang_signals,
                    *self.repetition_signals,
                    *self.content_signals,  # Include content signals
                ):
                    signals[signal_fn.field_name] = signal_fn(doc)
                results.append(signals)
        return results

    def save_results(self, results: List[Dict], output_file: Path):
        """Saves extracted quality signals to a JSON.gz file."""
        with gzip.open(output_file, "wt", encoding="utf-8") as f:  # Open in text mode
            for signals in results:
                f.write(json.dumps(signals, ensure_ascii=False) + "\n")

# --- Classes from deduper.py ---

class Hasher:
    """Calculates hashes for the 'raw_content' field in a JSON.gz file."""

    def __init__(self, output: Path):
        self.output = output
        self.hashes = dedup.FlatHashSet()

    def process_file(self, input_file: Path):
        """Processes the input file and calculates hashes."""
        for doc in jsonql.read_jsons(input_file):
            self.do(doc)
        self.close()

    def do(self, doc: dict) -> None:
        """Calculates hashes for a single document."""
        content = doc.get("raw_content")
        if not content:
            return
        doc_hashes = dedup.compute_hashes(content)
        if doc_hashes is None:
            return
        self.hashes.add(doc_hashes)

    def close(self):
        """Saves the calculated hashes to a file."""
        if self.output and self.hashes:
            self.hashes.dump(self.output)
            print(f"Saved {len(self.hashes)} hashes to {self.output}")


class Deduplicator(dedup.DuplicatesRemover):
    """Deduplicates a JSON.gz file based on the 'raw_content' field."""

    def __init__(self, hashes_file: Path):
        super().__init__(field="raw_content", hashes_files=[hashes_file])

    def deduplicate_file(self, input_file: Path, output_file: Path):
        """Deduplicates the given input file and writes the results."""
        jsonql.run_pipes(self, file=input_file, output=output_file)

# --- Classes from calculate_len.py ---

class calculate_len(jsonql.Transformer):
    """Calculates n_lines and length for the 'raw_content' field."""

    def __init__(self, field: str = "raw_content"):
        super().__init__()
        self.field = field
        self.ready = True

    def do(self, doc: dict) -> dict:
        """Processes a single document."""
        content = doc.get(self.field)
        if content:
            lines = content.splitlines()
            doc["ccnet_n_lines"] = len(lines)
            doc["ccnet_length"] = len(content)
        return doc

# --- Classes from smallremover.py ---

class SmallRemover:
    """Removes documents with a 'raw_content' field shorter than the specified minimum length."""

    def __init__(self, min_len: int = 300):
        self.min_len = min_len
        self.removed = 0

    def process_file(self, input_file: Path, output_file: Path):
        """Processes the input file and removes small documents."""
        with jsonql.open_write(output_file) as outfile:
            for doc in jsonql.read_jsons(input_file):
                if self.do(doc):
                    print(json.dumps(doc, ensure_ascii=False), file=outfile)
        print(f"Removed {self.removed} small documents.")

    def do(self, doc: dict) -> bool:
        """Checks the length of the 'raw_content' field and removes the document if it's too short."""
        content = doc.get("raw_content")
        if not content or len(content) < self.min_len:
            self.removed += 1
            return False
        return True

# --- Classes from filt_lan.py ---

class filt_lan:
    """Filters documents to keep only those in the specified target language."""

    def __init__(self, model_path: Path, target_language: str):
        self.model = fasttext.load_model(str(model_path))
        self.target_language = target_language

    def process_file(self, input_file: Path, output_file: Path):
        """Processes the input file and filters by language."""
        with gzip.open(input_file, 'rt') as fin, gzip.open(output_file, 'wt') as fout:
            for line in fin:
                doc = json.loads(line)
                processed_doc = self.process_document(doc)
                if processed_doc:
                    fout.write(json.dumps(processed_doc) + '\n')

    def process_document(self, doc):
        """Processes a single document and filters by language."""
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

# --- Classes from tokeniser.py ---

class Tokenizer:
    """Tokenizes the 'raw_content' field using a SentencePiece model and normalization."""

    def __init__(self, sp_model_path: Path):
        self.sp_model_path = sp_model_path
        self.sp_model: spm.SentencePieceProcessor = spm.SentencePieceProcessor() 
        self.sp_model.Load(str(self.sp_model_path))

    def tokenize_file(self, input_file: Path, output_file: Path):
        """Tokenizes the 'raw_content' field in the input file.""" 
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                tokenized_doc = self.tokenize_doc(doc)
                json_str = json.dumps(tokenized_doc, ensure_ascii=False)
                outfile.write(json_str + '\n')

    def tokenize_doc(self, doc: dict) -> dict:
        """Tokenizes the 'raw_content' field of a single document."""
        text = doc.get("raw_content")
        if not text:
            return doc

        # Normalize the text before tokenization
        normalized_text = text_normalizer.normalize(text) 
        tokenized_text = self.sp_model.EncodeAsPieces(normalized_text)
        doc["tokenized"] = " ".join(tokenized_text)  
        return doc

# --- Classes from perplexitier.py ---

class Perplexitier:
    """Calculates perplexity for the 'tokenized' field using a KenLM model."""

    def __init__(self, model_path: Path):
        self.model = kenlm.Model(str(model_path))

    def calculate_and_save_perplexity(self, input_file: Path, output_file: Path):
        """Calculates perplexity for each document."""
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                tokenized_text = doc.get("tokenized")
                if tokenized_text:
                    doc["perplexity"] = self.model.perplexity(tokenized_text)
                json_str = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_str + '\n')

# --- Classes from bucketr.py ---

class bucketr:
    """Performs perplexity bucketing based on cutoff values for a single language."""

    def __init__(self, cutoff_file: Path, language: str):
        """Loads cutoff values for a specific language from a CSV file."""
        cutoffs = pd.read_csv(cutoff_file, index_col=0)
        self.cutoffs = cutoffs[language]

    def get_bucket(self, doc: dict) -> str:
        """Determines the perplexity bucket for a document."""
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
        """Adds the 'bucket' field to documents and saves the results."""
        with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
            for line in infile:
                doc = json.loads(line)
                doc["bucket"] = self.get_bucket(doc)
                json_str = json.dumps(doc, ensure_ascii=False)
                outfile.write(json_str + '\n')


# --- Main Function ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Turkish Text Processing Pipeline")
    parser.add_argument(
        "--input_file", type=Path, required=True, help="Path to the input JSON.gz file"
    )
    parser.add_argument(
        "--sp_model_path", type=Path, required=True, help="Path to the SentencePiece model file"
    )
    parser.add_argument(
        "--lm_model_path", type=Path, required=True, help="Path to the KenLM model file"
    )
    parser.add_argument(
        "--lid_model_path",
        type=Path,
        required=True,
        help="Path to the fastText language identification model file",
    )
    parser.add_argument(
        "--cutoff_file", type=Path, required=True, help="Path to the cutoff CSV file"
    )
    parser.add_argument(
        "--target_language", type=str, required=True, help="Target language code (e.g., 'tr')"
    )
    parser.add_argument(
        "--min_len", type=int, default=300, help="Minimum length of 'raw_content' to keep"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("processed_data.json.gz"),
        help="Path to the output JSON.gz file",
    )
    args = parser.parse_args()

    pipeline = CustomPipeline(
        args.sp_model_path,
        args.lm_model_path,
        args.lid_model_path,
        args.cutoff_file,
        args.target_language,
        args.min_len,
        args.output_file,
    )
    pipeline.process_file(args.input_file)

# python3 my_codes/main.py --input_file tr_all_new_2.json.gz --sp_model_path my_codes/data/lm_sp/tr.sp.model --lm_model_path my_codes/data/lm_sp/tr.arpa.bin --lid_model_path lid.bin --cutoff_file cutoff_all_tr.csv --target_language tr --min_len 1000 --output_file outputs.json.gz