import json
import gzip
from ccnet.cc_net import jsonql

class calculate_len(jsonql.Transformer):
    """
    Calculates n_lines and length for the 'raw_content' field in a json.gz file 
    and saves the results to a new file. 
    """

    def __init__(self, field: str = "raw_content", output_file: str = "updated_file.json.gz"):
        super().__init__()
        self.field = field
        self.output_file = output_file
        self.ready = True 
        self.output_handle = gzip.open(self.output_file, 'wt')  # Open the output file for writing

    def do(self, doc: dict) -> dict:
        """
        Processes a single document.

        Args:
            doc (dict): The document as a dictionary.

        Returns:
            dict: The document with updated n_lines and length.
        """
        content = doc.get(self.field)
        if content:
            lines = content.splitlines()
            doc["ccnet_n_lines"] = len(lines)  
            doc["ccnet_length"] = len(content)  
        
        # Write the updated document to the output file
        print(json.dumps(doc), file=self.output_handle)  
        return doc

    def close(self):
        """Closes the output file handle when processing is finished."""
        self.output_handle.close()

if __name__ == "__main__":
    # Replace 'path/to/your/file.json.gz' with the actual path to your file
    file_path = "tr_all_new_2.json.gz"  
    output_path = "ccnet_tr_all_new_2.json.gz"  # Choose the output file name

    calculator = calculate_len(output_file=output_path)
    with gzip.open(file_path, 'rt') as f: 
        for line in f:
            doc = json.loads(line)
            calculator(doc)  # Process and save to the new file