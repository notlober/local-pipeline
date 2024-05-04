import json
from pathlib import Path
from typing import Dict, Set

_DEFAULT_LANGS = ("en", "fr", "it", "es", "de")


def load_bad_urls_index(bad_urls_dir: Path) -> Dict[str, int]:
    """
    Loads the URL blacklist from a directory structure with subfolders 
    containing domain files.

    Args:
        bad_urls_dir (Path): Path to the directory containing category subfolders.

    Returns:
        Dict[str, int]: A dictionary mapping domains to category IDs.
    """
    domain_to_category_id = {}
    category_id = 0
    for category_dir in bad_urls_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        with open(category_dir / "domains", "r") as f:
            for domain in f:
                domain = domain.strip()
                domain_to_category_id[domain] = category_id
        category_id += 1
    return domain_to_category_id

def load_bad_words(bad_words_dir: Path, lang: str) -> Set[str]:
    r""" load the LDNOOBW word list for a given language

    Source:
        https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

    Args:
        bad_words_dir (Path): The path to the resources directory where the
            list is stored
        lang (str): The language for which to fetch the word list

    Returns:
        A set of words
    """
    if lang not in _DEFAULT_LANGS:
        return set()

    ldnoobw_fp = bad_words_dir / f"{lang}.txt"

    if not ldnoobw_fp.exists():
        raise FileNotFoundError(f"LDNOOBW word list {ldnoobw_fp} not found!")

    with open(ldnoobw_fp, 'r') as f:
        data = set(ln.strip() for ln in f.readlines())

    return data
