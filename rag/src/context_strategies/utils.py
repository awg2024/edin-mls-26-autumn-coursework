
# /src/context_strategies/utils.py 
# shared utility functions for all context strategies 

import re


def split_sentences(text):
    """Split a document into candidate sentences."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

def make_candidates(documents):
    """
    documents: list of dictionaries containing: record_id, text
    Return one candidate for each unique sentence.
    """
    candidates = []
    seen = set()

    for doc in documents:
        for position, sentence in enumerate(split_sentences(doc["text"])):
            
            # Normalised key for exact deduplication.
            key = " ".join(sentence.lower().split())

            if key in seen:
                continue

            seen.add(key)
            candidates.append({"record_id": doc["record_id"], "sentence_index": position, "text": sentence})

    return candidates


def format_evidence(item):
    """Keep actual corpus IDs for traceability."""
    return (
        f"[Document {item['record_id']}]\n"
        f"{item['text']}")


def fits_budget(item, selected, max_chars):
    """Check evidence size, including source labels."""
    extra = len(format_evidence(item))  #  returns document record id and then text. 

    if selected:
        extra += 2  # The two newline separators.

    current = sum(len(format_evidence(s)) for s in selected) + 2 * max(0, len(selected) - 1)
    return current + extra <= max_chars


def pack_ranked(candidates, indices, max_chars):
    """Keep ranked candidates that fit the budget."""
    selected = []

    for index in indices:
        item = candidates[int(index)]

        if fits_budget(item, selected, max_chars):
            selected.append(item)

    return selected
