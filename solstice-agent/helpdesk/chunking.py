from . import config
import re

def chunk_text(text, size=None):
    """Split a document into fixed-size chunks for embedding."""
    size = size or config.CHUNK_SIZE
    # normalize whitespace so chunks are uniform
    # flat = " ".join(text.split())
    # chunks = []
    # for i in range(0, len(flat), size):
    #     chunks.append(flat[i : i + size])
    # return chunks
    # split on markdown headers or blank lines
    sections = re.split(r'\n(?=#{1,3} |\n)', text)
    chunks, current = [], ""
    for section in sections:
        if len(current) + len(section) > size and current:
            chunks.append(current.strip())
            current = section
        else:
            current += "\n" + section
    if current.strip():
        chunks.append(current.strip())
    return chunks
