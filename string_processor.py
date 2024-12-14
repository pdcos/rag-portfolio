import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class StringProcessor:
    """
    A class to process strings by splitting them into chunks and generating embeddings.
    
    Attributes:
        embedding_model (SentenceTransformer): Pre-trained model to generate embeddings.
        chunk_size (int): Maximum number of words in a chunk.
        chunk_overlap (int): Number of overlapping words between consecutive chunks.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_str_into_chunks(self, text: str) -> List[str]:
        """
        Splits a string into chunks of words with overlap.

        Args:
            text (str): Text to be split.

        Returns:
            List[str]: List of text chunks.
        """
        words_list = text.split()
        chunks = []

        for i in range(0, len(words_list), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words_list[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Converts a list of chunks into embeddings using the embedding model.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            np.ndarray: Array of embeddings.
        """
        if not chunks:
            raise ValueError("The chunks list is empty. Provide valid chunks.")

        return np.array(self.embedding_model.encode(chunks))


def main() -> None:
    """
    Main function to demonstrate the functionality of StringProcessor.
    """
    text = "This is a sample text to be split into chunks."
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Instantiate StringProcessor with appropriate parameters
    processor = StringProcessor(
        embedding_model=embedding_model,
        chunk_size=5,
        chunk_overlap=1
    )

    # Process text: split into chunks and generate embeddings
    chunks = processor.split_str_into_chunks(text)
    embeds = processor.embed_chunks(chunks)

    # Output results
    print("Chunks:", chunks)
    print("Embeddings shape:", embeds.shape)


if __name__ == "__main__":
    main()
