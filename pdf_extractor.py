from PyPDF2 import PdfReader
from abc import ABC, abstractmethod
import os
from typing import List

# Creates an abstract class for reading text
class TextExtractor(ABC):
    """
    Abstract base class for extracting text from documents.
    """
    @abstractmethod
    def extract_all_text(self) -> List[str]:
        """
        Extract all text from the document as a list of strings.
        """
        pass

# Creates a class for reading text from a PDF file
class PdfExtractor(TextExtractor):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file at {path} does not exist.")
        self.path = path
        self.reader = PdfReader(path)
        self._max_pages = len(self.reader.pages)
    
    def _clean_text(self, text: str) -> str:
        """
        Removes unnecessary newlines and extra spaces from the text.
        """
        return " ".join(text.split())
    
    def __len__(self) -> int:
        """
        Returns the total number of pages in the PDF.
        """
        return self._max_pages
    
    def __getitem__(self, page_number: int) -> str:
        """
        Returns the cleaned text of a specific page.
        """
        if page_number >= len(self):
            raise IndexError("Page index out of range.")
        raw_text = self.reader.pages[page_number].extract_text()
        return self._clean_text(raw_text or "")
    
    def __iter__(self):
        """
        Allows iteration over the pages of the PDF.
        """
        for page_number in range(len(self)):
            yield self[page_number]
    
    def extract_all_text(self) -> List[str]:
        """
        Extracts all text from the PDF as a list of strings, one per page.
        """
        return [self[page_number] for page_number in range(len(self))]

class EmbeddingConverter:
    """
    Converts text documents into embeddings using a given embedding model.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def embed_documents(self, documents: List[str]):
        """
        Converts a list of documents into embeddings.
        """
        return self.embedding_model.embed_documents(documents)

def main():
    pdf_path = 'articles/attention_is_all_you_need.pdf'

    try:
        pdf_reader = PdfExtractor(pdf_path)
        for page in pdf_reader:
            print(page)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
