from PyPDF2 import PdfReader
from abc import ABC, abstractmethod

# Creates an abstract class for reading text
class TextExtractor(ABC):
    @abstractmethod
    def extract_all_text(self):
        pass

# Creates a class for reading text from a PDF file
class PdfExtractor(TextExtractor):
    def __init__(self, path):
        self.path = path
        self.reader = PdfReader(path)

    def extract_all_text(self):
        return "\n".join(page.extract_text() for page in self.reader.pages)

    # Iterator to read pages
    def __iter__(self):
        for page in self.reader.pages:
            yield page.extract_text()


if __name__ == '__main__':
    pdf_path = 'articles/attention_is_all_you_need.pdf'

    pdf_reader = PdfExtractor(pdf_path)

    for page in pdf_reader:
        print(page)