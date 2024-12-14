import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from text_extractors import PdfExtractor
from tqdm import tqdm

class VectorDatabase():
    def __init__(self,
                 db_path: str,
                 embedding_model: HuggingFaceEmbeddings,
                 extractor_engine: PdfExtractor,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.extractor_engine = extractor_engine
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.initialize_db_files()

    def initialize_db_files(self):
        """"
        Create the necessary files for the database if they do not exist.
        """
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # create one .npz file for storing the embeddings and one .txt file for storing phrases with their corresponding indices
        self.embeddings_file_path = os.path.join(self.db_path, 'embeddings.npz')
        self.phrases_file_path = os.path.join(self.db_path, 'phrases.txt')

        if not os.path.exists(self.embeddings_file_path):
            np.save(self.embeddings_file_path, np.array([]))
        
        if not os.path.exists(self.phrases_file_path):
            with open(self.phrases_file_path, 'w') as f:
                f.write('')

    def append_db_files(self, embeddings, phrases):
        """
        Append the embeddings and phrases to the database files without loading the existing data.
        """
        np.save(self.embeddings_file_path, embeddings)
        with open(self.phrases_file_path, 'a') as f:
            for phrase in phrases:
                f.write(phrase + '\n')
        return


    def load_db_files(self):
        """
        Load the embeddings and phrases from the database files.
        """

        for page in tqdm(self.extractor_engine):
            embeddings = np.array(self.embedding_model.embed_documents(page))
            self.phrases = page

            # Separates the embeddings and phrases into chunks with overlap and saves them to the database
            for i in range(0, len(embeddings), self.chunk_size - self.chunk_overlap):
                chunk_embeddings = embeddings[i:i+self.chunk_size]
                chunk_phrases = self.phrases[i:i+self.chunk_size]

                self.append_db_files(chunk_embeddings, chunk_phrases)

            

if __name__ == '__main__':
    pdf_path = 'articles/attention_is_all_you_need.pdf'
    pdf_extractor = PdfExtractor(pdf_path)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = VectorDatabase('db', embeddings, pdf_extractor)
    db.load_db_files()
    print(db.embeddings)
    print(db.phrases)