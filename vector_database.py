import os
import numpy as np
from sentence_transformers import SentenceTransformer
from text_extractors import PdfExtractor
from string_processor import StringProcessor
from tqdm import tqdm

class VectorDatabase():
    def __init__(self,
                 db_path: str,
                 extractor_engine: PdfExtractor,
                 string_processor: StringProcessor,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        
        self.db_path = db_path
        self.extractor_engine = extractor_engine
        self.string_processor = string_processor
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.initialize_db_files()

    def initialize_db_files(self):
        """
        Create the necessary files for the database if they do not exist.
        """
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        self.embeddings_file_path = os.path.join(self.db_path, 'embeddings.npz.npy')
        self.phrases_file_path = os.path.join(self.db_path, 'phrases.txt')

        if not os.path.exists(self.embeddings_file_path):
            np.save(self.embeddings_file_path, np.empty((0, 384), dtype=np.float32))
        
        if not os.path.exists(self.phrases_file_path):
            with open(self.phrases_file_path, 'w') as f:
                f.write('')

    def append_db_files(self, embeddings, phrases):
        """
        Append the embeddings and phrases to the database files without overwriting the existing data.
        """
        # Load existing embeddings and append the new ones
        existing_embeddings = np.load(self.embeddings_file_path)
        updated_embeddings = np.vstack((existing_embeddings, embeddings))
        np.save(self.embeddings_file_path, updated_embeddings)

        # Append the new phrases
        with open(self.phrases_file_path, 'a') as f:
            f.writelines([phrase + '\n' for phrase in phrases])

    def delete_db_files(self):
        """
        Reset the database files by deleting the existing files and creating new empty ones.
        """
        if os.path.exists(self.embeddings_file_path):
            os.remove(self.embeddings_file_path)
        if os.path.exists(self.phrases_file_path):
            os.remove(self.phrases_file_path)
        self.initialize_db_files()

    def load_db_files(self):
        """
        Load the embeddings and phrases from the database files.
        """
        for page in tqdm(self.extractor_engine):
            chunks = self.string_processor.split_str_into_chunks(page)
            embeddings = self.string_processor.embed_chunks(chunks)
            # Separates the embeddings and phrases into chunks with overlap and saves them to the database
            self.append_db_files(embeddings, chunks)

    def calc_similarity(self, query: str):
        """
        Calculate the similarity between the query and the phrases in the database.
        """
        query_embedding = self.string_processor.embed_chunks([query])
        db_embeddings = np.load(self.embeddings_file_path)
        similarity = np.dot(db_embeddings, query_embedding.T).flatten()
        return similarity

    def find_k_most_similar(self, query: str, k: int):
        """
        Find the k most similar phrases to the query.
        """
        similarity = self.calc_similarity(query)
        top_k_indices = np.argsort(similarity)[-k:][::-1]
        with open(self.phrases_file_path, 'r') as f:
            phrases = f.readlines()
        return [phrases[i].strip() for i in top_k_indices]

if __name__ == '__main__':
    pdf_path = 'articles/attention_is_all_you_need.pdf'
    pdf_extractor = PdfExtractor(pdf_path)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    string_processor = StringProcessor(embedding_model=embedding_model,
                                       chunk_size=20, 
                                       chunk_overlap=5)
    
    db = VectorDatabase('db', pdf_extractor, string_processor)

    #db.delete_db_files()

    #db.load_db_files()

    query = "What is an embedding?"
    top_k = 5
    results = db.find_k_most_similar(query, top_k)
    print(results)
