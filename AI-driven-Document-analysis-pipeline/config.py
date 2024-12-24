# config.py

MODEL_PATH = "path_to_saved_ner_model.pth"
EMBEDDING_PATH = "ner_model/embeddings/glove.6B.100d.txt"
VOCAB_SIZE = 10000
EMBED_DIM = 100
HIDDEN_DIM = 256
NUM_TAGS = 10  # Update based on your NER task

DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'document_processing'
}
