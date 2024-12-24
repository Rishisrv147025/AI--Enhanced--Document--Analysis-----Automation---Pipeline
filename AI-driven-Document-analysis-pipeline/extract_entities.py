import torch
import numpy as np
from ner_model import BiLSTM_CRF_Attn
from torch.utils.data import DataLoader
from src.utils import tokenize, prepare_data
from collections import defaultdict
import time

# Load the BiLSTM-CRF + Attention NER model
def load_ner_model(model_path, embedding_matrix, vocab_size, embed_dim, hidden_dim, num_tags):
    """
    Load a pre-trained BiLSTM-CRF model with attention for NER.
    """
    model = BiLSTM_CRF_Attn(vocab_size, embed_dim, hidden_dim, num_tags, embedding_matrix)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def tokenize_and_prepare(text, tokenizer, max_length=512):
    """
    Tokenizes and prepares text for model inference.
    Pads and truncates to the max length.
    """
    tokens = tokenize(text)  # Tokenizing the input text
    token_ids = tokenizer(tokens)  # Converting tokens to ids
    token_ids = token_ids[:max_length]  # Truncate to max length
    token_ids = token_ids + [0] * (max_length - len(token_ids))  # Pad to max length
    return torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension

def extract_entities_batch(texts, model, tokenizer, tag_map, max_length=512):
    """
    Extract entities from a batch of text inputs.
    """
    batch_input = [tokenize_and_prepare(text, tokenizer, max_length) for text in texts]
    batch_input = torch.cat(batch_input, dim=0)  # Concatenate into a batch
    
    # Get the model's output
    emissions, _, _ = model(batch_input)
    
    # Decode the output to get entities for each text in the batch
    predicted_tags = model.crf.decode(emissions)
    all_entities = [decode_entities(tokens, tags, tag_map) for tokens, tags in zip(texts, predicted_tags)]
    
    return all_entities

def decode_entities(tokens, predicted_tags, tag_map):
    """
    Decodes predicted tags into entities and maps to a standardized tag.
    Filters out unnecessary tags.
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, predicted_tags):
        tag_name = tag_map.get(tag, 'O')  # Default to 'O' (Outside any entity)
        
        # Start of a new entity
        if tag_name != 'O':
            if current_entity is None:
                current_entity = {'text': token, 'type': tag_name}
            else:
                current_entity['text'] += ' ' + token
                current_entity['type'] = tag_name  # Update to the current entity type
        else:
            # End of a current entity
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # Append any remaining entity
    if current_entity is not None:
        entities.append(current_entity)
    
    return entities

def extract_entities(text, model, tokenizer, tag_map):
    """
    Extracts entities from a single text input.
    """
    tokens = tokenize(text)
    token_ids = tokenizer(tokens)
    token_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
    
    # Get the model's output
    emissions, context_vector, attn_weights = model(token_tensor)
    
    # Decode the output to get entities
    predicted_tags = model.crf.decode(emissions)
    entities = decode_entities(tokens, predicted_tags[0], tag_map)
    
    return entities

def inference_speed_test(model, texts, tokenizer, tag_map):
    """
    A function to test the inference speed for a batch of texts.
    Measures time taken for processing the batch.
    """
    start_time = time.time()
    _ = extract_entities_batch(texts, model, tokenizer, tag_map)
    end_time = time.time()
    
    print(f"Inference Time for {len(texts)} texts: {end_time - start_time:.4f} seconds")
    
# Example usage
if __name__ == '__main__':
    # Assume the following variables are initialized
    model_path = 'path_to_model.pth'
    embedding_matrix = np.random.rand(10000, 300)  # Example embedding matrix
    vocab_size = 10000
    embed_dim = 300
    hidden_dim = 128
    num_tags = 9  # Example number of tags
    tokenizer = lambda x: x.split()  # Simple tokenizer (space-separated words)
    tag_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG'}  # Example tag map

    # Load the NER model
    model = load_ner_model(model_path, embedding_matrix, vocab_size, embed_dim, hidden_dim, num_tags)

    # Extract entities from a single text
    text = "Elon Musk is the CEO of SpaceX."
    entities = extract_entities(text, model, tokenizer, tag_map)
    print("Entities:", entities)

    # Test batch inference speed
    batch_texts = [
        "Elon Musk is the CEO of SpaceX.",
        "Google was founded by Larry Page and Sergey Brin."
    ]
    inference_speed_test(model, batch_texts, tokenizer, tag_map)
