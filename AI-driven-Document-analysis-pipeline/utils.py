# src/utils.py

import torch
from transformers import BertTokenizer

def load_tokenizer(model_name="bert-base-uncased"):
    """
    Load a tokenizer for the NER model.
    :param model_name: Name of the pretrained model.
    :return: Tokenizer object
    """
    return BertTokenizer.from_pretrained(model_name)

def load_tag_map():
    """
    Load a tag map (e.g., mapping from tag indices to entity names).
    :return: A dictionary of tag to entity names
    """
    return {
        0: 'O',
        1: 'INVOICE_NUMBER',
        2: 'VENDOR',
        3: 'AMOUNT',
        4: 'DATE',
        5: 'EXPIRATION_DATE'
    }
