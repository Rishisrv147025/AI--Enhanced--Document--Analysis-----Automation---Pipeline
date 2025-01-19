# src/utils.py

import torch
from transformers import BertTokenizer

def load_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

def load_tag_map():
    return {
        0: 'O',
        1: 'INVOICE_NUMBER',
        2: 'VENDOR',
        3: 'AMOUNT',
        4: 'DATE',
        5: 'EXPIRATION_DATE'
    }
