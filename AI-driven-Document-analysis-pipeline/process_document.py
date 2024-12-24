# src/process_document.py

from src.extract_text import extract_text_from_pdf, extract_text_from_image, extract_text_from_csv, extract_text_from_xlsx
from src.extract_entities import extract_entities, load_ner_model
from src.decision_engine import DecisionEngineAdvanced
from src.database import save_invoice_data
from src.utils import load_tokenizer, load_tag_map

def process_document(file_path, file_type, decision_engine, ner_model=None, tokenizer=None, tag_map=None):
    """
    Main function for processing various document types and making decisions.
    :param file_path: Path to the input document.
    :param file_type: Type of document (PDF, CSV, XLSX, Image, etc.).
    :param decision_engine: The decision-making engine based on the advanced NN model.
    :param ner_model: Pretrained NER model (optional, can be None for documents without NER).
    :param tokenizer: Tokenizer for the NER model (optional).
    :param tag_map: Mapping from tags to entity names (optional).
    :return: Processed document data with decisions.
    """
    # Extract text based on document type
    file_extractors = {
        "pdf": extract_text_from_pdf,
        "image": extract_text_from_image,
        "csv": extract_text_from_csv,
        "xlsx": extract_text_from_xlsx
    }

    if file_type not in file_extractors:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    text = file_extractors[file_type](file_path)
    
    # If NER model is provided, extract entities
    document_data = {}
    if ner_model and tokenizer and tag_map:
        entities = extract_entities(text, ner_model, tokenizer, tag_map)
        document_data.update(entities)
    
    # Make decision based on the advanced model
    decision = decision_engine.make_decision(document_data)
    
    # Add decision to the document data
    document_data["decision"] = decision
    
    # Save document data to the database
    save_invoice_data(document_data)
    
    return document_data
