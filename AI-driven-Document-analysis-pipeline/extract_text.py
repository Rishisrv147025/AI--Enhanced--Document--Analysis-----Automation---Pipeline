# src/extract_text.py

import os
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
from io import StringIO
import logging
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path):
    try:
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''  # Handle case where no text is extracted
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return None

def preprocess_image(image):
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Apply binary thresholding
    threshold_image = gray_image.point(lambda p: p > 200 and 255)
    
    # Enhance the image (optional sharpening)
    enhancer = ImageEnhance.Sharpness(threshold_image)
    enhanced_image = enhancer.enhance(2.0)  # Increase sharpness
    
    # Optional: Use filter for denoising
    final_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
    
    return final_image

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from image {file_path}: {e}")
        return None

def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', engine='python')
        # If needed, clean data (e.g., remove empty rows, handle missing values)
        df_cleaned = df.fillna('')
        return df_cleaned.to_string(index=False, header=False)  # Return as plain text
    except Exception as e:
        logger.error(f"Error extracting text from CSV {file_path}: {e}")
        return None

def extract_text_from_xlsx(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
        text = ''
        for sheet_name, sheet_data in df.items():
            text += f"\nSheet: {sheet_name}\n"
            text += sheet_data.to_string(index=False, header=False)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from XLSX {file_path}: {e}")
        return None

def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    elif file_extension == '.csv':
        return extract_text_from_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return extract_text_from_xlsx(file_path)
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return None

# Parallelize text extraction for multiple files
def extract_text_from_multiple_files(file_paths):
    extracted_texts = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_file, file_paths)
        for file_path, text in zip(file_paths, results):
            extracted_texts[file_path] = text
    return extracted_texts

# Example usage
if __name__ == "__main__":
    file_paths = ["You can update the file path of the document/file/pdf here."]
    texts = extract_text_from_multiple_files(file_paths)
    for file_path, text in texts.items():
        print(f"Extracted from {file_path}: {text[:200]}...")
