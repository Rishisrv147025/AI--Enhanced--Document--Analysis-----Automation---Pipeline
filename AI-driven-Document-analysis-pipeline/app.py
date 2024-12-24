import streamlit as st
from datetime import datetime
import asyncio
import pytesseract
import pdfplumber
from PIL import Image
import io
import os
import sqlite3
import logging
from database import save_invoice_data, create_database  # Assuming database.py is available

# Set up logging for the Streamlit app
logging.basicConfig(filename='invoice_app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Streamlit app UI
st.title('Invoice Processing System')

# Sidebar with options
st.sidebar.header('Invoice Operations')
operation = st.sidebar.selectbox('Select Operation', ['Upload Invoice', 'View Saved Data'])

# Load the pre-trained NER model (For simplicity, we assume it's already trained)
def load_model():
    model_path = "path_to_saved_ner_model.pth"  # Set the correct path to your model
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Function to extract text from image using pytesseract (OCR)
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return None

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

# Function to extract information from the invoice using NER
def extract_invoice_data(invoice_text, model, tokenizer):
    inputs = tokenizer(invoice_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_tags = torch.argmax(logits, dim=-1).squeeze().tolist()
    tag_map = load_tag_map()
    tokens = tokenizer.tokenize(invoice_text)
    extracted_data = {tag_map[tag]: tokens[i] for i, tag in enumerate(predicted_tags) if tag != 0}  # Ignore 'O' tag
    return extracted_data

# Function to handle the invoice data processing
async def process_invoice(invoice_file):
    # Extract text from uploaded file (image or PDF)
    if invoice_file.type.startswith("image"):
        invoice_text = extract_text_from_image(invoice_file)
    elif invoice_file.type == "application/pdf":
        invoice_text = extract_text_from_pdf(invoice_file)
    else:
        st.error("Invalid file type. Please upload an image or PDF.")
        return

    if not invoice_text:
        st.error("Failed to extract text from the document. Please try again with a different file.")
        return

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Extract invoice data using NER
    extracted_data = extract_invoice_data(invoice_text, model, tokenizer)
    
    # Log the extracted data
    st.write(f"Extracted Invoice Data: {extracted_data}")
    logging.info(f"Extracted Data: {extracted_data}")

    # Define the columns and table name (from the database part)
    table_name = 'invoice_data'
    columns = {
        'amount': 'REAL',
        'vendor': 'TEXT',
        'vendor_type': 'INTEGER',
        'date_feature': 'INTEGER',
        'decision': 'TEXT',
        'timestamp': 'DATETIME'
    }

    # Create the database table dynamically if it doesn't exist
    create_database(table_name, columns)

    # Add the extracted data into the document_data dictionary
    document_data = {
        'amount': extracted_data.get('AMOUNT', 0.0),
        'vendor': extracted_data.get('VENDOR', 'Unknown'),
        'vendor_type': 1,  # This can be determined or updated based on logic
        'date_feature': 19134,  # Placeholder value for the date feature
        'decision': 'Pending',  # Set initial decision to "Pending"
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Expected columns (for validation)
    expected_columns = list(columns.keys())

    # Save the extracted data to the database
    await save_invoice_data(table_name, document_data, expected_columns)

# Streamlit app logic based on selected operation
if operation == 'Upload Invoice':
    # Invoice file upload (image or PDF)
    uploaded_file = st.file_uploader("Upload Invoice File", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image (if it's an image)
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Invoice', use_column_width=True)
        
        # Process the uploaded invoice
        if st.button("Extract and Save Invoice Data"):
            asyncio.run(process_invoice(uploaded_file))

elif operation == 'View Saved Data':
    # Display saved data from the database
    conn = sqlite3.connect('invoice_data.db')
    c = conn.cursor()
    
    # Retrieve all saved data
    c.execute("SELECT * FROM invoice_data")
    rows = c.fetchall()
    
    # Display data in a table format
    if rows:
        st.write("Saved Invoice Data:")
        st.write(rows)
    else:
        st.write("No data found in the database.")

