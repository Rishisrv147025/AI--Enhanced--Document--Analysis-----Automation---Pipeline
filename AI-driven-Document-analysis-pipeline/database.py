import sqlite3
import logging
from datetime import datetime
import asyncio

# Set up logging for database operations
logging.basicConfig(filename='invoice_data.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def create_database(table_name, columns):
    """
    Create the SQLite database and the table dynamically based on the provided columns.
    :param table_name: The name of the table to be created.
    :param columns: Dictionary of column names and their corresponding data types.
    """
    conn = sqlite3.connect('invoice_data.db')
    c = conn.cursor()

    # Create table query dynamically based on columns
    columns_definition = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition});"

    c.execute(create_table_query)
    conn.commit()
    conn.close()
    logging.info(f"Table '{table_name}' created with columns: {columns}")

def validate_document_data(document_data, expected_columns):
    """
    Validate the structure and contents of the document data against the expected columns.
    :param document_data: Dictionary with document information.
    :param expected_columns: List of expected column names.
    :return: Boolean indicating whether the data is valid.
    """
    missing_columns = [col for col in expected_columns if col not in document_data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return True

async def save_invoice_data(table_name, document_data, expected_columns):
    """
    Save the processed document data to a specified SQLite table dynamically.
    :param table_name: The name of the table where the data will be saved.
    :param document_data: Dictionary with document information and decision.
    :param expected_columns: List of expected column names for validation.
    """
    try:
        # Validate the input data against expected columns
        validate_document_data(document_data, expected_columns)

        # Prepare the columns and values for insertion
        columns = ", ".join(document_data.keys())
        values = tuple(document_data.values())

        # Create a connection and cursor for the database
        conn = sqlite3.connect('invoice_data.db')
        c = conn.cursor()

        # Insert data into the database dynamically
        placeholders = ", ".join(["?" for _ in document_data])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        c.execute(insert_query, values)

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        # Log the successful saving of data
        logging.info(f"Invoice data saved to '{table_name}': {document_data}")
        print(f"Saved document data: {document_data}")

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        print(f"Error: {ve}")
    
    except sqlite3.DatabaseError as db_err:
        logging.error(f"Database error: {db_err}")
        print(f"Error: {db_err}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")

# Function to run the save operation asynchronously
async def main():
    # Example: Dynamic table name and columns
    table_name = 'invoice_data'

    # Columns and their data types (can change for different datasets)
    columns = {
        'amount': 'REAL',
        'vendor': 'TEXT',
        'vendor_type': 'INTEGER',
        'date_feature': 'INTEGER',
        'decision': 'TEXT',
        'timestamp': 'DATETIME'
    }

    # Create the table dynamically based on columns
    create_database(table_name, columns)

    # Mocked invoice data
    document_data = {
        'amount': 1500.75,
        'vendor': 'Amazon',
        'vendor_type': 1,
        'date_feature': 19134,  # Example date feature (days since epoch)
        'decision': 'Approved',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # List of expected columns for validation
    expected_columns = list(columns.keys())

    # Save the document data to the database
    await save_invoice_data(table_name, document_data, expected_columns)

# Run the async save operation
asyncio.run(main())
