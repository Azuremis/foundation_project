"""
Database initialization script for MNIST application.
Creates the necessary tables if they don't exist.
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

# Add parent directory to path to import db_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_utils import get_db_connection

def init_database():
    """
    Initialize the database by creating tables if they don't exist.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_digit INTEGER NOT NULL,
            user_label INTEGER,
            confidence FLOAT,
            image_data BYTEA,
            correct BOOLEAN GENERATED ALWAYS AS (predicted_digit = user_label) STORED
        );
        """)
        
        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Database initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize the database
    success = init_database()
    
    if success:
        print("Database tables created successfully!")
    else:
        print("Failed to initialize database. Check the error message above.")
        sys.exit(1) 