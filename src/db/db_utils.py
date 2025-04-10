"""
Database utility functions for connecting to PostgreSQL and performing queries.
"""
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """
    Create a connection to the PostgreSQL database.
    
    Returns:
        connection: A psycopg2 connection object
    """
    # Get database configuration from environment variables with fallback
    db_config = {
        'dbname': os.environ.get('DB_NAME', 'mnistlogs'),
        'user': os.environ.get('DB_USER', 'mnistuser'),
        'password': os.environ.get('DB_PASSWORD', ''),
        'host': os.environ.get('DB_HOST', 'localhost')
    }
    
    # Check if password is empty and warn (but don't print the password)
    if not db_config['password']:
        print("Warning: Database password is empty! Check your .env file.")
    
    try:
        connection = psycopg2.connect(**db_config)
        return connection
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        # Don't include password in the error message
        error_msg = f"Could not connect to database {db_config['dbname']} " \
                    f"on {db_config['host']} as user {db_config['user']}"
        raise ConnectionError(error_msg) from e

def log_prediction(predicted_digit, user_label=None, confidence=None, image_data=None):
    """
    Log a prediction to the database.
    
    Args:
        predicted_digit (int): The predicted digit (0-9)
        user_label (int, optional): The actual digit according to user
        confidence (float, optional): Confidence score of prediction
        image_data (bytes, optional): Binary image data
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Prepare SQL query based on provided parameters
        if all(param is not None for param in [user_label, confidence, image_data]):
            query = """
                INSERT INTO predictions 
                (predicted_digit, user_label, confidence, image_data) 
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (predicted_digit, user_label, confidence, image_data))
        elif user_label is not None and confidence is not None:
            query = """
                INSERT INTO predictions 
                (predicted_digit, user_label, confidence) 
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (predicted_digit, user_label, confidence))
        elif user_label is not None:
            query = """
                INSERT INTO predictions 
                (predicted_digit, user_label) 
                VALUES (%s, %s)
            """
            cursor.execute(query, (predicted_digit, user_label))
        else:
            query = """
                INSERT INTO predictions 
                (predicted_digit) 
                VALUES (%s)
            """
            cursor.execute(query, (predicted_digit,))
            
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

def get_prediction_stats():
    """
    Get statistics about predictions.
    
    Returns:
        dict: Statistics about predictions
    """
    stats = {
        "total": 0,
        "correct": 0,
        "accuracy": 0.0
    }
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats["total"] = cursor.fetchone()[0]
        
        # Get correct predictions
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE correct = TRUE")
        stats["correct"] = cursor.fetchone()[0]
        
        # Calculate accuracy
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching stats: {e}")
    
    return stats

def get_prediction_history(limit=50):
    """
    Get prediction history from the database.
    
    Args:
        limit (int): Maximum number of records to return
        
    Returns:
        list: List of dictionaries containing prediction history
    """
    history = []
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent predictions with timestamp, prediction, and user label
        query = """
            SELECT 
                timestamp, 
                predicted_digit, 
                user_label,
                confidence,
                correct
            FROM predictions
            WHERE user_label IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        
        # Convert results to list of dictionaries
        columns = ["timestamp", "predicted_digit", "user_label", "confidence", "correct"]
        for row in cursor.fetchall():
            history.append(dict(zip(columns, row)))
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error fetching prediction history: {e}")
    
    return history 