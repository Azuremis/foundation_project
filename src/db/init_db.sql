-- Database initialization script for MNIST application
-- Creates the necessary tables if they don't exist

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_digit INTEGER NOT NULL,
    user_label INTEGER,
    confidence FLOAT,
    image_data BYTEA,
    correct BOOLEAN GENERATED ALWAYS AS (predicted_digit = user_label) STORED
); 