# Emotion Detection using Random Forest

This project implements an emotion detection system using a Random Forest classifier. The system can detect whether a given text is emotional or non-emotional.

## Features
- Trains a Random Forest model on labeled emotional/non-emotional text data
- Simple and intuitive GUI for text input and analysis
- Displays prediction confidence scores
- Real-time emotion detection

## Requirements
- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python emotion_detector.py
   ```
2. The application will automatically train the model on first launch
3. Enter text in the input box and click "Analyze Emotion" to detect emotion

## Model Details
- Uses TF-IDF for text vectorization
- Random Forest classifier with 100 trees
- Includes basic text preprocessing (stopword removal, punctuation removal, etc.)
- Automatically trains on the dataset in the `../dataset` directory

## File Structure
- `emotion_detector.py` - Main application file
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Dataset
Place your training data in the `../dataset` directory. Each file should contain labeled examples in the format:
```
Input: [text]
Tag: [Emotional/Non Emotional]
```
