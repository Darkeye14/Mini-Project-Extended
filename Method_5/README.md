# Cricket Emotion Detection using SVM

This project implements a Support Vector Machine (SVM) based emotion detector specifically designed for cricket-related text. The model classifies text as either "Emotional" or "Non-Emotional".

## Features

- SVM classifier with linear kernel
- TF-IDF text vectorization
- Custom cricket-specific stop words
- Simple and intuitive GUI interface
- Model persistence using joblib

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

```
Method_5/
├── emotion_detector_svm.py  # Main application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## How to Run

1. Ensure you have activated your virtual environment
2. Run the application:
   ```bash
   python emotion_detector_svm.py
   ```

## Usage

1. Click the "Train Model" button to train the SVM classifier on the dataset
2. Once trained, enter cricket-related text in the text box
3. Click "Analyze Emotion" to see the prediction

## Dataset

The application looks for text files in the `../dataset` directory. Each file should contain examples in the following format:

```
Input: [text to analyze]
Tag: [Emotional/Non Emotional]
```

## Notes

- The first time you run the application, it will download the required NLTK data
- The trained model is saved as `emotion_model_svm.joblib` in the same directory
- If the dataset is not found or is empty, the application will use a small built-in fallback dataset

## Troubleshooting

- If you encounter any issues, ensure all dependencies are installed
- Check that the dataset files are in the correct format
- Make sure you have write permissions in the application directory
