# Cricket Emotion Detection using Custom TF-IDF

This project implements a custom TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer from scratch and uses it for emotion detection in cricket-related text. The implementation includes a user-friendly GUI for training the model and making predictions.

## Features

- Custom TF-IDF vectorizer implemented from scratch
- Emotion detection in cricket-related text
- Interactive GUI for training and predictions
- Detailed analysis of emotional and non-emotional terms
- Visualization of TF-IDF scores and decision thresholds

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Model**:
   - Run the application: `python tfidf_emotion_detector.py`
   - Click the "Train Model" button to train the TF-IDF model on the dataset
   - The model will analyze the dataset and calculate TF-IDF scores for all terms

2. **Analyzing Text**:
   - Enter cricket-related text in the input box
   - Click "Analyze Emotion" to see the prediction
   - The results will show:
     - Prediction (Emotional/Non Emotional)
     - TF-IDF score and decision threshold
     - List of emotional and non-emotional terms found
     - All terms from the input text

## How It Works

1. **TF-IDF Vectorization**:
   - Term Frequency (TF): Measures how often a term appears in a document
   - Inverse Document Frequency (IDF): Measures how important a term is across all documents
   - TF-IDF = TF * IDF

2. **Emotion Detection**:
   - The model calculates a score for each input text based on TF-IDF weights
   - A threshold is determined during training to classify texts as Emotional or Non-Emotional
   - The decision is based on whether the score exceeds the threshold

## Output Interpretation

- **TF-IDF Score**: The calculated score for the input text
- **Decision Threshold**: The score threshold for classification
- **Emotional Terms**: Terms that contribute to emotional classification
- **Non-Emotional Terms**: Terms that don't indicate emotion
- **All Terms**: All terms extracted from the input text

## Notes

- The model requires a dataset of cricket-related texts with emotion labels
- The dataset should be in the `dataset` folder with files named `cricket_emotion_generated_*.txt`
- Each file should contain examples in the format:
  ```
  Input: [text]
  Tag: [Emotional/Non Emotional]
  ```

## License

This project is licensed under the MIT License.
