# Cricket Emotion Detection - Ensemble Model

This project implements an ensemble model for detecting emotions in cricket-related text. The ensemble combines Logistic Regression and Random Forest classifiers to improve prediction accuracy and robustness.

## Features

- **Ensemble Learning**: Combines the strengths of multiple models for better prediction
- **Advanced Text Processing**: Includes custom tokenization and cleaning specific to cricket terminology
- **User-friendly GUI**: Built with CustomTkinter for an intuitive user experience
- **Model Persistence**: Saves trained models for future use
- **Detailed Reporting**: Provides confidence scores and class probabilities

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

3. Ensure you have the NLTK stopwords data:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Directory Structure

```
Method_7/
├── emotion_detector_ensemble.py  # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── dataset/                     # Directory containing training data
    ├── cricket_emotion_generated_01.txt
    ├── cricket_emotion_generated_02.txt
    └── ...
```

## Usage

1. **Training the Model**:
   - Run the application: `python emotion_detector_ensemble.py`
   - Click the "Train Model" button to train the ensemble model on the dataset
   - The model will be saved as `emotion_model_ensemble.joblib` in the same directory

2. **Analyzing Text**:
   - Enter cricket-related text in the input box
   - Click "Analyze Emotion" to see the prediction
   - The result will show the predicted emotion (Emotional/Non Emotional), confidence score, and class probabilities

## Model Architecture

The ensemble model combines:

1. **Logistic Regression**
   - TF-IDF vectorization
   - L2 regularization
   - Balanced class weights

2. **Random Forest**
   - 100 decision trees
   - Maximum depth of 10
   - Balanced class weights

Both models are combined using soft voting, where the final prediction is based on the average predicted probabilities of the base models.

## Dataset

The model is trained on cricket-related text data with the following format:

```
Input: [cricket-related text]
Tag: [Emotional/Non Emotional]
```

Example:
```
Input: What an amazing catch by the fielder!
Tag: Emotional

Input: The batsman defended the ball to the off side.
Tag: Non Emotional
```

## Performance

The model's performance is evaluated using:
- Accuracy: Percentage of correctly classified examples
- Classification report: Includes precision, recall, and F1-score for each class

## Troubleshooting

- **Model not training**: Ensure the dataset directory exists and contains properly formatted `.txt` files
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **NLTK data not found**: Run `nltk.download('stopwords')` in a Python shell

## License

This project is licensed under the MIT License - see the LICENSE file for details.
