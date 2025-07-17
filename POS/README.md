# POS-based Cricket Emotion Detector

This project implements a Parts of Speech (POS) based emotion detector specifically designed for cricket-related text. It analyzes the grammatical structure of sentences to identify emotional content.

## Features

- POS tagging for emotion detection
- Custom scoring for different parts of speech
- Emotional phrase detection
- Simple and intuitive GUI
- Detailed analysis of emotional components

## Requirements

- Python 3.8+
- NLTK
- CustomTkinter
- Pillow

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python pos_emotion_detector.py
   ```

## Usage

1. **Training the Model**:
   - Click the "Train Model" button to train the POS-based emotion detector
   - The model will analyze the dataset and learn which parts of speech are indicative of emotion

2. **Analyzing Text**:
   - Enter cricket-related text in the input box
   - Click "Analyze Emotion" to see the prediction
   - The results will show:
     - Prediction (Emotional/Non Emotional)
     - POS-based emotion score and threshold
     - Emotional terms found
     - All POS tags in the text

## How It Works

The detector uses the following approach:

1. **POS Tagging**:
   - Tags each word in the text with its part of speech
   - Focuses on adjectives, adverbs, verbs, and interjections

2. **Scoring**:
   - Assigns weights to different POS tags based on their emotional significance
   - Detects emotional phrases and intensifiers
   - Calculates a total emotion score

3. **Decision**:
   - Compares the score against a threshold
   - Classifies the text as Emotional or Non Emotional

## Output Interpretation

- **Prediction**: Final classification (Emotional/Non Emotional)
- **Score**: Numeric score based on POS analysis
- **Threshold**: Decision boundary for classification
- **Emotional Terms**: Words and phrases contributing to emotional score
- **All POS Tags**: Complete part-of-speech analysis of the input text

## Notes

- The model requires NLTK's punkt and averaged_perceptron_tagger resources
- These will be downloaded automatically during the first run
- The dataset should be in the `dataset` folder with files named `cricket_emotion_generated_*.txt`
