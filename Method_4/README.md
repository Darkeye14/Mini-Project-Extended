# Emotion Detection using Logistic Regression

This project implements an emotion detection system using Logistic Regression. The system classifies text as either "Emotional" or "Non-Emotional".

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization with bigram support
- Logistic Regression classifier with balanced class weights
- Simple and intuitive GUI using CustomTkinter
- Real-time emotion analysis
- Cross-platform compatibility

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Setup Instructions

### 1. Clone the Repository (Optional)
If you haven't already, clone the repository:
```bash
git clone <repository-url>
cd Mini-Project/Method_4
```

### 2. Set Up Virtual Environment (Recommended)
It's recommended to use a virtual environment to manage dependencies:

#### Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install NLTK data (required for text processing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Running the Application

### Basic Usage
```bash
python emotion_detector_lr.py
```

### Command Line Options
- To run with a specific dataset path:
  ```bash
  python emotion_detector_lr.py --dataset ../path/to/dataset
  ```

## Dataset Structure
Place your training data in the `../dataset` directory with the following format in `.txt` files:
```
Input: [text]
Tag: [Emotional/Non-Emotional]
```

## Model Details
- **Vectorizer**: TF-IDF with max 2000 features
- **N-grams**: Unigrams and bigrams
- **Classifier**: Logistic Regression with balanced class weights
- **Features**: Text preprocessing with stopword removal and tokenization

## Usage Guide
1. The application will automatically train on startup using the dataset in `../dataset`
2. Enter text in the input box
3. Click "Analyze Emotion" to see the prediction
4. Use "Train Model" to retrain if needed

## Troubleshooting

### Common Issues
1. **NLTK Data Download Issues**:
   - If you encounter NLTK data download errors, run:
     ```bash
     python -m nltk.downloader stopwords punkt
     ```

2. **GUI Not Loading**:
   - Make sure all dependencies are installed
   - Try running with administrator privileges

3. **Dataset Not Found**:
   - Ensure the dataset is in the correct location (`../dataset` from the script's directory)
   - Check that the files follow the required format

## Performance Notes
- The model is trained on a small dataset by default
- For better performance, consider adding more training examples
- The model includes basic text preprocessing and stopword removal
- Training progress is shown in the status bar

## License
[Specify your license here]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
