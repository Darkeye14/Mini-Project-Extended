import os
import re
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Create a simple tokenizer that doesn't rely on punkt
class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())

# Set NLTK data path if not found
nltk_data = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data):
    os.makedirs(nltk_data, exist_ok=True)
    nltk.data.path.append(nltk_data)

class EmotionDetectorLR:
    def __init__(self):
        # Create a custom stop words list
        custom_stop_words = set(stopwords.words('english'))
        # Remove words that might be important for emotion detection
        custom_stop_words -= {'not', 'no', 'never', 'very', 'too', 'so', 'much', 'many', 'more', 'most', 'such', 'only', 'own'}
        
        # Cricket terms that should not be in stopwords
        cricket_terms = {
            'kohli', 'rohit', 'rahul', 'pandya', 'bumrah', 'bhuvi', 'chahal', 'dhoni', 'dravid', 'sachin', 'ganguly',
            'virat', 'sharma', 'iyer', 'surya', 'sky', 'jadeja', 'ashwin', 'pant', 'kishan', 'shami', 'siraj', 'kuldeep',
            'ipl', 't20', 'odi', 'test', 'worldcup', 'world cup', 'six', 'four', 'wicket', 'century', 'fifty', 'duck',
            'boundary', 'catch', 'stumping', 'runout', 'yorker', 'bouncer', 'googly', 'doosra', 'carrom', 'sweep', 'reverse',
            'champions', 'trophy', 'toss', 'innings', 'powerplay', 'death over', 'super over', 'man of the match'
        }
        
        # Remove cricket terms from stopwords
        custom_stop_words = custom_stop_words - cricket_terms
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,  # Increased to capture more features
                ngram_range=(1, 3),  # Using unigrams, bigrams, and trigrams
                stop_words=list(custom_stop_words),
                min_df=2,           # Minimum document frequency
                max_df=0.85,        # Maximum document frequency
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'  # More permissive token pattern
            )),
            ('classifier', LogisticRegression(
                max_iter=2000,      # Increased iterations for better convergence
                random_state=42,
                class_weight='balanced',
                C=1.0,             # Regularization strength
                solver='liblinear', # Better for small datasets
                penalty='l2',       # L2 regularization
                n_jobs=-1           # Use all cores
            ))
        ])
        self.is_trained = False

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle common contractions and abbreviations
        contractions = {
            r"won't": 'will not',
            r"can't": 'can not',
            r"n't": ' not',
            r"'re": ' are',
            r"'s": ' is',
            r"'d": ' would',
            r"'ll": ' will',
            r"'t": ' not',
            r"'ve": ' have',
            r"'m": ' am',
            r"i'm": 'i am',
            r"u": 'you',
            r"ur": 'your',
            r"thx": 'thanks'
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        
        # Standardize cricket player names and terms
        cricket_terms = {
            r'king kohli': 'virat kohli',
            r'hitman': 'rohit sharma',
            r'brohit': 'rohit sharma',
            r'msd': 'ms dhoni',
            r'thala': 'ms dhoni',
            r'captain cool': 'ms dhoni',
            r'ipl': 'indian premier league',
            r't20i': 't20 international',
            r'odi': 'one day international',
            r'\b4\b': 'four',
            r'\b6\b': 'six',
            r'\b2\b': 'two',
            r'\b1\b': 'one'
        }
        
        for pattern, replacement in cricket_terms.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle repeated letters (e.g., 'soooo' -> 'sooo')
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        
        # Add space around punctuation for better tokenization
        text = re.sub(r'([!?])\1+', r' \1 ', text)  # Multiple ! or ?
        text = re.sub(r'([.,!?()])', r' \1 ', text)  # Single punctuation
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add special tokens for emotional emphasis
        emotional_phrases = [
            (r'\b(stunn(ing|ed)?|amaz(ing|ed)|wow|incredible|unbelievable|fantastic|brilliant|superb|outstanding|exceptional|extraordinary|phenomenal|mind.?blow(ing)?|speechless|awe(some)?|thrill(ing|ed)|excit(ing|ed))\b', ' _strong_pos_ '),
            (r'\b(goosebumps|chill(s|ed)|shiver(ed|ing)|tear(s|ing)|cry(ing)?|emotional|overwhelm(ed|ing))\b', ' _phys_emot_ '),
            (r'\b(love|lovely|adore|adored|passion(ate)?|heart|soul|spirit|pride|proud|honor|honour|respect|admire|cherish|treasure|value|appreciate|enjoy|delight(ed|ful)|pleasur(e|able)|happy|happiness|joy(ful)?|ecstatic|bliss(ful)?|elat(ed|ing)|jubilant|overjoyed|thrilled|euphoric|rapturous|gleeful|jovial|jolly|merry|cheer(ful|y)|glad|content(ed)?|satisf(y|ied)|gratif(y|ied)|fulfill(ed|ing)|triumph(ant)?|victor(y|ious)|win(ning)?|succeed|success(ful)?|achieve(ment)?|accomplish(ed|ment)?|excel(lent|led)?|superior|best|great(est)?|wonder(ful|ous)|marvel(ous|lous)|splendid|glorious|magnificent|majestic|sublime|divine|heavenly|perfect|flawless|impeccable|exquisite|elegant|beautiful|gorgeous|stunning|breathtaking|mesmeriz(ing|ed)|captivat(ing|ed)|enchant(ing|ed)|charm(ing|ed)|allur(ing|ed)|fascinat(ing|ed)|enthrall(ing|ed)|spellbind(ing|ed)|hypnotiz(ing|ed)|transfix(ed|ing))\b', ' _pos_emot_ '),
            (r'\b(hate|hatred|disgust(ing|ed)|repulsive|revolting|sickening|nauseating|vile|loathsome|abhorrent|abominable|detestable|despicable|contemptible|disgraceful|shameful|deplorable|lamentable|regrettable|unfortunate|sad|sadden(ed|ing)|unhappy|miserable|wretched|sorrow(ful)?|grief|griev(e|ing)|heartbroken|heartbreak(ing)?|anguish(ed)?|distress(ed|ing)|despair(ing)?|hopeless|desperate|devastat(ing|ed)|crush(ed|ing)|destroy(ed)?|ruin(ed)?|wreck(ed)?|demolish(ed)?|annihilat(ed|ing)|obliterat(ed|ing)|eradicate(d|ing)|eliminate(d|ing)|remove(d)?|erase(d)?|delete(d)?|kill(ed|ing)?|murder(ed|ing)?|slay(ing|n)?|assassinat(ed|ing)|execute(d|ing)|hang(ed|ing)|lynch(ed|ing)|torture(d|ing)|torment(ed|ing)|abuse(d|ing)|mistreat(ed|ing)|maltreat(ed|ing)|ill.treat(ed|ing)|harm(ed|ing)|hurt(ing)?|injure(d|ing)|wound(ed|ing)|damage(d|ing))\b', ' _neg_emot_ ')
        ]
        
        for pattern, replacement in emotional_phrases:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up any extra spaces that might have been introduced
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        total_files = 0
        total_examples = 0
        
        # Count total files first
        all_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
        total_files = len(all_files)
        print(f"Found {total_files} dataset files")
        
        try:
            # Sort files to ensure consistent ordering
            all_files.sort()
            
            for filename in all_files:
                file_path = os.path.join(dataset_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    # Split by 'Input:' or 'Input:' with any whitespace
                    examples = re.split(r'\nInput\s*:', content)
                    # First element is empty or content before first Input:
                    if examples and not examples[0].strip():
                        examples = examples[1:]
                    
                    print(f"\nProcessing {filename} - Found {len(examples)} examples")
                    file_examples = 0
                    
                    for example in examples:
                        if not example.strip():
                            continue
                            
                        # Extract text and tag using more flexible regex
                        tag_match = re.search(r'Tag\s*:\s*([^\n]*)', example, re.IGNORECASE)
                        if not tag_match:
                            print(f"  - Warning: Could not find Tag in example: {example[:100]}...")
                            continue
                            
                        # Get the text part (everything before Tag:)
                        text = example[:tag_match.start()].strip()
                        tag = tag_match.group(1).strip()
                        
                        # Debug print for first few examples
                        if total_examples < 3:  # Print first 3 examples for verification
                            print(f"  - Example {total_examples + 1}:")
                            print(f"    Text: {text[:80]}{'...' if len(text) > 80 else ''}")
                            print(f"    Raw Tag: '{tag}'")
                        
                        # Normalize tag - handle different cases, whitespace, and hyphenation
                        tag_lower = tag.lower().strip()
                        
                        # Debug output for the first few tags
                        if total_examples < 3:  # Only show for first few examples
                            print(f"    Raw tag: '{tag}'")
                            print(f"    Lowercase tag: '{tag_lower}'")
                        
                        # Handle different tag formats
                        if 'emotional' in tag_lower:
                            if tag_lower.startswith('non') or 'non' in tag_lower.split():
                                label = 0  # Non Emotional
                                normalized_tag = 'Non Emotional'
                                if total_examples < 3:
                                    print(f"    Classified as: {normalized_tag} (label: {label})")
                            else:
                                label = 1  # Emotional
                                normalized_tag = 'Emotional'
                                if total_examples < 3:
                                    print(f"    Classified as: {normalized_tag} (label: {label})")
                        else:
                            print(f"  - Warning: Unrecognized tag format: '{tag}'. Expected 'Emotional' or 'Non Emotional'. Skipping example.")
                            continue  # Skip examples with invalid tags
                        
                        if text:
                            cleaned_text = self.clean_text(text)
                            texts.append(cleaned_text)
                            labels.append(label)
                            total_examples += 1
                            file_examples += 1
                            
                            if total_examples <= 3:  # Print classification of first 3 examples
                                print(f"    Classified as: {normalized_tag} (label: {label})")
                    
                    print(f"  - Successfully processed {file_examples} examples from {filename}")
            
            print(f"\nTotal examples loaded: {len(texts)}")
            print(f"Emotional examples: {sum(labels)}")
            print(f"Non-emotional examples: {len(labels) - sum(labels)}")
            
            if len(set(labels)) >= 2 and len(texts) >= 2:
                return pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
            else:
                print("Warning: Insufficient data or only one class found in the dataset.")
                print(f"Total examples: {len(texts)}, Classes found: {set(labels)}")
                
                # If we have examples but only one class, print some examples
                if len(texts) > 0:
                    print("\nFirst few examples and their labels:")
                    for i in range(min(5, len(texts))):
                        print(f"  {i+1}. Label: {labels[i]}, Text: {texts[i][:80]}...")
                
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Fallback to a balanced dummy dataset if loading fails
        print("\nUsing fallback dataset")
        return pd.DataFrame({
            'text': [
                'I am so happy today!', 
                'This is a neutral statement.',
                'I feel really excited about this!',
                'The weather is normal today.'
            ],
            'label': [1, 0, 1, 0]
        })

    def train(self, dataset_path):
        try:
            print("\n" + "="*50)
            print("STARTING MODEL TRAINING")
            print("="*50)
            
            # Load the dataset with detailed logging
            print("\nLoading dataset...")
            df = self.load_dataset(dataset_path)
            
            if df is None or len(df) == 0:
                error_msg = "Error: No data to train on."
                print(error_msg)
                return 0.0, error_msg
            
            # Check class distribution
            class_dist = df['label'].value_counts()
            print("\nClass distribution in training data:")
            print(f"- Emotional: {class_dist.get(1, 0)} examples")
            print(f"- Non Emotional: {class_dist.get(0, 0)} examples")
            
            if len(class_dist) < 2:
                error_msg = "Error: Need both classes for training"
                print(error_msg)
                return 0.0, error_msg
            
            # Adjust test size based on dataset size
            test_size = 0.2  # Default test size
            min_test_size = 2  # Minimum number of test samples per class
            
            # Calculate minimum required test size
            min_required_test = min(class_dist) * 0.2
            if min_required_test < min_test_size:
                test_size = min_test_size * 2 / len(df)  # Adjust test size for small datasets
                test_size = min(0.3, test_size)  # Don't use more than 30% for test
                print(f"\nAdjusting test size to {test_size:.2f} for small dataset")
            
            # Split the dataset
            print("\nSplitting dataset...")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    df['text'], 
                    df['label'], 
                    test_size=test_size,
                    random_state=42, 
                    stratify=df['label']
                )
            except ValueError as ve:
                # Fallback to non-stratified split if stratification fails
                print(f"Warning: Stratified split failed: {ve}. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    df['text'], 
                    df['label'], 
                    test_size=test_size,
                    random_state=42
                )
            
            print(f"Training on {len(X_train)} examples, validating on {len(X_test)} examples")
            
            # Print some training examples
            print("\nSample training examples:")
            for i in range(min(3, len(X_train))):
                label = 'Emotional' if y_train.iloc[i] == 1 else 'Non Emotional'
                print(f"  {i+1}. [{label}] {X_train.iloc[i][:80]}{'...' if len(X_train.iloc[i]) > 80 else ''}")
            
            # Train the model
            print("\nTraining model...")
            self.pipeline.fit(X_train, y_train)
            
            # Make predictions
            print("\nEvaluating model...")
            y_pred = self.pipeline.predict(X_test)
            y_pred_proba = self.pipeline.predict_proba(X_test)
            
            # Print classification report
            print("\n" + "-"*50)
            print("MODEL EVALUATION")
            print("-"*50)
            
            # Calculate and print accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            
            # Print classification report
            report = classification_report(y_test, y_pred, 
                                         target_names=['Non Emotional', 'Emotional'],
                                         digits=4)
            print("\nClassification Report:")
            print(report)
            
            # Print some example predictions
            print("\nExample predictions:")
            for i in range(min(5, len(X_test))):
                text = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
                true_label = 'Emotional' if (y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]) == 1 else 'Non Emotional'
                pred_label = 'Emotional' if y_pred[i] == 1 else 'Non Emotional'
                confidence = max(y_pred_proba[i]) * 100
                
                print(f"\nText: {text[:100]}...")
                print(f"True: {true_label}, Predicted: {pred_label} ({confidence:.1f}% confidence)")
            
            # Save the model
            model_path = os.path.join(os.path.dirname(__file__), 'emotion_model_lr.joblib')
            joblib.dump(self.pipeline, model_path)
            print(f"\nModel saved to {model_path}")
            
            self.is_trained = True
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*50 + "\n")
            
            return accuracy, report
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            print("\n" + "!"*50)
            print("ERROR DURING TRAINING")
            print("!"*50)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return 0.0, error_msg

    def predict_emotion(self, text):
        if not hasattr(self, 'is_trained') or not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
            
        try:
            if not text or not isinstance(text, str):
                return "Error: Invalid input", 0.0
                
            # Clean the input text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text.strip():
                return "Error: No valid text after cleaning", 0.0
            
            # Debug print
            print(f"\nPredicting emotion for text: {text}")
            print(f"Cleaned text: {cleaned_text}")
            
            # Make prediction
            try:
                prediction = self.pipeline.predict([cleaned_text])
                probas = self.pipeline.predict_proba([cleaned_text])[0]
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return f"Error in prediction: {str(e)}", 0.0
            
            # Get the predicted class
            emotion = "Emotional" if prediction[0] == 1 else "Non Emotional"
            confidence = probas[1] if prediction[0] == 1 else probas[0]
            
            # Debug info
            print(f"Predicted class: {prediction[0]} ({emotion})")
            print(f"Class probabilities: {probas}")
            
            return emotion, confidence
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, 0.0

class EmotionDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.detector = EmotionDetectorLR()
        
        self.title("Emotion Detection (Logistic Regression)")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(
            self.container, 
            text="Emotion Detection using Logistic Regression",
            font=("Helvetica", 20, "bold")
        )
        self.title_label.pack(pady=10)
        
        self.train_button = ctk.CTkButton(
            self.container,
            text="Train Model",
            command=self.train_model,
            height=40,
            font=("Helvetica", 14)
        )
        self.train_button.pack(pady=10)
        
        self.input_frame = ctk.CTkFrame(self.container)
        self.input_frame.pack(fill=tk.X, pady=10)
        
        self.input_label = ctk.CTkLabel(
            self.input_frame, 
            text="Enter your text:",
            font=("Helvetica", 14)
        )
        self.input_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_input = ctk.CTkTextbox(
            self.input_frame,
            height=100,
            font=("Helvetica", 12)
        )
        self.text_input.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_button = ctk.CTkButton(
            self.input_frame,
            text="Analyze Emotion",
            command=self.analyze_text,
            state=tk.DISABLED,
            height=35,
            font=("Helvetica", 14)
        )
        self.analyze_button.pack(pady=5)
        
        self.result_frame = ctk.CTkFrame(self.container)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Result will appear here",
            font=("Helvetica", 14, "bold")
        )
        self.result_label.pack(pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=("Helvetica", 12)
        )
        self.confidence_label.pack(pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ctk.CTkLabel(
            self.container,
            textvariable=self.status_var,
            height=20,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        self.after(100, self.initialize_model)
    
    def initialize_model(self):
        self.status_var.set("Initializing model...")
        self.update_idletasks()
        
        try:
            accuracy, _ = self.detector.train("../dataset")
            self.analyze_button.configure(state=tk.NORMAL)
            self.status_var.set(f"Model trained successfully! Training Accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            self.status_var.set(f"Error initializing model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
    
    def train_model(self):
        self.status_var.set("Training model...")
        self.analyze_button.configure(state=tk.DISABLED)
        self.update_idletasks()
        
        try:
            accuracy, report = self.detector.train("../dataset")
            self.analyze_button.configure(state=tk.NORMAL)
            self.status_var.set(f"Model trained successfully! Training Accuracy: {accuracy*100:.2f}%")
            
            messagebox.showinfo(
                "Training Complete",
                f"Model training completed!\n\n"
                f"Training Accuracy: {accuracy*100:.2f}%\n\n"
                f"Classification Report:\n{report}"
            )
        except Exception as e:
            self.status_var.set(f"Error during training: {str(e)}")
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
    
    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
        
        try:
            self.result_label.configure(text="Analyzing...")
            self.confidence_label.configure(text="")
            self.update_idletasks()
            
            prediction, confidence = self.detector.predict_emotion(text)
            
            self.result_label.configure(
                text=f"Predicted Emotion: {prediction}",
                text_color="green" if prediction == "Emotional" else "red"
            )
            self.confidence_label.configure(
                text=f"Confidence: {confidence:.2f}%"
            )
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set(f"Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"Failed to analyze text: {str(e)}")

if __name__ == "__main__":
    app = EmotionDetectorApp()
    app.mainloop()
