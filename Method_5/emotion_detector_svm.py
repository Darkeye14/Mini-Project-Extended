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
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())

class EmotionDetectorSVM:
    def __init__(self):
        custom_stop_words = set(stopwords.words('english'))
        custom_stop_words -= {'not', 'no', 'never', 'very', 'too', 'so', 'much', 'many', 'more', 'most', 'such', 'only', 'own'}
        cricket_terms = {
            'kohli', 'rohit', 'rahul', 'pandya', 'bumrah', 'bhuvi', 'chahal', 'dhoni', 'dravid', 'sachin', 'ganguly',
            'virat', 'sharma', 'iyer', 'surya', 'sky', 'jadeja', 'ashwin', 'pant', 'kishan', 'shami', 'siraj', 'kuldeep',
            'ipl', 't20', 'odi', 'test', 'worldcup', 'world cup', 'six', 'four', 'wicket', 'century', 'fifty', 'duck',
            'boundary', 'catch', 'stumping', 'runout', 'yorker', 'bouncer', 'googly', 'doosra', 'carrom', 'sweep', 'reverse',
            'champions', 'trophy', 'toss', 'innings', 'powerplay', 'death over', 'super over', 'man of the match'
        }
        custom_stop_words = custom_stop_words - cricket_terms
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words=list(custom_stop_words),
                min_df=3,
                max_df=0.7,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                norm='l2'
            )),
            ('classifier', SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])
        self.is_trained = False

    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+|@\S+', '', text)
        contractions = {"won't": "will not", "can't": "can not", "n't": " not",
                      "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
                      "'t": " not", "'ve": " have", "'m": " am", "i'm": "i am"}
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        try:
            all_files = [f for f in os.listdir(dataset_path) if f.startswith('cricket_emotion_generated_') and f.endswith('.txt')]
            all_files.sort()  # Sort files to ensure consistent ordering
            
            for filename in all_files:
                file_path = os.path.join(dataset_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Split by 'Input:' to separate each example
                    examples = [e.strip() for e in content.split('Input:') if e.strip()]
                    
                    for example in examples:
                        # Extract tag and text
                        tag_line = ''
                        text_lines = []
                        
                        for line in example.split('\n'):
                            line = line.strip()
                            if line.startswith('Tag:'):
                                tag_line = line
                            elif line:  # Only add non-empty lines that aren't tags
                                text_lines.append(line)
                        
                        if not tag_line:
                            continue
                            
                        # Extract tag value
                        tag_match = re.search(r'Tag\s*:\s*(.*?)(?:\n|$)', tag_line, re.IGNORECASE)
                        if not tag_match:
                            continue
                            
                        tag = tag_match.group(1).strip().lower()
                        text = ' '.join(text_lines).strip()
                        
                        # Determine label
                        if 'emotional' in tag and 'non' not in tag and 'not' not in tag:
                            label = 1  # Emotional
                        elif 'non' in tag and 'emotional' in tag:
                            label = 0  # Non Emotional
                        else:
                            continue  # Skip if tag is not in expected format
                            
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            texts.append(cleaned_text)
                            labels.append(label)
                            
            if not texts:
                raise ValueError("No valid examples found in the dataset")
                
            # Verify we have both classes
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"Warning: Only found labels: {unique_labels} in the dataset.")
                print("Using fallback dataset instead.")
                return self._create_fallback_dataset()
                
            return pd.DataFrame({'text': texts, 'label': labels})
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self._create_fallback_dataset()

    def _create_fallback_dataset(self):
        fallback_data = {
            'text': [
                'What an amazing catch by the fielder!',
                'The batsman defended the ball to the off side.',
                'Incredible six over long on!',
                'The bowler delivered a good length ball.',
                'The team collapsed under pressure.',
                'The fielder stopped the boundary with a dive.',
                'The match ended in a thrilling finish!',
                'The batsman took a single to retain strike.'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        return pd.DataFrame(fallback_data)

    def train(self, dataset_path):
        try:
            print("Loading dataset...")
            df = self.load_dataset(dataset_path)
            if df is None or len(df) == 0:
                return 0.0, "No data to train on"
            
            print(f"Loaded {len(df)} examples from the dataset")
            print(f"Class distribution:\n{df['label'].value_counts()}")
            
            X = df['text']
            y = df['label']
            
            if len(set(y)) < 2:
                return 0.0, "Need both classes for training"
            
            print("Splitting dataset into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("Training model...")
            self.pipeline.fit(X_train, y_train)
            
            print("Evaluating model...")
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Non Emotional', 'Emotional'])
            
            print(f"Training complete. Accuracy: {accuracy*100:.2f}%")
            print("Classification Report:")
            print(report)
            
            model_path = os.path.join(os.path.dirname(__file__), 'emotion_model_svm.joblib')
            joblib.dump(self.pipeline, model_path)
            self.is_trained = True
            
            return accuracy, report
            
        except Exception as e:
            import traceback
            error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return 0.0, error_msg

    def predict_emotion(self, text):
        if not self.is_trained:
            model_path = os.path.join(os.path.dirname(__file__), 'emotion_model_svm.joblib')
            if os.path.exists(model_path):
                self.pipeline = joblib.load(model_path)
                self.is_trained = True
            else:
                return "Model not trained yet", 0.0
        
        try:
            if not text or not isinstance(text, str):
                return "Invalid input", 0.0
                
            cleaned_text = self.clean_text(text)
            if not cleaned_text.strip():
                return "No valid text after cleaning", 0.0
                
            probas = self.pipeline.predict_proba([cleaned_text])[0]
            prediction = self.pipeline.predict([cleaned_text])[0]
            confidence = max(probas)
            
            return "Emotional" if prediction == 1 else "Non Emotional", confidence
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", 0.0

class EmotionDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.detector = EmotionDetectorSVM()
        self.title("Emotion Detection (SVM)")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(
            self.container, 
            text="Cricket Emotion Detection using SVM",
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
            text="Enter cricket-related text:",
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
        self.status_var.set("Ready. Click 'Train Model' to begin.")
        
        self.status_bar = ctk.CTkLabel(
            self.container,
            textvariable=self.status_var,
            height=20,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def train_model(self):
        self.status_var.set("Training model...")
        self.update()
        
        try:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
            print(f"Looking for dataset in: {dataset_path}")
            
            # Check if dataset directory exists
            if not os.path.exists(dataset_path):
                error_msg = f"Dataset directory not found at: {dataset_path}"
                print(error_msg)
                self.status_var.set("Error: Dataset directory not found")
                messagebox.showerror("Error", error_msg)
                return
                
            # Check if there are any dataset files
            dataset_files = [f for f in os.listdir(dataset_path) 
                           if f.startswith('cricket_emotion_generated_') and f.endswith('.txt')]
            
            if not dataset_files:
                error_msg = f"No dataset files found in: {dataset_path}"
                print(error_msg)
                self.status_var.set("Error: No dataset files found")
                messagebox.showerror("Error", error_msg)
                return
                
            print(f"Found {len(dataset_files)} dataset files")
            
            # Train the model
            accuracy, report = self.detector.train(dataset_path)
            
            if accuracy > 0:
                self.analyze_button.configure(state=tk.NORMAL)
                status_msg = f"Model trained successfully! Accuracy: {accuracy*100:.2f}%"
                self.status_var.set(status_msg)
                print(status_msg)
                messagebox.showinfo(
                    "Training Complete",
                    f"Model trained successfully!\n\n"
                    f"Accuracy: {accuracy*100:.2f}%\n\n"
                    f"Classification Report:\n{report}"
                )
            else:
                error_msg = f"Training failed: {report}"
                print(error_msg)
                self.status_var.set("Training failed. Using default model.")
                messagebox.showerror("Training Failed", error_msg)
                
        except Exception as e:
            import traceback
            error_msg = f"Error during training: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.status_var.set("Error during training")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}\n\nCheck console for details.")
    
    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
            
        self.status_var.set("Analyzing text...")
        self.update()
        
        try:
            emotion, confidence = self.detector.predict_emotion(text)
            self.result_label.configure(text=f"Prediction: {emotion}")
            self.confidence_label.configure(text=f"Confidence: {confidence*100:.1f}%")
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set("Error during analysis")
            messagebox.showerror("Error", f"Failed to analyze text: {str(e)}")

if __name__ == "__main__":
    try:
        nltk.download('stopwords', quiet=True)
        app = EmotionDetectorApp()
        app.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        raise
