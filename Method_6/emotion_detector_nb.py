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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.lower())

class EmotionDetectorNB:
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
                min_df=2,
                max_df=0.8,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('classifier', MultinomialNB(alpha=0.1))
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
            for filename in os.listdir(dataset_path):
                if not filename.endswith('.txt'):
                    continue
                with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    examples = re.split(r'\nInput\s*:', content)
                    for example in examples:
                        if not example.strip():
                            continue
                        tag_match = re.search(r'Tag\s*:\s*([^\n]*)', example, re.IGNORECASE)
                        if not tag_match:
                            continue
                        text = example[:tag_match.start()].strip()
                        tag = tag_match.group(1).strip().lower()
                        if 'emotional' in tag:
                            label = 1
                        elif 'non' in tag and 'emotional' in tag:
                            label = 0
                        else:
                            continue
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            texts.append(cleaned_text)
                            labels.append(label)
            if not texts:
                raise ValueError("No valid examples found in the dataset")
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
            df = self.load_dataset(dataset_path)
            if df is None or len(df) == 0:
                return 0.0, "No data to train on"
            
            X = df['text']
            y = df['label']
            
            if len(set(y)) < 2:
                return 0.0, "Need both classes for training"
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Non Emotional', 'Emotional'])
            
            model_path = os.path.join(os.path.dirname(__file__), 'emotion_model_nb.joblib')
            joblib.dump(self.pipeline, model_path)
            self.is_trained = True
            
            return accuracy, report
            
        except Exception as e:
            return 0.0, f"Error during training: {str(e)}"

    def predict_emotion(self, text):
        if not self.is_trained:
            model_path = os.path.join(os.path.dirname(__file__), 'emotion_model_nb.joblib')
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
        self.detector = EmotionDetectorNB()
        self.title("Emotion Detection (Naive Bayes)")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(
            self.container, 
            text="Cricket Emotion Detection using Naive Bayes",
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
            accuracy, report = self.detector.train(dataset_path)
            
            if accuracy > 0:
                self.analyze_button.configure(state=tk.NORMAL)
                self.status_var.set(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                messagebox.showinfo("Training Complete", f"Model trained successfully!\nAccuracy: {accuracy*100:.2f}%")
            else:
                self.status_var.set("Training failed. Using default model.")
                messagebox.showerror("Training Failed", report)
                
        except Exception as e:
            self.status_var.set("Error during training")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
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
