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
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words=stopwords.words('english')
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        self.is_trained = False

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)  # Keep only letters and whitespace
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text)
        return ' '.join(tokens)

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        
        try:
            # First, try to load from the provided dataset
            for filename in os.listdir(dataset_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        examples = content.split('\nInput: ')[1:]
                        for example in examples:
                            if 'Tag: ' in example:
                                try:
                                    text_part, tag_part = example.split('Tag: ', 1)
                                    text = text_part.strip()
                                    tag = tag_part.split('\n')[0].strip()
                                    if text and tag in ['Emotional', 'Non-Emotional']:
                                        texts.append(self.clean_text(text))
                                        labels.append(1 if tag == 'Emotional' else 0)
                                except Exception as e:
                                    print(f"Error processing example in {filename}: {e}")
                                    continue
            
            # If we have both classes, return the loaded dataset
            if len(set(labels)) >= 2 and len(texts) >= 2:
                return pd.DataFrame({
                    'text': texts,
                    'label': labels
                })
            
            print("Warning: Insufficient data or only one class found in the dataset. Using fallback dataset.")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
        
        # Fallback to a balanced dummy dataset
        return pd.DataFrame({
            'text': [
                'I am so happy today!', 
                'This is a neutral statement.',
                'I feel really excited about this!',
                'The weather is normal today.',
                'I am feeling absolutely wonderful!',
                'This is a regular day.',
                'I am overjoyed with the results!',
                'The meeting was quite ordinary.'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0]  # Alternating emotional and non-emotional
        })

    def train(self, dataset_path):
        try:
            df = self.load_dataset(dataset_path)
            
            # Ensure we have at least 2 classes and enough samples
            if len(df['label'].unique()) < 2:
                raise ValueError("Dataset must contain at least 2 classes (Emotional and Non-Emotional)")
                
            if len(df) < 2:
                raise ValueError("Dataset must contain at least 2 samples")
            
            X = df['text']
            y = df['label']
            
            # If we have very few samples, use all for training
            if len(df) <= 10:
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True
            
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Non-Emotional', 'Emotional'])
            
            print("\nTraining completed successfully!")
            print(f"Dataset size: {len(df)} samples")
            print(f"Classes: {dict(df['label'].value_counts())}")
            print(f"Accuracy: {accuracy:.2f}")
            
            return accuracy, report
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Return default values in case of error
            return 0.0, "Training failed - " + str(e)
    
    def predict_emotion(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
        
        cleaned_text = self.clean_text(text)
        proba = self.pipeline.predict_proba([cleaned_text])[0]
        prediction = self.pipeline.predict([cleaned_text])[0]
        confidence = max(proba) * 100
        
        return "Emotional" if prediction == 1 else "Non-Emotional", confidence

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
