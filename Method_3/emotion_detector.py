import os
import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

class EmotionDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=stopwords.words('english')
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        
        for filename in os.listdir(dataset_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    examples = content.split('\nInput: ')[1:]
                    for example in examples:
                        if 'Tag: ' in example:
                            text, tag = example.split('Tag: ')
                            texts.append(self.clean_text(text.strip()))
                            labels.append(tag.strip())
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })

    def train(self, dataset_path):
        df = self.load_dataset(dataset_path)
        
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']
        
        self.model.fit(X, y)
        self.is_trained = True
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        
        
        return accuracy, report
    
    def predict_emotion(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
        
        cleaned_text = self.clean_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])
        
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        confidence = max(probabilities) * 100
        
        return prediction, confidence

class EmotionDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.detector = EmotionDetector()
        
        self.title("Emotion Detection Tool")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(
            self.container, 
            text="Emotion Detection using Random Forest",
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
