import os
import re
import math
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox, scrolledtext
import customtkinter as ctk

class TfIdfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0
        self.avg_doc_length = 0
        
    def fit(self, documents):
        self.doc_count = len(documents)
        doc_freq = defaultdict(int)
        total_terms = 0
        
        for doc in documents:
            terms = self.tokenize(doc)
            total_terms += len(terms)
            for term in set(terms):
                doc_freq[term] += 1
                
        self.avg_doc_length = total_terms / self.doc_count
        
        for term, count in doc_freq.items():
            self.vocab[term] = len(self.vocab)
            self.idf[term] = math.log((self.doc_count + 1) / (count + 1)) + 1
            
    def transform(self, documents):
        tfidf_vectors = []
        for doc in documents:
            terms = self.tokenize(doc)
            tf = defaultdict(int)
            for term in terms:
                if term in self.vocab:
                    tf[term] += 1
                    
            vector = [0.0] * len(self.vocab)
            for term, count in tf.items():
                if term in self.vocab:
                    tf_val = count / len(terms)
                    tf_idf = tf_val * self.idf[term]
                    vector[self.vocab[term]] = tf_idf
                    
            tfidf_vectors.append(vector)
            
        return np.array(tfidf_vectors)
        
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

class EmotionDetector:
    def __init__(self):
        self.vectorizer = TfIdfVectorizer()
        self.threshold = 0
        self.emotional_terms = set()
        self.non_emotional_terms = set()
        
    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        
        for filename in os.listdir(dataset_path):
            if not filename.startswith('cricket_emotion_generated_') or not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(dataset_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                examples = [e.strip() for e in content.split('Input:') if e.strip()]
                
                for example in examples:
                    if 'Tag:' not in example:
                        continue
                        
                    parts = example.split('Tag:')
                    if len(parts) < 2:
                        continue
                        
                    text = parts[0].strip()
                    tag = parts[1].strip().lower()
                    
                    if not text or not tag:
                        continue
                        
                    if 'emotional' in tag and 'non' not in tag and 'not' not in tag:
                        label = 1
                    elif 'non' in tag and 'emotional' in tag:
                        label = 0
                    else:
                        continue
                        
                    texts.append(text)
                    labels.append(label)
                    
        return texts, labels
        
    def train(self, dataset_path):
        texts, labels = self.load_dataset(dataset_path)
        
        if not texts or len(set(labels)) < 2:
            return False, "Insufficient data for training"
            
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        
        emotional_scores = []
        non_emotional_scores = []
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            terms = set(self.vectorizer.tokenize(text))
            score = 0
            
            for term in terms:
                if term in self.vectorizer.vocab:
                    idx = self.vectorizer.vocab[term]
                    score += X[i][idx]
                    
            if label == 1:
                emotional_scores.append(score)
                self.emotional_terms.update(terms)
            else:
                non_emotional_scores.append(score)
                self.non_emotional_terms.update(terms)
                
        if emotional_scores and non_emotional_scores:
            mean_emotional = sum(emotional_scores) / len(emotional_scores)
            mean_non_emotional = sum(non_emotional_scores) / len(non_emotional_scores)
            self.threshold = (mean_emotional + mean_non_emotional) / 2
            return True, "Training completed successfully"
            
        return False, "Could not calculate threshold"
        
    def predict(self, text):
        terms = set(self.vectorizer.tokenize(text))
        X = self.vectorizer.transform([text])
        score = 0
        
        for term in terms:
            if term in self.vectorizer.vocab:
                idx = self.vectorizer.vocab[term]
                score += X[0][idx]
                
        is_emotional = score > self.threshold
        
        emotional_terms = terms & self.emotional_terms
        non_emotional_terms = terms & self.non_emotional_terms
        
        return {
            'is_emotional': is_emotional,
            'score': score,
            'threshold': self.threshold,
            'emotional_terms': emotional_terms,
            'non_emotional_terms': non_emotional_terms,
            'all_terms': terms
        }

class EmotionDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.detector = EmotionDetector()
        self.title("Cricket Emotion Detector (TF-IDF)")
        self.geometry("1000x700")
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.header = ctk.CTkLabel(
            self,
            text="Cricket Emotion Detection using Custom TF-IDF",
            font=("Arial", 20, "bold")
        )
        self.header.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.train_frame = ctk.CTkFrame(self)
        self.train_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.train_frame.grid_columnconfigure(0, weight=1)
        
        self.train_button = ctk.CTkButton(
            self.train_frame,
            text="Train Model",
            command=self.train_model,
            height=40,
            font=("Arial", 14)
        )
        self.train_button.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_label = ctk.CTkLabel(
            self.input_frame,
            text="Enter cricket-related text:",
            font=("Arial", 14)
        )
        self.input_label.grid(row=0, column=0, padx=10, pady=(10,5), sticky="w")
        
        self.text_input = ctk.CTkTextbox(
            self.input_frame,
            height=150,
            font=("Arial", 12)
        )
        self.text_input.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        self.analyze_button = ctk.CTkButton(
            self.input_frame,
            text="Analyze Emotion",
            command=self.analyze_text,
            state="disabled",
            height=35,
            font=("Arial", 14)
        )
        self.analyze_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Results will appear here",
            font=("Arial", 14, "bold")
        )
        self.result_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.details_text = scrolledtext.ScrolledText(
            self.result_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=10
        )
        self.details_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            height=25,
            corner_radius=0,
            font=("Arial", 10)
        )
        self.status_bar.grid(row=4, column=0, sticky="ew")
        
    def train_model(self):
        self.status_var.set("Training model...")
        self.update()
        
        try:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
            success, message = self.detector.train(dataset_path)
            
            if success:
                self.status_var.set("Model trained successfully")
                self.analyze_button.configure(state="normal")
                messagebox.showinfo("Success", "Model trained successfully!")
            else:
                self.status_var.set("Training failed")
                messagebox.showerror("Error", f"Training failed: {message}")
                
        except Exception as e:
            self.status_var.set("Error during training")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
            
        self.status_var.set("Analyzing text...")
        self.update()
        
        try:
            result = self.detector.predict(text)
            
            emotion = "Emotional" if result['is_emotional'] else "Non Emotional"
            self.result_label.configure(text=f"Prediction: {emotion}")
            
            details = f"TF-IDF Score: {result['score']:.4f}\n"
            details += f"Decision Threshold: {result['threshold']:.4f}\n\n"
            
            details += f"Emotional Terms ({len(result['emotional_terms'])}):\n"
            details += ", ".join(sorted(result['emotional_terms'])) + "\n\n"
            
            details += f"Non-Emotional Terms ({len(result['non_emotional_terms'])}):\n"
            details += ", ".join(sorted(result['non_emotional_terms'])) + "\n\n"
            
            details += f"All Terms ({len(result['all_terms'])}):\n"
            details += ", ".join(sorted(result['all_terms']))
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, details)
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set("Error during analysis")
            messagebox.showerror("Error", f"Failed to analyze text: {str(e)}")

if __name__ == "__main__":
    app = EmotionDetectorApp()
    app.mainloop()
