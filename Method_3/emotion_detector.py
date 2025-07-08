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
        # Enhanced TF-IDF vectorizer to better capture emotional content
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size to top 10,000 features
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            stop_words=stopwords.words('english'),
            min_df=2,            # Minimum document frequency
            max_df=0.85,         # Maximum document frequency (to filter out too common terms)
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[^\d\W]+\b'  # Better token pattern
        )
        
        # Enhanced Random Forest with better parameters for text classification
        self.model = RandomForestClassifier(
            n_estimators=300,           # Increased number of trees
            max_depth=30,                # Increased max depth
            min_samples_split=3,         # Reduced to capture more patterns
            min_samples_leaf=2,          # Minimum samples required at each leaf
            max_features='sqrt',         # Number of features to consider at each split
            bootstrap=True,              # Bootstrap samples when building trees
            oob_score=True,              # Use out-of-bag samples for validation
            random_state=42,             # For reproducibility
            class_weight='balanced',     # Handle class imbalance
            n_jobs=-1,                   # Use all CPU cores
            verbose=1                    # Show training progress
        )
        self.is_trained = False

    def clean_text(self, text):
        # 1. Convert to string and lowercase
        text = str(text).lower().strip()
        
        # 2. Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 3. Remove user @ mentions and hashtags
        text = re.sub(r'(@\w+|#\w+)', '', text)
        
        # 4. Keep emotional punctuation but pad with spaces
        text = re.sub(r'([!?]+)', r' \1 ', text)  # Keep multiple ! or ? as they indicate emphasis
        
        # 5. Handle common contractions and special cases
        text = re.sub(r"won't", 'will not', text)
        text = re.sub(r"can'?t", 'can not', text)
        text = re.sub(r"n't", ' not', text)
        text = re.sub(r"'re", ' are', text)
        text = re.sub(r"'s", ' is', text)
        text = re.sub(r"'d", ' would', text)
        text = re.sub(r"'ll", ' will', text)
        text = re.sub(r"'t", ' not', text)
        text = re.sub(r"'ve", ' have', text)
        text = re.sub(r"'m", ' am', text)
        
        # 6. Remove special characters but keep basic punctuation that might indicate emotion
        text = re.sub(r'[^a-z0-9\s.?!]', ' ', text)
        
        # 7. Handle elongated words (e.g., 'soooo' -> 'so')
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # 8. Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        return text

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        print("\n--- Loading Dataset ---")
        total_examples_found = 0
        total_emotional = 0
        total_non_emotional = 0
        file_count = 0

        # Regex to find all 'Input: ... Tag: ...' pairs
        pattern = re.compile(r"Input:\s*(.*?)\s*Tag:\s*(.*?)(?=\nInput:|$)", re.DOTALL)

        for filename in os.listdir(dataset_path):
            if filename.endswith('.txt'):
                file_count += 1
                file_path = os.path.join(dataset_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        matches = pattern.findall(content)
                        
                        if not matches:
                            print(f"Warning: No examples found in '{filename}'")
                            continue
                            
                        file_emotional = 0
                        file_non_emotional = 0
                        
                        for text, tag in matches:
                            text = text.strip()
                            tag = tag.strip()
                            
                            # Normalize tag format
                            if tag.lower() in ['non emotional', 'non-emotional']:
                                tag = 'Non-Emotional'
                            elif tag.lower() == 'emotional':
                                tag = 'Emotional'
                            
                            if text and tag in ['Emotional', 'Non-Emotional']:
                                texts.append(self.clean_text(text))
                                labels.append(tag)
                                
                                if tag == 'Emotional':
                                    file_emotional += 1
                                    total_emotional += 1
                                else:
                                    file_non_emotional += 1
                                    total_non_emotional += 1
                                
                                # Print first 3 examples from first file for verification
                                if file_count == 1 and len(texts) <= 3:
                                    print(f"  Example {len(texts)} - Tag: {tag}")
                                    print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                        
                        print(f"Processed '{filename}': {len(matches)} examples ({file_emotional} Emotional, {file_non_emotional} Non-Emotional)")
                        total_examples_found += len(matches)
                        
                except Exception as e:
                    print(f"Error processing file '{filename}': {str(e)}")
        
        print("\n--- Dataset Summary ---")
        print(f"Total files processed: {file_count}")
        print(f"Total examples found: {total_examples_found}")
        print(f"Total examples loaded: {len(texts)}")
        print(f"  - Emotional: {total_emotional} ({total_emotional/len(texts)*100:.1f}%)")
        print(f"  - Non-Emotional: {total_non_emotional} ({total_non_emotional/len(texts)*100:.1f}%)")
        print("-------------------------------------")
        
        if total_emotional == 0 or total_non_emotional == 0:
            print("WARNING: Dataset is imbalanced - one of the classes has no examples!")
            print("This will cause training to fail. Please check your dataset files.")
            print("-------------------------------------")
            
            # Print sample tags found for debugging
            sample_tags = set()
            for filename in os.listdir(dataset_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tags = re.findall(r'Tag:\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
                        sample_tags.update(t.strip() for t in tags if t.strip())
                        if len(sample_tags) > 5:  # Don't collect too many samples
                            break
            
            print(f"Sample tags found in dataset: {', '.join(f'"{t}"' for t in list(sample_tags)[:10])}")
            print("-------------------------------------")
        
        if not texts:
            raise ValueError("Dataset is empty or format is incorrect. No examples were loaded.")

        return pd.DataFrame({
            'text': texts,
            'label': labels
        })

    def train(self, dataset_path):
        # Load and prepare dataset
        dataset = self.load_dataset(dataset_path)
        if dataset.empty:
            raise ValueError("The loaded dataset is empty. Cannot train the model.")

        # Label Encoding
        dataset['label_encoded'] = dataset['label'].apply(lambda x: 1 if x == 'Emotional' else 0)
        
        # Display class distribution
        class_dist = dataset['label'].value_counts()
        print("\n--- Class Distribution ---")
        print(class_dist)
        print("--------------------------\n")
        
        # Handle severe class imbalance if needed
        if min(class_dist) / max(class_dist) < 0.2:
            print("Warning: Significant class imbalance detected. Consider collecting more balanced data.")
        
        X = dataset['text']
        y = dataset['label_encoded']

        # Check for class imbalance before splitting
        if len(y.unique()) < 2:
            raise ValueError(
                f"The dataset contains only one class ('{dataset['label'].iloc[0]}'). "
                f"The model requires examples of both 'Emotional' and 'Non-Emotional' tags to train."
            )
        
        # Split data into train and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        print("Fitting vectorizer...")
        # Fit the vectorizer on training data only
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Get feature names for debugging
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"Number of features: {len(feature_names)}")
        
        # Train the model with verbose output
        print("\nTraining model...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Print out-of-bag score
        if hasattr(self.model, 'oob_score_'):
            print(f"\nOut-of-bag score: {self.model.oob_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)[:, 1]  # Probabilities for positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=['Non-Emotional', 'Emotional'],
            zero_division=0,
            output_dict=True
        )
        
        # Print detailed report
        print("\n--- Model Evaluation ---")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                 target_names=['Non-Emotional', 'Emotional'],
                                 zero_division=0))
        
        # Print some example predictions for debugging
        print("\n--- Example Predictions ---")
        test_examples = X_test.sample(min(5, len(X_test)), random_state=42)
        for i, example in enumerate(test_examples):
            true_label = y_test[X_test == example].iloc[0]
            pred_label = y_pred[X_test == example][0]
            pred_prob = y_pred_proba[X_test == example][0]
            print(f"\nExample {i+1}:")
            print(f"Text: {example[:150]}...")
            print(f"True: {'Emotional' if true_label == 1 else 'Non-Emotional'}")
            print(f"Pred: {'Emotional' if pred_label == 1 else 'Non-Emotional'} ({pred_prob*100:.1f}%)")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n--- Top 20 Important Features ---")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            print("\nMost important features for Emotional class:")
            for i in indices:
                if importances[i] > 0.001:  # Only show features with some importance
                    print(f"{feature_names[i]}: {importances[i]:.4f}")
        
        return accuracy, str(report)
    
    def predict_emotion(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
        
        cleaned_text = self.clean_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Get prediction and probabilities directly from the model
        prediction_idx = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # The label is based on the model's prediction
        label = "Emotional" if prediction_idx == 1 else "Non-Emotional"
        # The confidence is the probability of that predicted class
        confidence = probabilities[prediction_idx] * 100
        
        # Print debug info
        print(f"\nPrediction for: {text}")
        print(f"Cleaned text: {cleaned_text}")
        print(f"Predicted: {label} (Confidence: {confidence:.2f}%)")
        print(f"Probabilities: [Non-Emotional: {probabilities[0]*100:.2f}%, Emotional: {probabilities[1]*100:.2f}%]")
        
        return label, confidence

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
        
        self.train_model_button = ctk.CTkButton(
            self.container,
            text="Train Model",
            command=self.train_model,
            height=40,
            font=("Helvetica", 14)
        )
        self.train_model_button.pack(pady=10)
        
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
            state=tk.DISABLED,  # Start disabled until model is trained
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
            self,
            textvariable=self.status_var,
            height=20,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def train_model(self):
        self.status_var.set("Training model... Please wait.")
        self.update_idletasks()  # Force GUI to update
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
            accuracy, report = self.detector.train(dataset_path)
            
            self.analyze_button.configure(state=tk.NORMAL)  # Enable analyze button
            self.status_var.set(f"Model trained. Test Accuracy: {accuracy*100:.2f}%")
            
            messagebox.showinfo(
                "Training Complete",
                f"Model training is complete.\n\nAccuracy: {accuracy*100:.2f}%\n\nClassification Report:\n{report}"
            )
        except Exception as e:
            self.status_var.set(f"Error during training: {str(e)}")
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")

    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return

        self.status_var.set("Analyzing...")
        self.result_label.configure(text="", text_color="black")
        self.confidence_label.configure(text="")
        self.update_idletasks()

        try:
            label, confidence = self.detector.predict_emotion(text)
            color = "green" if label == "Emotional" else "red"
            self.result_label.configure(text=f"Result: {label}", text_color=color)
            self.confidence_label.configure(text=f"Confidence: {confidence:.2f}%")
            self.status_var.set("Analysis complete. Ready for next input.")
        except Exception as e:
            self.result_label.configure(text="Error during prediction", text_color="red")
            self.status_var.set("Analysis failed.")
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = EmotionDetectorApp()
    app.mainloop()
