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
        cricket_keep_words = [
            'six', 'four', 'wicket', 'century', 'fifty', 'hundred', 'boundary',
            'dismissal', 'catch', 'runout', 'stumping', 'appeal', 'review', 'drs',
            'powerplay', 'death', 'overs', 'innings', 'partnership', 'chase', 'target',
            'ipl', 'worldcup', 't20', 'odi', 'test', 'nidahas', 'asia', 'trophy'
        ]
        stop_words = [word for word in stopwords.words('english') if word not in cricket_keep_words]
        self.cricket_terms = [
            'six', 'four', 'wicket', 'century', 'fifty', 'hundred', 'duck',
            'maiden', 'hattrick', 'noball', 'wide', 'lbw', 'bouncer', 'yorker',
            'googly', 'doosra', 'slog', 'sweep', 'drive', 'pull', 'hook', 'cut',
            'cover', 'midwicket', 'longon', 'fineleg', 'thirdman', 'gully', 'slip',
            'powerplay', 'deathovers', 'superover', 'drs', 'umpire', 'referral'
        ]
        self.indian_players = [
            'kohli', 'rohit', 'rahul', 'pandya', 'bumrah', 'bhuvi', 'chahal',
            'dhoni', 'dravid', 'sachin', 'ganguly', 'dada', 'dhonis', 'kohlis',
            'rohit', 'rohit', 'rahul', 'pant', 'iyer', 'ishan', 'surya', 'sky',
            'bumrah', 'shami', 'siraj', 'jadeja', 'ashwin', 'axar', 'chahal', 'kuldeep'
        ]
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 3),
            stop_words=stop_words,
            min_df=1,
            max_df=0.85,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=50,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            verbose=1,
            max_features='sqrt',
            min_impurity_decrease=0.0001
        )
        self.is_trained = False

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'([!?]+)', r' \1 ', text)
        text = self._standardize_cricket_terms(text)
        emotional_phrases = [
            (r'\b(stunn(ing|ed)?|amaz(ing|ed)|wow|incredible|unbelievable|fantastic|brilliant|superb|outstanding|exceptional|extraordinary|phenomenal|mind.?blow(ing)?|speechless|awe(some)?|thrill(ing|ed)|excit(ing|ed))\b', ' _emph_ '),
            (r'\b(goosebumps|chill(s|ed)|shiver(ed|ing)|tear(s|ing)|cry(ing)?|emotional|overwhelm(ed|ing))\b', ' _emot_ '),
            (r'\b(love|lovely|adore|adored|passion(ate)?|heart|soul|spirit|pride|proud|honor|honour|respect|admire|cherish|treasure|value|appreciate|enjoy|delight(ed|ful)|pleasur(e|able)|happy|happiness|joy(ful)?|ecstatic|bliss(ful)?|elat(ed|ing)|jubilant|overjoyed|thrilled|euphoric|rapturous|gleeful|jovial|jolly|merry|cheer(ful|y)|glad|content(ed)?|satisf(y|ied)|gratif(y|ied)|fulfill(ed|ing)|triumph(ant)?|victor(y|ious)|win(ning)?|succeed|success(ful)?|achieve(ment)?|accomplish(ed|ment)?|excel(lent|led)?|superior|best|great(est)?|wonder(ful|ous)|marvel(ous|lous)|splendid|glorious|magnificent|majestic|sublime|divine|heavenly|perfect|flawless|impeccable|exquisite|elegant|beautiful|gorgeous|stunning|breathtaking|mesmeriz(ing|ed)|captivat(ing|ed)|enchant(ing|ed)|charm(ing|ed)|allur(ing|ed)|fascinat(ing|ed)|enthrall(ing|ed)|spellbind(ing|ed)|hypnotiz(ing|ed)|transfix(ed|ing))\b', ' _pos_emot_ '),
            (r'\b(hate|hatred|disgust(ing|ed)|repulsive|revolting|sickening|nauseating|vile|loathsome|abhorrent|abominable|detestable|despicable|contemptible|disgraceful|shameful|deplorable|lamentable|regrettable|unfortunate|sad|sadden(ed|ing)|unhappy|miserable|wretched|sorrow(ful)?|grief|griev(e|ing)|heartbroken|heartbreak(ing)?|anguish(ed)?|distress(ed|ing)|despair(ing)?|hopeless|desperate|devastat(ing|ed)|crush(ed|ing)|destroy(ed)?|ruin(ed)?|wreck(ed)?|demolish(ed)?|annihilat(ed|ing)|obliterat(ed|ing)|eradicate(d|ing)|eliminate(d|ing)|remove(d)?|erase(d)?|delete(d)?|destroy(ed)?|kill(ed|ing)?|murder(ed|ing)?|slay(ing|n)?|assassinat(ed|ing)|execute(d|ing)|hang(ed|ing)|lynch(ed|ing)|torture(d|ing)|torment(ed|ing)|abuse(d|ing)|mistreat(ed|ing)|maltreat(ed|ing)|ill.treat(ed|ing)|harm(ed|ing)|hurt(ing)?|injure(d|ing)|wound(ed|ing)|damage(d|ing)|ruin(ed|ing)|wreck(ed|ing)|destroy(ed|ing)|demolish(ed|ing)|annihilat(ed|ing)|obliterat(ed|ing)|eradicate(d|ing)|eliminate(d|ing)|remove(d)?|erase(d)?|delete(d)?|kill(ed|ing)?|murder(ed|ing)?|slay(ing|n)?|assassinat(ed|ing)|execute(d|ing)|hang(ed|ing)|lynch(ed|ing)|torture(d|ing)|torment(ed|ing)|abuse(d|ing)|mistreat(ed|ing)|maltreat(ed|ing)|ill.treat(ed|ing)|harm(ed|ing)|hurt(ing)?|injure(d|ing)|wound(ed|ing)|damage(d|ing))\b', ' _neg_emot_ ')
        ]
        for pattern, replacement in emotional_phrases:
            text = re.sub(pattern, f' {replacement} ', text, flags=re.IGNORECASE)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\b6\b', ' six ', text)
        text = re.sub(r'\b4\b', ' four ', text)
        text = re.sub(r'\b1\b', ' one ', text)
        text = re.sub(r'([!?]+)', r' \1 ', text)
        contractions = {
            r"won't": 'will not',
            r"can'?t": 'can not',
            r"n't": ' not',
            r"'re": ' are',
            r"'s": ' is',
            r"'d": ' would',
            r"'ll": ' will',
            r"'t": ' not',
            r"'ve": ' have',
            r"'m": ' am',
            r'\bwk(t)?\b': 'wicket',
            r'\b4s\b': 'fours',
            r'\b6s\b': 'sixes',
            r'\bvs\b': 'versus',
            r'\bvs\.': 'versus',
            r'\bno\s*\d+': 'number',
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        def reduce_elongated(word):
            if (word in self.cricket_terms or 
                word in self.indian_players or
                any(term in word for term in self.cricket_terms) or
                any(player in word for player in self.indian_players if len(player) > 3)):
                return word
            return re.sub(r'(\w)(\1{2,})', r'\1\1\1', word)
        words = text.split()
        words = [reduce_elongated(word) for word in words]
        for i, word in enumerate(words):
            if re.search(r'(\w)\1{2,}', word):
                words[i] = f"{word} _emph_"
            if any(term in word for term in ['six', 'four', 'wicket', 'out', 'catch', 'stump']):
                words[i] = f"{word} _cricket_action_"
            if any(player in word for player in self.indian_players if len(player) > 3):
                words[i] = f"{word} _player_"
        text = ' '.join(words)
        text = re.sub(r'[^a-z0-9\s.?!]', ' ', text)
        text = ' '.join(text.split())
        return text

    def _standardize_cricket_terms(self, text):
        player_mapping = {
            'vk': 'virat kohli',
            'king kohli': 'virat kohli',
            'kingkohli': 'virat kohli',
            'rohit sharma': 'rohit sharma',
            'hitman': 'rohit sharma',
            'hit man': 'rohit sharma',
            'msd': 'ms dhoni',
            'thala': 'ms dhoni',
            'captain cool': 'ms dhoni',
            'jaddu': 'ravindra jadeja',
            'sir jadeja': 'ravindra jadeja',
            'boom boom': 'jasprit bumrah',
            'bumrah': 'jasprit bumrah',
            'bhuvi': 'bhuvneshwar kumar',
            'chiku': 'virat kohli',
            'cheeku': 'virat kohli',
        }
        cricket_terms = {
            't20i': 't20',
            't20i': 't20',
            'odi': 'one day',
            'test match': 'test',
            'test matches': 'test',
            'world cup': 'worldcup',
            'worldcup': 'worldcup',
            'ipl': 'indian premier league',
            'iplt20': 'indian premier league',
            't20 worldcup': 't20 worldcup',
            't20wc': 't20 worldcup',
            'wt20': 't20 worldcup',
            'icc': 'international cricket council',
            'bcci': 'board of control for cricket in india',
            'nca': 'national cricket academy',
            'ranji': 'ranji trophy',
            'vijay hazare': 'vijay hazare trophy',
            'sherf': 'sherf rutherford'
        }
        for term, replacement in {**player_mapping, **cricket_terms}.items():
            text = re.sub(r'\b' + re.escape(term) + r'\b', replacement, text, flags=re.IGNORECASE)
        return text

    def load_dataset(self, dataset_path):
        texts = []
        labels = []
        print("\n--- Loading Dataset ---")
        total_examples_found = 0
        total_emotional = 0
        total_non_emotional = 0
        file_count = 0
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
            sample_tags = set()
            for filename in os.listdir(dataset_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tags = re.findall(r'Tag:\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
                        sample_tags.update(t.strip() for t in tags if t.strip())
                        if len(sample_tags) > 5:
                            break
            print(f"Sample tags found in dataset: {', '.join(f'\"{t}\"' for t in list(sample_tags)[:10])}")
            print("-------------------------------------")
        if not texts:
            raise ValueError("Dataset is empty or format is incorrect. No examples were loaded.")
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })

    def train(self, dataset_path):
        dataset = self.load_dataset(dataset_path)
        if dataset.empty:
            raise ValueError("The loaded dataset is empty. Cannot train the model.")
        dataset['label_encoded'] = dataset['label'].apply(lambda x: 1 if x == 'Emotional' else 0)
        class_dist = dataset['label'].value_counts()
        print("\n--- Class Distribution ---")
        print(class_dist)
        print("--------------------------\n")
        if min(class_dist) / max(class_dist) < 0.2:
            print("Warning: Significant class imbalance detected. Consider collecting more balanced data.")
        X = dataset['text']
        y = dataset['label_encoded']
        if len(y.unique()) < 2:
            raise ValueError(
                f"The dataset contains only one class ('{dataset['label'].iloc[0]}'). "
                f"The model requires examples of both 'Emotional' and 'Non-Emotional' tags to train."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        print("Fitting vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"Number of features: {len(feature_names)}")
        print("\nTraining model...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        if hasattr(self.model, 'oob_score_'):
            print(f"\nOut-of-bag score: {self.model.oob_score_:.4f}")
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, 
            target_names=['Non-Emotional', 'Emotional'],
            zero_division=0,
            output_dict=True
        )
        print("\n--- Model Evaluation ---")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                 target_names=['Non-Emotional', 'Emotional'],
                                 zero_division=0))
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
        if hasattr(self.model, 'feature_importances_'):
            print("\n--- Top 20 Important Features ---")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            print("\nMost important features for Emotional class:")
            for i in indices:
                if importances[i] > 0.001:
                    print(f"{feature_names[i]}: {importances[i]:.4f}")
        return accuracy, str(report)
    
    def predict_emotion(self, text):
        if not self.is_trained:
            raise Exception("Model not trained yet. Please train the model first.")
        cleaned_text = self.clean_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])
        prediction_idx = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        label = "Emotional" if prediction_idx == 1 else "Non-Emotional"
        confidence = probabilities[prediction_idx] * 100
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
            self,
            textvariable=self.status_var,
            height=20,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    def train_model(self):
        self.status_var.set("Training model... Please wait.")
        self.update_idletasks()
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
            accuracy, report = self.detector.train(dataset_path)
            self.analyze_button.configure(state=tk.NORMAL)
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
