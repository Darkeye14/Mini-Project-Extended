import os
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext
import customtkinter as ctk

class POSEmotionDetector:
    def __init__(self):
        self.emotional_words = {
            'amazing', 'brilliant', 'fantastic', 'wonderful', 'terrible', 'awful',
            'horrible', 'disappointing', 'frustrating', 'exciting', 'thrilling',
            'nerve-wracking', 'heartbreaking', 'unbelievable', 'sensational',
            'stunning', 'shocking', 'devastating', 'outstanding', 'exceptional',
            'phenomenal', 'remarkable', 'extraordinary', 'incredible', 'unreal',
            'great', 'poor', 'bad', 'good', 'excellent', 'perfect', 'worst', 'best',
            'superb', 'magnificent', 'splendid', 'marvelous', 'terrific', 'awful',
            'dreadful', 'horrifying', 'frightening', 'scary', 'intense', 'dramatic',
            'emotional', 'passionate', 'intense', 'fierce', 'powerful', 'strong',
            'weak', 'pathetic', 'disastrous', 'tragic', 'miraculous', 'magical'
        }
        
        self.emotional_phrases = {
            'what a': 2.0, 'how': 1.5, 'so much': 1.5, 'too much': 1.5, 
            'very': 1.0, 'really': 1.0, 'absolutely': 1.5, 'completely': 1.0,
            'totally': 1.0, 'extremely': 1.5, 'incredibly': 1.5, 'unbelievably': 1.5,
            'filled with': 2.0, 'eyes with': 1.5, 'made me': 1.5, 'i feel': 1.5,
            'i felt': 1.5, 'i love': 2.0, 'i hate': 2.0, 'so happy': 2.0,
            'so sad': 2.0, 'so excited': 2.0, 'so angry': 2.0, 'can\'t believe': 1.5,
            'no way': 1.5, 'oh my': 1.5, 'oh no': 1.5, 'oh wow': 2.0,
            'that\'s amazing': 2.5, 'that\'s incredible': 2.5, 'that\'s awesome': 2.5,
            'that\'s terrible': 2.5, 'that\'s horrible': 2.5, 'that\'s wonderful': 2.5
        }
        
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}
        # Lower threshold to be more sensitive to emotional content in short texts
        self.threshold = 1.5
        self.trained = True
        
    def analyze_text(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        emotional_terms = []
        score = 0.0
        
        # Check for emotional punctuation
        emotional_punctuation = sum(1 for c in text if c in '!?')  # Count ! and ?
        score += emotional_punctuation * 0.5  # Add 0.5 for each emotional punctuation
        
        # Check for all caps words (often indicate strong emotion)
        all_caps_words = sum(1 for w in re.findall(r'\b[A-Z]{2,}\b', text))
        score += all_caps_words * 0.5
        
        # Check for repeated letters (e.g., soooo good)
        repeated_letters = sum(1 for w in words if any(letter * 3 in w for letter in 'aeioulmnrswy'))
        score += repeated_letters * 0.3
        
        # Check for emotional phrases with weights
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            phrase_weight = self.emotional_phrases.get(bigram, 0)
            if phrase_weight > 0:
                emotional_terms.append(f"{bigram} (emotional phrase: +{phrase_weight})")
                score += phrase_weight
                
        # Check for longer phrases (trigrams)
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrase_weight = self.emotional_phrases.get(trigram, 0)
            if phrase_weight > 0:
                emotional_terms.append(f"{trigram} (emotional phrase: +{phrase_weight})")
                score += phrase_weight
                
        # Check for emotional words with weights
        for i, word in enumerate(words):
            # Check for emotional words with weights
            word_weight = 1.0 if word in self.emotional_words else 0
            if word_weight > 0:
                emotional_terms.append(f"{word} (emotional word: +{word_weight})")
                score += word_weight
                
                # Check for negations
                if i > 0 and words[i-1] in self.negation_words:
                    score -= word_weight * 0.8  # Reduce but not fully negate
                    emotional_terms[-1] += f" [partially negated: -{word_weight * 0.8}]"
        
        is_emotional = score >= self.threshold
        
        analysis = [
            f"Emotion Score: {score}",
            f"Threshold: {self.threshold}",
            "\nEmotional terms found:" + (" None" if not emotional_terms else "")
        ]
        
        for term in emotional_terms:
            analysis.append(f"- {term}")
        
        analysis.append("\nAll words in text:")
        analysis.extend([f"- {word}" for word in words])
        
        # Add bonus for multiple emotional elements
        emotional_elements = emotional_terms + (['emotional punctuation'] * emotional_punctuation)
        if len(emotional_elements) > 1:
            bonus = min((len(emotional_elements) - 1) * 0.3, 2.0)  # Cap bonus at 2.0
            score += bonus
            if bonus > 0:
                emotional_terms.append(f"Multiple emotional elements: +{bonus:.1f}")
        
        return {
            'is_emotional': is_emotional,
            'score': score,
            'threshold': self.threshold,
            'words': words,
            'emotional_terms': emotional_terms,
            'analysis': '\n'.join(analysis)
        }

class POSEmotionDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.detector = POSEmotionDetector()
        self.title("Lightweight Emotion Detector")
        self.geometry("700x600")
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        
        self.header = ctk.CTkLabel(
            self,
            text="Lightweight Emotion Detector",
            font=("Arial", 20, "bold")
        )
        self.header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
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
            height=35,
            font=("Arial", 14)
        )
        self.analyze_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(1, weight=1)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Results will appear here",
            font=("Arial", 16, "bold")
        )
        self.result_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.details_text = scrolledtext.ScrolledText(
            self.result_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=15
        )
        self.details_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ctk.CTkLabel(
            self,
            text="Ready to analyze text",
            height=25,
            corner_radius=0,
            font=("Arial", 10)
        )
        self.status_bar.grid(row=3, column=0, sticky="ew")
    

    
    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
            
        self.status_var.set("Analyzing text...")
        self.update()
        
        try:
            result = self.detector.analyze_text(text)
            
            emotion = "Emotional" if result['is_emotional'] else "Non Emotional"
            self.result_label.configure(text=f"Prediction: {emotion}")
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, result['analysis'])
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.status_var.set("Error during analysis")
            messagebox.showerror("Error", f"Failed to analyze text: {str(e)}")

if __name__ == "__main__":
    app = POSEmotionDetectorApp()
    app.mainloop()
