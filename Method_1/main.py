import tkinter as tk
from tkinter import messagebox
import os

def load_emotional_words(filepath="Method_1/emotional_words.txt"):
    """Loads emotional words from a file, one word per line."""
    try:
        
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        
        
        if os.path.basename(base_dir) == "Method_1":
            words_file_path = os.path.join(base_dir, "emotional_words.txt")
        elif os.path.exists(os.path.join(base_dir, "Method_1", "emotional_words.txt")):
             words_file_path = os.path.join(base_dir, "Method_1", "emotional_words.txt")
        else:
            words_file_path = filepath


        if not os.path.exists(words_file_path):
            alt_path = "emotional_words.txt" 
            if os.path.basename(os.getcwd()) == "Mini Project" and os.path.exists(os.path.join("Method_1", alt_path)):
                 words_file_path = os.path.join("Method_1", alt_path)
            else: 
                messagebox.showerror("Error", f"Emotional words file not found at expected paths: {words_file_path} or {alt_path}")
                return set()

        with open(words_file_path, "r") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        return words
    except Exception as e:
        messagebox.showerror("Error loading words", f"Could not load emotional words: {e}")
        return set()

emotional_words_set = load_emotional_words()

def check_emotion():
    """Checks the input sentence for emotional words and displays the result."""
    input_sentence = sentence_entry.get().lower()
    if not input_sentence:
        result_label.config(text="Please enter a sentence.", fg="orange")
        return


    words_in_sentence = set(input_sentence.split())
    
    is_emotional = False
    found_emotional_words = []

    for word in words_in_sentence:
        
        cleaned_word = word.strip('''!"?,.:;''')
        if cleaned_word in emotional_words_set:
            is_emotional = True
            found_emotional_words.append(cleaned_word)

    if is_emotional:
        result_label.config(text=f"Emotional (found: {', '.join(found_emotional_words)})", fg="red")
    else:
        result_label.config(text="Non Emotional", fg="green")

root = tk.Tk()
root.title("Emotion Checker - Method 1")
root.geometry("450x200")

root.configure(bg="#f0f0f0")
font_style = ("Arial", 12)
button_font_style = ("Arial", 10, "bold")

tk.Label(root, text="Enter a sentence about cricket:", font=font_style, bg="#f0f0f0").pack(pady=(10,5))
sentence_entry = tk.Entry(root, width=50, font=font_style)
sentence_entry.pack(pady=5, padx=10)

check_button = tk.Button(root, text="Check Emotion", command=check_emotion, font=button_font_style, bg="#4CAF50", fg="white", relief=tk.RAISED)
check_button.pack(pady=10)

result_label = tk.Label(root, text="", font=font_style, bg="#f0f0f0")
result_label.pack(pady=5)

tk.Label(root, text="Checks if any word in the sentence matches a predefined list of emotional words.", font=("Arial", 9),bg="#f0f0f0").pack(pady=(5,0))


if not emotional_words_set:
    check_button.config(state=tk.DISABLED)
    result_label.config(text="Error: Emotional words list could not be loaded. Check console/popups.", fg="red")
    if not os.path.exists("Method_1/emotional_words.txt") and not os.path.exists("emotional_words.txt"):
         messagebox.showwarning("File Missing", "emotional_words.txt not found in Method_1/ or current directory.")


root.mainloop()
