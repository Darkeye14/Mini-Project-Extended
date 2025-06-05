import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import threading

load_dotenv()

llm = None
llm_chain = None

def initialize_llm():
    global llm, llm_chain
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        messagebox.showerror("API Key Error", "GOOGLE_API_KEY not found. Please set it in a .env file.")
        return False

    try:

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.1)

        prompt_template_text = """Analyze the following cricket-related sentence and determine if it expresses emotion.
Sentence: "{sentence}"
Your response must be ONLY "Emotional" or "Non Emotional". Do not add any other words or explanations.
Classification:"""
        
        prompt = PromptTemplate(template=prompt_template_text, input_variables=["sentence"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return True
    except Exception as e:
        messagebox.showerror("LLM Initialization Error", f"Failed to initialize Gemini LLM: {e}")
        return False

def classify_sentence_thread():
    """Handles the LLM classification in a separate thread to keep UI responsive."""
    input_sentence = sentence_entry.get()
    if not input_sentence:
        result_text.set("Please enter a sentence.")
        status_label.config(text="Status: Waiting for input", fg="orange")
        return

    if not llm_chain:
        result_text.set("LLM not initialized. Check API key and logs.")
        status_label.config(text="Status: LLM Error", fg="red")
        return

    status_label.config(text="Status: Processing...", fg="blue")
    check_button.config(state=tk.DISABLED) 

    try:
        response = llm_chain.run(sentence=input_sentence)


        if isinstance(response, dict) and 'text' in response:
            classification = response['text'].strip()
        elif isinstance(response, str):
            classification = response.strip()
        else:
            classification = "Error: Unexpected response format"
            print(f"Unexpected LLM response format: {response}")


        if classification in ["Emotional", "Non Emotional"]:
            result_text.set(f"LLM Classification: {classification}")
            status_label.config(text="Status: Done!", fg="green" if classification == "Non Emotional" else "red")
        else:
            result_text.set(f"LLM Classification: Error (unexpected output: \"{classification}\")")
            status_label.config(text="Status: Error - Unexpected LLM output", fg="red")
            print(f"Unexpected LLM output: {classification}")
            
    except Exception as e:
        result_text.set(f"Error: {e}")
        status_label.config(text="Status: Error during classification", fg="red")
        messagebox.showerror("Classification Error", f"An error occurred: {e}")
    finally:
        check_button.config(state=tk.NORMAL)

def classify_sentence():
    threading.Thread(target=classify_sentence_thread, daemon=True).start()

root = tk.Tk()
root.title("Emotion Checker - Method 2 (LLM)")
root.geometry("500x300")

root.configure(bg="#e6e6fa")
font_style = ("Arial", 12)
button_font_style = ("Arial", 10, "bold")
status_font_style = ("Arial", 10, "italic")

main_frame = tk.Frame(root, bg="#e6e6fa", padx=10, pady=10)
main_frame.pack(expand=True, fill=tk.BOTH)

tk.Label(main_frame, text="Enter a cricket sentence:", font=font_style, bg="#e6e6fa").pack(pady=(5,5))
sentence_entry = tk.Entry(main_frame, width=60, font=font_style)
sentence_entry.pack(pady=5, padx=5)

check_button = tk.Button(main_frame, text="Check Emotion with LLM", command=classify_sentence, font=button_font_style, bg="#4a90e2", fg="white", relief=tk.RAISED)
check_button.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(main_frame, textvariable=result_text, font=font_style, bg="#e6e6fa", wraplength=450)
result_label.pack(pady=10)
result_text.set("Click the button to classify.")

status_label = tk.Label(main_frame, text="Status: Waiting for LLM initialization...", font=status_font_style, bg="#e6e6fa", fg="gray")
status_label.pack(pady=(5,0), side=tk.BOTTOM, fill=tk.X)

if initialize_llm():
    status_label.config(text="Status: LLM Ready", fg="green")
else:
    check_button.config(state=tk.DISABLED)
    status_label.config(text="Status: LLM Initialization Failed. Check .env and API key.", fg="red")
    result_text.set("LLM could not be initialized. Please ensure GOOGLE_API_KEY is set in a .env file.")

root.mainloop()
