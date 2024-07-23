import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import filedialog
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Downloading the required resources from nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Prepare the text by tokenizing it and using the root of the words for better comparison
def prepare_text(text):
    common_words = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    unique_tokens = [lemma.lemmatize(word) for word in tokens if word not in common_words]
    resultant_text = ' '.join(unique_tokens)

    return resultant_text

# Compare the 2 texts: first vectorize them and then use cosine comparison
def tfidf_vectorize(texts):
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(texts)
    return vector

def score_by_cosine(text1_vector, text2_vector):
    if text1_vector.shape[1] != text2_vector.shape[1]:
        text2_vector = text2_vector.T
    score = cosine_similarity(text1_vector, text2_vector)
    return score

def check_for_plagiarism():
    text1 = text_area.get(1.0, "end")
    text2 = ref_text_area.get(1.0, "end")
    if not text1 or not text2:
        messagebox.showerror("Error", "Both text areas must be filled.")
        return

    prep_text1 = prepare_text(text1)
    prep_text2 = prepare_text(text2)

    vectors = tfidf_vectorize([prep_text1, prep_text2])
    text2_vector = vectors[1:2]
    text1_vector = vectors[0:1]

    plag_scores = score_by_cosine(text1_vector, text2_vector)
    result_message = f"{plag_scores[0][0] * 100:.2f}%"

    results_area.delete(1.0, "end")
    results_area.insert("end", result_message)


def load_and_display(entry, text_widget):
    file_path = entry.get()

    if not file_path:
        file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, "end")
        entry.insert("end", file_path)
        with open(file_path, 'r') as file:
            text = file.read()
            text_widget.delete(1.0, "end")
            text_widget.insert("end", text)

GUI = tk.Tk()
GUI.geometry("720x400")
GUI.title("Plagiarism Checker")

frame = tk.Frame(GUI)
frame.grid(padx=10, pady=10)

# Text Labels
textLabel = tk.Label(frame, text="Text:")
textLabel.grid(row=0, column=0, padx=5, pady=5)
ref_textLabel = tk.Label(frame, text="Reference Text:")
ref_textLabel.grid(row=0, column=1, padx=5, pady=5)

# Text Areas
text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=40, height=10, font=("Times", 12))
text_area.grid(row=1, column=0, padx=5, pady=5)
ref_text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=40, height=10, font=("Times", 12))
ref_text_area.grid(row=1, column=1, padx=5, pady=5)

# Entry fields and buttons for loading text
text_entry = tk.Entry(frame, width=50)
text_entry.grid(row=2, column=1, padx=5, pady=5)
load_text_button = tk.Button(frame, text="Load Text File", command=lambda: load_and_display(text_entry, text_area))
load_text_button.grid(row=2, column=0, padx=5, pady=5)

ref_text_entry = tk.Entry(frame, width=50)
ref_text_entry.grid(row=3, column=1, padx=5, pady=5)
load_ref_button = tk.Button(frame, text="Load Reference Text", command=lambda: load_and_display(ref_text_entry, ref_text_area))
load_ref_button.grid(row=3, column=0, padx=5, pady=5)

# Button to check for plagiarism
func_button = tk.Button(frame, text="Check for Plagiarism", command=check_for_plagiarism)
func_button.grid(row=4, column=0, columnspan=2, pady=5)

# Result
resultsLabel = tk.Label(frame, text="Similarity Score:")
resultsLabel.grid(row=5, column=0, padx=5, pady=5)
results_area = tk.Text(frame, wrap=tk.WORD, width=10, height=1)
results_area.grid(row=5, column=1, columnspan=2, padx=5, pady=5)

GUI.mainloop()