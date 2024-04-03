import nltk
nltk.download("punkt")  # If not previously downloaded
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2

# Read content from a text file
with open('dummy.txt', 'r') as file:
    text = file.read().replace('\n', ' ')

# Read content from a PDF file
pdf_file = 'dummy.pdf'
pdf_obj = open(pdf_file, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_obj)

pdf_text = ""
for page in pdf_reader.pages:
    pdf_text += page.extract_text()
pdf_obj.close()  # close the pdf file

# Combine text from both sources
all_text = text + " " + pdf_text

sentences = nltk.tokenize.sent_tokenize(all_text)

def ask_question(question):
    query_vect = vectorizer.transform([question])
    similarities = cosine_similarity(query_vect, doc_vector)
    closest = np.argmax(similarities, axis=1)
    best_similarity_score = np.max(similarities, axis=1)

    if best_similarity_score < 0.01:  # you might need to adjust this threshold
        return "I don't know about that."
    return sentences[closest[0]]

vectorizer = TfidfVectorizer().fit(sentences)
doc_vector = vectorizer.transform(sentences)

# Loop to continuously ask questions
while True:
    # Read question from terminal
    question = input("\nEnter your question (or type 'exit' to stop): ")
    if question.lower() == 'exit':
        break
    print(ask_question(question))