import os
import re
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

source_folder = 'source-folder'
cleaned_html_content = []

# Function to clean HTML content
# takes raw HTML content as input 
# BeautifulSoup, a library for parsing HTML, to remove unnecessary elements like scripts and styles.
# extracts the text content and cleans it by removing extra spaces and leading/trailing whitespace.
# uses regular expression to clean text
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


# Load and clean HTML files
# iterate all files in a directory named source_folder
# checks if file name ends with .html indicating HTML file
# constructs the full path using os.path.join
# opens the file for reading with UTF-8 encoding ('r', encoding='utf-8')
# cleans the read HTML content to a string cleaned_html
# cleaned HTML content is appended to a list named cleaned_html_content
for filename in os.listdir(source_folder):
    if filename.endswith('.html'):
        file_path = os.path.join(source_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            cleaned_html = clean_html(html_content)
            cleaned_html_content.append(cleaned_html)

print(f"Loaded and cleaned {len(cleaned_html_content)} HTML files.")

# Combine all cleaned text into a single large string named 'all_cleaned_text'
all_cleaned_text = " ".join(cleaned_html_content)

# Adjust chunking parameters
# RecursiveCharacterTextSplitter class is from langchain used to split large text into smaller chunks
# Why chunking: Efficiency(Processing very large strings can be computationally expensive. Splitting into smaller chunks can improve processing speed for tasks like vectorization), Memory Constraints(Large language models or embedding models might have memory limitations. Chunking helps manage memory usage during processing), Parallelization(Chunking allows for potential parallelization of tasks, where different chunks can be processed simultaneously on multiple cores or machines (if available))
# chunk_size: This sets the maximum size (in characters) for each chunk (default 2000 here).
# chunk_overlap: This specifies the number of characters (default 400 here) to overlap between consecutive chunks. Overlap helps avoid information loss at chunk boundaries.
# How to decide chunk size and overlap, LLMs can handle larger chunks due to their processing power. Some embedding models might have limitations on the maximum sequence length they can handle. There is always trade-off between Performance vs. Accuracy, Larger chunks: Faster processing but potentially less accurate due to loss of context at chunk boundaries. Smaller chunks: Slower processing but potentially more accurate as context is better preserved within each chunk.
# If memory is limited, smaller chunks might be necessary to avoid overloading the system
# Overlapping chunks ensures information from the end of one chunk is included at the beginning of the next, improving context preservation. Too much overlap: Increases the overall data size and processing time.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
# Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(all_cleaned_text)

# Initialize embeddings and vector store
# pre-trained embedding model (from Hugging Face) to convert the cleaned text from all webpages into numerical vectors. These vectors capture the semantic meaning of the text, focusing on what the webpage content is "about."
# vectorstore: This is a database specifically designed to store and efficiently search these numerical representations (embeddings) of text data. The FAISS library is used to create a vector store that efficiently stores and indexes these embedding vectors.
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(text_chunks, embeddings)

# Initialize a more suitable model for text generation
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# max_length=150 will be overriden by function calling max_length=300 its python thing for preference.
# The code utilizes an f-string (indicated by the f before the quotation mark) to format the input text.
# it inserts the actual query and context variables within curly braces ({}).
# It adds line breaks (\n\n) to separate the question, context, and answer sections.
# input_text = "Question: What are the different types of cats?\n\nContext: This is a webpage about cats and their behavior. Here's an article on different dog breeds.\n\nAnswer:" 
# a tokenizer object (tokenizer) is used that can convert text into a suitable format for a LLM
# return_tensors="pt": This argument specifies that the output should be a PyTorch tensor (commonly used for deep learning models).
# max_length=1024: This sets the maximum length of the encoded sequence (in number of tokens). Longer sequences will be truncated (cut short).
# truncation=True: This instructs the tokenizer to truncate the sequence if it exceeds the max_length.
# Imagine the tokenizer converts each word into a unique numerical ID
# example:
# Question: 3 (ID for "question")
# What: 4 (ID for "what")
# are: 5 (ID for "are")
# the: 6 (ID for "the")
# different: 7 (ID for "different")
# types: 8 (ID for "types")
# of: 9 (ID for "of")
# cats: 10 (ID for "cats")
# Context: 11 (ID for "context")
# ... (IDs for other words in the context)
# Answer: 12 (ID for "answer")
# the tokenizer might truncate the sequence if the total length (considering all word IDs) goes beyond 1024. In this example, the context might be truncated if it's very long.
# The inputs variable will hold the encoded version of the input_text as a PyTorch tensor. This tensor represents the combined information about the question, context, and an empty answer placeholder, ready to be processed by the large language model (LLM).
# 12 (ID for "answer") # Likely the first token since it's the answer section
# Here (decoded from its corresponding ID)
# are (decoded from its corresponding ID)
# some (decoded from its corresponding ID)
# ... (decoded words based on the context, potentially truncated)
# answer = "Here are some ..." (depending on the context truncation)
def generate_answer(query, context, max_length=150):
    input_text = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Function to determine if a question is specific or broad
def is_broad_question(query):
    broad_keywords = ["list", "top", "examples", "questions"]
    return any(keyword in query.lower() for keyword in broad_keywords)

# Function to get an answer to a query
# vectorstore 
# Document Collection: Suppose you have a collection of three webpages about different topics:
# Page 1: "This is a webpage about cats and their behavior." (Focus: cats, behavior)
# Page 2: "Here's an article on different dog breeds." (Focus: dogs, breeds)
# Page 3: "Learn about various types of birds and their habitats." (Focus: birds, habitats)
# These vectors might capture the document's meaning based on word frequency, word co-occurrence, or other techniques. Let's assume the simplified vectors for these pages are:
# Page 1 Vector: [0.8, 0.5, 0.1, 0, 0] (High values for "cats" and "behavior")
# Page 2 Vector: [0, 0.7, 0.4, 0.2, 0] (High values for "dogs" and "breeds")
# Page 3 Vector: [0, 0, 0.1, 0.8, 0.6] (High values for "birds" and "habitats")
# If the user asks "What are the different types of cats?", the query might also be converted into a vector (e.g., [0.9, 0.3, 0, 0, 0] with high values for "cats" and potentially "types").
# The vectorstore.similarity_search function likely calculates the similarity between the query vector and the document vectors. Documents with vectors closest to the query vector (in terms of distance within the high-dimensional space) are considered the most relevant.
# with k=10 function retrieves the top 10 most similar documents based on the calculated similarities. In this example, Page 1 (about cats) would likely be the most relevant, followed by documents containing words related to cats or behavior (depending on the specific similarity calculation).
# eg: docs = [
#     # Document 1 content
#     {"page_content": "This is the content of the first webpage about cats."},
#     # Document 2 content
#     {"page_content": "Here's some information on dogs from the second webpage."},
#     # Document 3 content
#     {"page_content": "This webpage discusses different types of birds."}
# ]
# context = "This is the content of the first webpage about cats. Here's some information on dogs from the second webpage. This webpage discusses different types of birds."
def get_answer(query):
    docs = vectorstore.similarity_search(query, k=10)  # Retrieve top 10 relevant documents
    context = " ".join([doc.page_content for doc in docs])
    combined_answer = generate_answer(query, context, max_length=300)
    return combined_answer

# Example questions for testing
specific_question = "What is Lazy Loading vs. Eager Loading?"
specific_answer = get_answer(specific_question)
print(f"Question: {specific_question}")
print(f"Answer: {specific_answer}")