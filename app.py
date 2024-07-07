import os
import re
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


source_folder = 'source-folder'
cleaned_html_content = []

# Function to clean HTML content
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# Load and clean HTML files
for filename in os.listdir(source_folder):
    if filename.endswith('.html'):
        file_path = os.path.join(source_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            cleaned_html = clean_html(html_content)
            cleaned_html_content.append(cleaned_html)

print(f"Loaded and cleaned {len(cleaned_html_content)} HTML files.")

# Combine all cleaned text into a single string
all_cleaned_text = " ".join(cleaned_html_content)

# Adjust chunking parameters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
# Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(all_cleaned_text)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(text_chunks, embeddings)

# Initialize the QA model and tokenizer
# model_name = "distilbert-base-uncased-distilled-squad"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Initialize a more suitable model for text generation
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

# def extract_questions(text):
#     # Pattern to match potential questions
#     question_pattern = r'(?:^|\n)(?:Q:|Question:|What is|How to|Explain|Describe|Define|Compare|List|Why|When).*?\?'
#     questions = re.findall(question_pattern, text, re.IGNORECASE | re.MULTILINE)
#     return [q.strip() for q in questions]

# def extract_questions(text):
#     # More flexible pattern to match potential questions
#     question_pattern = r'(?:^|\n)(?:\d+[\.\)]\s*|Q:|Question:|What|How|Explain|Describe|Define|Compare|List|Why|When).*?[?:]'
#     questions = re.findall(question_pattern, text, re.IGNORECASE | re.MULTILINE)
#     return [q.strip() for q in questions]

# def extract_content(text):
#     # Fallback method to extract content if no questions are found
#     lines = text.split('\n')
#     return [line.strip() for line in lines if line.strip()]

# Function to get an answer to a query
def get_answer(query):
    docs = vectorstore.similarity_search(query, k=10)  # Retrieve top 10 relevant documents
    context = " ".join([doc.page_content for doc in docs])
    # if is_broad_question(query):
    #     # Extract questions from the context
    #     questions = extract_questions(context)
    #     # If no questions found, fall back to content extraction
    #     if not questions:
    #         questions = extract_content(context)
    #     # Remove duplicates while preserving order
    #     unique_questions = list(dict.fromkeys(questions))
    #     # Limit to top 60 questions (or less if not enough found)
    #     top_questions = unique_questions[:60]
    #     # Format the answer
    #     combined_answer = "\n".join([f"{i+1}. {q}" for i, q in enumerate(top_questions)])
    #     if len(top_questions) < 60:
    #         combined_answer += f"\n\nNote: Only {len(top_questions)} relevant items were found in the provided context."
    # else:
        # For specific questions, use the existing method
    combined_answer = generate_answer(query, context, max_length=300)
    return combined_answer

# Example questions for testing
specific_question = "What is Lazy Loading vs. Eager Loading?"
specific_answer = get_answer(specific_question)
print(f"Question: {specific_question}")
print(f"Answer: {specific_answer}")

# broad_question = "what is DI?"
# broad_answer = get_answer(broad_question)
# print(f"Question: {broad_question}")
# print(f"Answer:\n{broad_answer}")

# broad_question = "What spring?"
# broad_answer = get_answer(broad_question)
# print(f"Question: {broad_question}")
# print(f"Answer:\n{broad_answer}")
