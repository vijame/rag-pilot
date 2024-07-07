Goal: Learn AI using langchain and build a simple system that could answer questions based on set of HTML documents.

Functional Requirement:
1. Post question
2. Get relevant answer

Non-fuctional Requirement: 
1. Correctness
2. Highly available

Approach:
1. Data Preparation
    i. Need HTML documents.
    ii. Clean HTML documents remove unecessary elements like scripts and styles using BeautifulSoup library.
    iii. Extract actual text from HTML document.
2. Text Embedding
    i. Break extraced text into chunks to process.
    ii. Convert chunk of text into numerical representation which is known as embedding, using pre-trained model from HuggingFace. Which helps to capture the semantic meaning of the text.
3. Store smaller chunks as Vector & query processing
    i. Store text embedding in FAISS(Facebook AI similarity Search) index, which is a highly efficient data structure for nearest neighbor search. 
    ii. Search chunks of text using similarity search to given query based on indexing. 
4. Question answering
    i. Use a pre-trained question answering model from HuggingFace to answer questions based on the relevant text chunks. This model is been trained on large dataset of question-answer pairs and can generate answers to new questions based on a given context.
5. Answer Aggregation
    i. Split context into multiple chunks and generates an answer for ach chunk 
    ii. selects best one based on the score provided by QA model.

Use Case
1. Used to help ourselves/e-learning to find exact value to questions or keys provided data, like what is value of container for xyz enviornment.
2. Used to build a customer support chatbot thatcan answer question based on company's documentation.

Challenges & Improvements or ToLearn 
1. Get Intelligent and Correct answers. 
2. Analyze different format of data like PDF, Images, Word, Webpages, etc.
3. Scope of training data.
4. Keep converstion i.e query and answer till reach correct answer.
5. Self-train QA model.

Things to know!
1. The choice of the model
Refer: https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads
“distilbert-base-uncased-distilled-squad” is based on the specific task at hand, which is question answering. This model is a distilled version of the BERT model that has been fine-tuned on the SQuAD (Stanford Question Answering Dataset), making it well-suited for question answering tasks.

“DistilBERT” is a smaller, faster, cheaper version of BERT that retains most of its accuracy. The “uncased” part means that the model does not make a distinction between upper and lower case letters (i.e., it treats “apple”, “Apple”, and “APPLE” the same). The “distilled-squad” part indicates that the model has been fine-tuned on the SQuAD dataset.

The choice of this specific model would have been based on considerations such as the nature of the task, the available computational resources, and the desired trade-off between speed and accuracy. It’s worth noting that the transformers library provides a wide range of pre-trained models


Problem: 
Sometimes model is not likely providing the expected answer.
Answers are not coherent and reads more like a list of fragmented chunks.
Not retrieving enough chunks from the documents to cover scope.
Focus on topic of question, clear definitions.
Why: 
i. May be due to insufficent context, retrieved chunks from vector store does not contain that information. The model can only provide answers based on the context it receives. 
ii. Question might be too broad for context.
iii. Embeddings might not be capturing nuances of the question well enough to retrieve most relevant chunks. 
Solution: 
i. Increase k in similarity_search to receive more chunks
ii. Refine question 
iii. Presenting the answer is also important like aggregrate and present only give 1 best answer or list the answers
iv: need to refine and make it more structured and understandable
v: We need strategy to define broad answers like increase number of retreived chunks to cover breadth of the topic
vi: Filter duplicate and irrelevant chunks
vii: Format the answer in proper way so you are able to undersatnd may be as a numbered list, etc
viii: We need chunking strategy, eg: `RecursiveCharacterTextSplitter` currently splits the text into 1000 characters with overlap of 200 characters (chunk_size=1000, chunk_overlap=200). This might not be sufficient to cover all parts of the documents where relevant answers are located.
ix: Document Relevance: The vector similarity search (vectorstore.similarity_search(query, k=30)) might not be retrieving enough relevant documents that contain answers to whole scope.
Increasing k (number of retrieved documents) could help fetch more relevant context.


Load and Clean HTML Files: This step remains the same.
Split Text into Chunks: This step remains the same.
Initialize Embeddings and Vector Store: This step remains the same.
Initialize the QA Model and Tokenizer: This step remains the same.
Determine if a Question is Specific or Broad: This step remains the same.
Aggregate and Present Answers: Modify this part to handle both specific and broad questions.

Commands:
Recommended to use python virtual environment:
python3 -m venv `<python-virtual-env-directory>`

Activate the virtual environment
`source <python-virtual-env-directory>bin/activate`

In case to de-activate: `deactivate`

To check version
`python3 --version`
`pip3 --version`

To add alias
`alias python=python3`
`alias pip=pip3`

Activate the virtual environment
`source <python-virtual-env-directory>/bin/activate`

Select virtual environment python interpretor in VS code:
Cmd+Shift+P
Search for and select "Python: Select Interpreter."
Then enter the path of the virtual environment.

Verify if the Python interpreter exists
`ls <python-virtual-env-directory>/bin/python`

To install all dependencies
`pip3 install -r requirements.txt`

If you have issues with langchain use below command
`pip show langchain`

Remove existing virtual environment if it exists
`rm -rf <python-virtual-env-directory>`

Recreate virtual environment
`python3 -m venv <python-virtual-env-directory>`

How to generate open-api keys -> this does not train on your data 
`https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/`