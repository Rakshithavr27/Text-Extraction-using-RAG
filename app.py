import pdfplumber
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

pdf_path = r"C:\Users\rashm\Downloads\RESUME_rakshitha27.pdf"  # Replace with your PDF file path
text = extract_text_from_pdf(pdf_path) 
print(text)
import pdfplumber
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

pdf_path = r"C:\Users\rashm\Downloads\RESUME_rakshitha27.pdf"  # Replace with your PDF file path
text = extract_text_from_pdf(pdf_path) 
print(text)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s@]', ' ', text)  # Removing any non-alphabetic characters
    stop_words = set(["the", "and", "is", "in", "to", "of", "for", "with", "a", "an"])
    text = " ".join([word for word in text.split() if word not in stop_words])  # Removing stopwords
    return text

# Function to split text into chunks of a specified size (e.g., 30 words per chunk)
def split_text_into_chunks(text, chunk_size=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Main function to process a single document or all documents in a folder
def process_documents(path):
    all_chunks = []
    
    if os.path.isdir(path):
        # If the path is a directory, get all files inside it
        files = os.listdir(path)
        file_paths = [os.path.join(path, filename) for filename in files]
    else:
        file_paths = [path]
    
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)  # Extract text using pdfplumber for PDFs
        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()  # Read text file
        else:
            print(f"Unsupported file format: {file_path}")
            continue
        
        # Preprocess and split text into chunks
        processed_text = preprocess_text(text)
        chunks = split_text_into_chunks(processed_text)
        all_chunks.extend(chunks)  # Add chunks to the list
    
    return all_chunks

# Test the function with a PDF file path
pdf_path = r"C:\Users\rashm\Downloads\RESUME_rakshitha27.pdf"  # Replace with your actual PDF path
document_chunks = process_documents(pdf_path)

# Display the first 5 chunks for verification
for i, chunk in enumerate(document_chunks[:5]):
    print(f"Chunk {i+1}:\n{chunk}\n")
model = SentenceTransformer('all-MiniLM-L6-v2')
def create_faiss_index(chunks):
    # Encode chunks into embeddings
    embeddings = model.encode(chunks)
    print("Float embeddings:\n", embeddings)  # Print embeddings in float format

    # Convert embeddings to binary format for each vector
    binary_embeddings = [np.packbits(np.asarray(embedding, dtype=np.float32).view(np.uint8)) for embedding in embeddings]
    print("\nBinary embeddings:")
    for binary_embedding in binary_embeddings:
        print(binary_embedding)  # Print each embedding in binary format

    # Create a FAISS index with float32 embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    index.add(np.array(embeddings).astype('float32'))
    print("\nFAISS Index:", index)  # Print index for verification
    
    return index, embeddings
chunks = ["This is a sample chunk.", "This is another sample chunk."]
create_faiss_index(chunks)
# Function to generate an answer using retrieved chunks
def generate_answer(query, index, embeddings, chunks):
    # Encode the query into an embedding
    query_embedding = model.encode([query])
    
    # Search for the top 5 most similar chunks in the FAISS index
    _, indices = index.search(np.array(query_embedding).astype('float32'), k=5)
    
    # Collect the top matching chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    
    # Use a generative model to answer the question based on the relevant chunks
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    context = " ".join(relevant_chunks)
    
    # Generate an answer based on the combined context
    input_text = f"Question: {query} Context: {context}"
    response = qa_pipeline(input_text)
    
    return response[0]['generated_text']
# Example Usage
pdf_path = r"C:\Users\rashm\Downloads\RESUME_rakshitha27.pdf"  # Replace with your PDF file path
text = extract_text_from_pdf(pdf_path)  # Step 1: Extract text from PDF
chunks = split_text_into_chunks(text)  # Step 2: Split text into chunks
index, embeddings = create_faiss_index(chunks)  # Step 3: Create FAISS index

# Step 4: Ask a question and generate an answer
query = "Extract the email from the resume"
answer = generate_answer(query, index, embeddings, chunks)
print("Answer:", answer)
# Step 4: Ask a question and generate an answer
query = "Extract the cgpa from the resume"
answer = generate_answer(query, index, embeddings, chunks)
print("Answer:", answer)
