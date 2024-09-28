import os
import boto3
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# OCR Function
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        if not text.strip():
            logger.warning("OCR produced empty result")
        return text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

# PDF Processing Function
def process_pdf(file):
    text = ""
    try:
        # Extract text using PyMuPDF
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        
        # Perform OCR on all pages
        file.seek(0)  # Reset file pointer
        pdf_bytes = file.read()
        images = convert_from_bytes(pdf_bytes)
        ocr_text = ""
        for image in images:
            ocr_text += perform_ocr(image) + "\n"
        
        # Combine PyMuPDF text and OCR text
        combined_text = text + "\n" + ocr_text
        
        if not combined_text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise

# Data Ingestion Implementation
def data_ingestion(file):
    if file.filename.endswith('.pdf'):
        text = process_pdf(file)
    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file)
        text = perform_ocr(image)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image file.")

    if not text.strip():
        raise ValueError("No text could be extracted from the document. Please check your uploaded file.")

    logger.info(f"Extracted text (first 100 chars): {text[:100]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    return [Document(page_content=t) for t in docs]

# Vector Embedding and Vector Store Implementation
def get_vector_store(docs):
    try:
        embeddings = bedrock_embeddings.embed_documents([doc.page_content for doc in docs])
        if not embeddings:
            raise ValueError("No embeddings were generated. Please check your documents and embedding model.")
        
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        return vectorstore_faiss
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 2000})
    return llm

# Summarization function
def summarize_documents(docs, llm):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

prompt_template = """
Human: You are a medical coding assistant.
Your task is to suggest the most appropriate ICD-10 code based on the given medical text and potential codes.
Explain broader about the disease if you know. 
<context>
{context}
</context>
Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

@app.route('/process', methods=['POST'])
def process_input():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            docs = data_ingestion(file)
            vectorstore = get_vector_store(docs)
            llm = get_llama3_llm()
            summary = summarize_documents(docs, llm)
            
            # Load the pre-existing FAISS index for ICD codes
            icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            
            response = get_response_llm(llm, icd_vectorstore, summary)
            
            return jsonify({
                "summary": summary,
                "icd_codes": response
            }), 200
        
        elif 'query' in request.form:
            query = request.form['query']
            
            # Load the pre-existing FAISS index for ICD codes
            icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            
            llm = get_llama3_llm()
            response = get_response_llm(llm, icd_vectorstore, query)
            
            return jsonify({
                "icd_codes": response
            }), 200
        
        else:
            return jsonify({"error": "No file or query provided"}), 400
    
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)