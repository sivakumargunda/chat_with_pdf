import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_advanced_conversational_chain():
    prompt_template = """
give the responce based on the pdf and add some intlligent to answer the qestions based on the pdf .
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.write("No relevant information found.")
            return

        chain = get_advanced_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])
        collect_feedback(response["output_text"])

    except ValueError as e:
        st.error(f"Error loading vector store: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def collect_feedback(response):
    feedback = st.text_area("Provide feedback on the response:")
    if st.button("Submit Feedback"):
        with open("feedback.txt", "a") as f:
            f.write(f"Response: {response} | Feedback: {feedback}\n")
        st.success("Thank you for your feedback!")

def display_pdf_metadata(pdf_docs):
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        metadata = pdf_reader.metadata
        st.write(f"Title: {metadata.get('/Title', 'Unknown')}")
        st.write(f"Author: {metadata.get('/Author', 'Unknown')}")
        st.write(f"Number of Pages: {len(pdf_reader.pages)}")

def main():
    st.set_page_config(page_title="Interactive PDF Chatbot", page_icon=":book:")
    st.header("Interactive PDF Chatbot with Gemini AI")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                st.write("Extracted Text:", raw_text)  # Debugging line to display extracted text
                if not raw_text.strip():
                    st.error("No text found in the provided PDF files.")
                    return
                text_chunks = get_text_chunks(raw_text)
                st.write("Text Chunks:", text_chunks)  # Debugging line to display text chunks
                get_vector_store(text_chunks)
                display_pdf_metadata(pdf_docs)
                st.success("Done")

if __name__ == "__main__":
    main()
