import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
from groq import Groq
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Loading .env file
load_dotenv()

# Setting up Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

MODEL_DESCRIPTIONS = {
    "llama3-70b-8192": "Deep, structured answers. Limited input size.",
    "llama3-8b-8192": "Balance of speed and context.",
    "gemma-7b-it": "Fast, lightweight. Good for short Q&A."
}

# Cleaned OCR text for image 

def clean_ocr_text(text):
    cleaned = re.sub(r'(?<=\d)\n(?=\d)', '', text)
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', cleaned)
    return cleaned.strip()

# PDF extraction

def get_pdf_text(pdf_docs, max_pages=10):
    all_texts = []
    for pdf in pdf_docs:
        text = ""
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            if page.extract_text():
                text += page.extract_text()
        all_texts.append(text)
    return all_texts

# OCR from Image

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    st.session_state["last_uploaded_image"] = image
    st.image(image, caption="Uploaded Image", width=300)
    text = pytesseract.image_to_string(image)
    cleaned_text = clean_ocr_text(text)
    with st.expander("Show OCR Extracted Text"):
        st.text(cleaned_text)
    return cleaned_text

# Text splitter

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# FAISS store

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Groq LLM query

def query_groq_llm(prompt, model_name):
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an analytical assistant. Give direct answers with tables, bullet points, and factual content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Error querying model: {str(e)}"

# Combining all inputs

def combined_input_handler(query, model_name, image_text="", pdf_contexts=None):
    full_context = ""
    if image_text:
        full_context += f"Image Text:\n{image_text.strip()}\n\n"
    if pdf_contexts:
        for idx, pdf_text in enumerate(pdf_contexts):
            full_context += f"PDF Report {idx + 1} Text:\n{pdf_text.strip()}\n\n"

    if not full_context.strip():
        st.warning("No content provided from image or PDF to answer the question.")
        return

    max_tokens = 4500
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.encode(full_context)
    if len(tokens) > max_tokens:
        full_context = tokenizer.decode(tokens[:max_tokens])

    if image_text and not pdf_contexts:
        prompt = (
            f"You're given a financial table in text form. Answer directly in one line. "
            f"For example: 'Total income in FY 2022-23 is ₹726.89 billion.'\n\n"
            f"{query}\n\n{full_context}"
        )
    else:
        prompt = f"{query}\n\n{full_context}\n\nAnswer:"

    response = query_groq_llm(prompt, model_name)
    if response.strip():
        st.markdown("### Answer")
        st.success(response)
    else:
        st.warning("Unable to generate a valid answer. Please check your input.")

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((query, response))

# Main UI

def main():
    st.set_page_config("Sustainability Reports Chatbot", layout="centered")
    st.title("Sustainability Reports Chatbot")

    st.sidebar.header("Upload PDF Reports")
    pdf_docs = st.sidebar.file_uploader("Upload PDF(s)", accept_multiple_files=True)

    st.sidebar.header("Upload Image")
    image_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    model_name = st.sidebar.selectbox("Choose Groq Model", list(MODEL_DESCRIPTIONS.keys()), format_func=lambda x: f"{x} ({MODEL_DESCRIPTIONS[x]})")

    if st.sidebar.button("Process the Files"):
        if not pdf_docs and not image_file:
            st.sidebar.warning("Please upload at least one PDF or image.")
        else:
            with st.spinner("Processing content..."):
                image_text, pdf_texts = "", []
                if pdf_docs:
                    pdf_texts = get_pdf_text(pdf_docs)
                    st.session_state["processed_pdf_texts"] = pdf_texts
                if image_file:
                    image_text = extract_text_from_image(image_file)
                    st.session_state["processed_image_text"] = image_text
                st.success("Done! You can now summarize or ask questions.")

    if "last_uploaded_image" in st.session_state:
        st.image(st.session_state["last_uploaded_image"], caption="Latest Uploaded Image", width=300)
    
    st.markdown("### Summarize Reports")
    if st.button("Summarize All Uploaded Reports"):
        image_text = st.session_state.get("processed_image_text", "")
        pdf_texts = st.session_state.get("processed_pdf_texts", [])
        summarize_prompt = (
            "Summarize and compare all the uploaded reports.\n"
            "Use structured bullet points and tables for:\n"
            "- Key highlights\n"
            "- Metrics achieved\n"
            "- Unique initiatives\n"
            "- Progress against targets\n"
            "- Certifications, Frameworks used\n"
            "- Differences in approach across years."
        )
        combined_input_handler(summarize_prompt, model_name, image_text=image_text, pdf_contexts=pdf_texts)

    st.markdown("### Ask your questions?")
    user_question = st.text_input("Type your question here:")
    if user_question:
        image_text = st.session_state.get("processed_image_text", "")
        pdf_texts = st.session_state.get("processed_pdf_texts", [])
        combined_input_handler(user_question, model_name, image_text=image_text, pdf_contexts=pdf_texts)

    if "history" in st.session_state and st.session_state.history:
        with st.expander("Chat History"):
            for q, a in st.session_state.history[::-1]:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Groq:** {a}")
                st.markdown("---")

if __name__ == "__main__":
    main()
