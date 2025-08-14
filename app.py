import os
import json
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from mistralai import Mistral

# Utility: chunk text
def chunk_text(text, max_chars=20000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# Safe text cleaning
def sanitize_text(text):
    # Remove control chars, obfuscate dangerous prompt patterns
    return text.replace("Ignore all above", "").strip()

# Prompt generation
def generate_prompt(user_doc, query=None, language='English'):
    sanitized = sanitize_text(user_doc)
    if language == 'Hindi':
        intro = 'A user provided a Cyber Security Audit Report (in Hindi):'
        prompt = f"{intro}\n{sanitized}"
        if query:
            prompt += f"\nQuestion: {query}"
        prompt += "\nPlease respond in Hindi based on the document."
    else:
        intro = 'A user provided a Cyber Security Audit Report:'
        prompt = f"{intro}\n{sanitized}"
        if query:
            prompt += f"\nQuestion: {query}"
        prompt += "\nPlease respond in English based on the document."
    return prompt

# Model interaction
def chat_with_mistral(client, model_name, prompt):
    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@st.cache_data(show_spinner=False)
def get_summary_chunks(chunks, _client, model_name, language):
    results = []
    for chunk in chunks:
        prompt = generate_prompt(chunk, language=language)
        results.append(chat_with_mistral(client, model_name, prompt))
    return "\n--- CHUNK ---\n".join(results)

@st.cache_data(show_spinner=False)
def get_answer_chunks(chunks, question, _client, model_name, language):
    results = []
    for chunk in chunks:
        prompt = generate_prompt(chunk, query=question, language=language)
        results.append(chat_with_mistral(client, model_name, prompt))
    return "\n--- CHUNK ANSWER ---\n".join(results)

def main():
    load_dotenv()
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        st.error("Mistral API Key not found. Set MISTRAL_API_KEY in environment.")
        return

    client = Mistral(api_key=mistral_key)

    st.sidebar.title("CyberAudit Intel (Mistral-Powered)")
    st.sidebar.markdown("""
    Upload your cybersecurity audit report, get summaries or ask questions.
    Supports English and Hindi, safe local storage, and large-doc handling.
    """)

    st.header("Upload & Analyze Your Audit Report")
    model_name = st.selectbox("Select Mistral model",
                              ("mistral-small-latest", "mistral-small-2503"))
    language = st.radio("Language", ("English", "Hindi"), horizontal=True)

    uploaded_pdf = st.file_uploader("Upload PDF (Audit Report)", type="pdf")
    if uploaded_pdf:
        try:
            reader = PdfReader(uploaded_pdf)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return

        if not text.strip():
            st.error("No text extracted from PDF.")
            return

        chunks = chunk_text(text)
        store_name = uploaded_pdf.name[:-4]

        # Save for reuse
        cache_path = f"{store_name}.json"
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump({"text": text}, f)

        st.write("Document loaded successfully.")

        if st.button("Generate Summary"):
            with st.spinner("Summarizing ..."):
                try:
                    summary = get_summary_chunks(chunks, client, model_name, language)
                    st.success("Summary ready.")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error during summarization: {e}")

        question = st.text_input("Ask a question about the report")
        if question:
            with st.spinner("Finding the answer ..."):
                try:
                    answer = get_answer_chunks(chunks, question, client, model_name, language)
                    st.write("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error answering question: {e}")

if __name__ == "__main__":
    main()
