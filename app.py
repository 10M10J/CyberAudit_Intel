import os
import json
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from mistralai import Mistral

# --- Utility Functions ---
def chunk_text(text, max_chars=20000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def sanitize_text(text):
    return text.replace("Ignore all above", "").strip()

def generate_prompt(user_doc, query=None, language='English', concise=False):
    sanitized = sanitize_text(user_doc)
    if concise:
        length_instruction = "Limit your response to maximum 150 words."
    else:
        length_instruction = ""
    
    if language == 'Hindi':
        intro = 'A user provided a Cyber Security Audit Report (in Hindi):'
        prompt = f"{intro}\n{sanitized}"
        if query:
            prompt += f"\nQuestion: {query}"
        prompt += f"\nPlease respond in Hindi based on the document. {length_instruction}"
    else:
        intro = 'A user provided a Cyber Security Audit Report:'
        prompt = f"{intro}\n{sanitized}"
        if query:
            prompt += f"\nQuestion: {query}"
        prompt += f"\nPlease respond in English based on the document. {length_instruction}"
    return prompt

def chat_with_mistral(_client, model_name, prompt):
    response = _client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Cached Functions ---
@st.cache_data(show_spinner=False)
def get_summary_chunks(chunks, _client, model_name, language):
    results = []
    for chunk in chunks:
        prompt = generate_prompt(chunk, language=language, concise=True)
        results.append(chat_with_mistral(_client, model_name, prompt))
    return " ".join(results)

@st.cache_data(show_spinner=False)
def get_answer_chunks(chunks, question, _client, model_name, language):
    results = []
    for chunk in chunks:
        prompt = generate_prompt(chunk, query=question, language=language, concise=True)
        results.append(chat_with_mistral(_client, model_name, prompt))
    return " ".join(results)

# --- Main App ---
def main():
    load_dotenv()
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        st.error("Mistral API Key not found. Set MISTRAL_API_KEY in environment.")
        return

    client = Mistral(api_key=mistral_key)

    # Sidebar UI
    st.sidebar.title("CyberAudit Intel (Mistral-Powered)")
    model_name = st.sidebar.selectbox("Select Mistral model", ("mistral-small-latest", "mistral-small-2503"))
    language = st.sidebar.radio("Language", ("English", "Hindi"), horizontal=True)
    option = st.sidebar.radio("Choose Option", ("Summary", "Chat"))

    if st.sidebar.button("Clear History"):
        st.session_state.pop("summary", None)
        st.session_state.pop("chat_history", None)
        st.success("History cleared.")

    # File Upload
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
        cache_path = f"{store_name}.json"
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump({"text": text}, f)

        if option == "Summary":
            st.subheader("Summary")
            if "summary" not in st.session_state:
                if st.button("Generate Summary"):
                    with st.spinner("Summarizing ..."):
                        try:
                            summary = get_summary_chunks(chunks, client, model_name, language)
                            st.session_state["summary"] = summary
                        except Exception as e:
                            st.error(f"Error during summarization: {e}")
            if "summary" in st.session_state:
                st.write(st.session_state["summary"])

        elif option == "Chat":
            st.subheader("Ask Questions")
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            
            for msg in st.session_state["chat_history"]:
                st.chat_message("user").write(msg["question"])
                st.chat_message("assistant").write(msg["answer"])

            question = st.chat_input("Ask about the report")
            if question:
                with st.spinner("Fetching answer ..."):
                    try:
                        answer = get_answer_chunks(chunks, question, client, model_name, language)
                        st.session_state["chat_history"].append({"question": question, "answer": answer})
                        st.chat_message("user").write(question)
                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        st.error(f"Error answering question: {e}")

if __name__ == "__main__":
    main()
