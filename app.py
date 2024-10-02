import time
import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from PyPDF2 import PdfReader
import google.generativeai as genai
from google.ai import generativelanguage as glm

def chat_with_google(prompt, _model):
    """
    Sends a prompt to the Google API and retrieves the AI's response.

    Parameters:
    prompt (str): The prompt to send to the AI.
    _model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response.
    """
    completion = _model.generate_content(prompt)
    return completion.text

@st.cache_data()
def get_summarization(user_doc, _model, language_option=None):
    """
    Generates a summarization prompt based on the user's document and retrieves the AI's response.

    Parameters:
    user_doc (str): The user's document.
    _model (str): The AI model to use for the response.
    language_option (str): Language for summarization.

    Returns:
    str: The content of the AI's response to the summarization prompt.
    """
    prompt = generate_prompt(user_doc=user_doc, language=language_option)
    return chat_with_google(prompt, _model)

def get_answers(user_doc, question, _model, language_option=None):
    """
    Generates a question-answer prompt based on the user's question and document.

    Parameters:
    user_doc (str): The user's document.
    question (str): The user's question.
    _model (str): The AI model to use for the response.
    language_option (str): Language for the answer.

    Returns:
    str: The content of the AI's response to the question answer prompt.
    """
    prompt = generate_prompt(user_doc=user_doc, query=question, language=language_option)
    return chat_with_google(prompt, _model)

def generate_prompt(user_doc, query=None, language='English'):
    """
    Helper function to generate prompts for summarization and question answering.

    Parameters:
    user_doc (str): The user's document.
    query (str): The user's question.
    language (str): The language for the response (English or Hindi).

    Returns:
    str: A prompt to send to the AI.
    """
    if language == 'Hindi':
        intro = 'A user has uploaded a Cyber Security Audit Report document. Here is the document in Hindi:'
        query_section = f'\nUser asked the following question:\n{query}\n' if query else ''
        prompt = f"{intro}\n{user_doc}\n{query_section}\nPlease summarize in Hindi or answer the question based on the document."
    else:
        intro = 'A user has uploaded a Cyber Security Audit Report document.'
        query_section = f'\nUser asked the following question:\n{query}\n' if query else ''
        prompt = f"{intro}\n{user_doc}\n{query_section}\nPlease summarize in English or answer the question based on the document."
    return prompt

def main():
    # Sidebar contents
    with st.sidebar:
        st.title('CyberAudit Intel - Your Audit Report Chatbot')
        st.markdown('''
        ## About
        Discover CyberAudit Intel, revolutionizing cybersecurity audit report analysis with advanced NLP and AI. 
        Effortlessly upload unstructured audit reports for in-depth analysis and actionable insights. 
        Streamline your decision-making process by obtaining real-time answers to critical questions. 
        Leverage our AI-powered chatbot for instant support and enhanced clarity. Your security decisions, simplified!

        ''')
        st.write('Made in India :flag-in: by [StirPot](https://stirpot.in/)')

    # Initialize text variable
    text = ""
    language_option = ""

    # Get the Google API key
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')

    if not google_api_key:
        st.error("Google API Key not found. Please ensure it is properly configured.")
        return

    genai.configure(api_key=google_api_key)

    # Create the model
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    _model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    st.header("Chat with your CyberSecurity Audit Reports", divider='rainbow')

    language_option = st.radio('Select your preferred Language of interaction',
                               ('English', 'Hindi'), index=0, horizontal=True)

    # Upload a PDF file
    pdf = st.file_uploader("Upload your Audit Report file (pdf only) ", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Display spinner while summarizing the document
        with st.spinner('Processing your document for summarization...'):
            summarization = get_summarization(text, _model, language_option)
        st.success("Summary generated successfully!")
        st.write(summarization)

        store_name = pdf.name[:-4]

        # Store the file content locally if not already stored
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                file_contents = pickle.load(f)
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(text, f)

    # Accept user inputs/questions
    query = st.chat_input("Ask questions about your Audit Report PDF")
    
    if query:
        # Display spinner while generating the answer
        with st.spinner('Fetching the answer from your document...'):
            llm_response = get_answers(user_doc=text, question=query, _model=_model, language_option=language_option)
        st.write("CyberAudit Intel:", llm_response)


if __name__ == '__main__':
    main()
