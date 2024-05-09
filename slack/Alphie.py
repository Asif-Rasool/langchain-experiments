# The Base Version

import logging
import os
import time
from functools import wraps

from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, abort
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

import os
import re
from flask import Flask, request
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
import pandas as pd
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

# Initialize the Slack app and Flask app only once
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)  # Corrected initialization
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def require_slack_verification(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not verify_slack_request():
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


def verify_slack_request():
    # Get the request headers
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # Check if the timestamp is within five minutes of the current time
    current_timestamp = int(time.time())
    if abs(current_timestamp - int(timestamp)) > 60 * 5:
        return False

    # Verify the request signature
    return signature_verifier.is_valid(
        body=request.get_data().decode("utf-8"),
        timestamp=timestamp,
        signature=signature,
    )

def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# OpenAI-based language model with conversational capabilities
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a memory buffer for maintaining context in conversations
memory = ConversationBufferMemory()

# Create system and human prompts
system_template = """
You are Alphie, a conversational AI assistant at New Mexico Tax & Rev. 
Your role is to engage in meaningful conversations and help with various tasks.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "User says: {user_input}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Conversational chain with memory
chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    return raw_text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()  # Strip leading and trailing whitespace
    return text

# Function to analyze PDF content
def analyze_pdf(pdf_path, question):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n", " ", "."],  # Better context splitting
    )
    texts = text_splitter.split_text(cleaned_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    qa_chain = load_qa_chain(OpenAI(), chain_type="refine")

    docs = docsearch.similarity_search(question, k=5)  # Context for similarity search
    answer = qa_chain.run(input_documents=docs, question=question)

    return answer

# Load the mapping from the CSV file for keyword-to-PDF mapping
csv_path = "C:/Users/asifr\OneDrive - State of New Mexico\Documents/GitHub/langchain-experiments/slack/output_tables/article_titles.csv"  # Update with the correct path
pdf_dir = "articles_output"  # Directory with PDFs

article_pdf_map = {}
df = pd.read_csv(csv_path)

# Populate the dictionary with ARTICLE numbers and short titles
for _, row in df.iterrows():
    article_number = row["ARTICLE Number"]
    short_title = row["Short Title"]

    pdf_name = f"ARTICLE {article_number}.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_name)

    article_pdf_map[article_number] = pdf_path
    article_pdf_map[short_title.lower()] = pdf_path

# Function to get the PDF path based on a keyword
def get_pdf_path_from_keyword(keyword):
    keyword = keyword.lower()
    for key in article_pdf_map:
        if keyword in key:
            return article_pdf_map[key]
    return None

# Slack event handler for message events
@app.event("message")
def handle_message_events(event, say):
    user_text = event.get("text", "").lower()

    # Respond to greetings
    if user_text in ["hi", "hello", "hey"]:
        say("Hello! I'm Alphie, your friendly AI assistant. How can I help you today?")
        return

    # Check if the message contains a keyword that maps to a PDF
    pdf_path = get_pdf_path_from_keyword(user_text)

    if pdf_path:
        question = re.sub(r"(" + "|".join(article_pdf_map.keys()) + ")", "", user_text).strip()
        
        response = analyze_pdf(pdf_path, question)

        if not response or "I don't know" in response:
            say("I'm not sure I understood that. Could you please rephrase your question?")
        else:
            say(response)
        return

    # Default response for other queries
    response = chain.run({"user_input": user_text})
    say(response)

# Flask endpoint for Slack events
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

# Run the Flask app
if __name__ == "__main__":
    logging.info("Flask app started")
    flask_app.run(host="0.0.0.0", port=8000)


