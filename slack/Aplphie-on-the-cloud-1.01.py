#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install functions
import sys
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from functions import draft_email


# In[2]:


# Load environment variables from .env file
load_dotenv(find_dotenv())


# In[3]:


# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]


# In[4]:


# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)


# In[5]:


# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


# In[6]:


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


# In[8]:


import os
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from slack_bolt import App
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv(find_dotenv())

# Load the EPUB document
epub_loader = UnstructuredEPubLoader("Chapter-7-NMSA-1978.epub")
documents = epub_loader.load()

# Refine text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([text.page_content for text in texts], embeddings)

# Define the RetrievalQA chain
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),  # Consistency
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 results
    return_source_documents=True  # Enable source documents for debugging
)

# Memory for conversation context
memory = ConversationBufferMemory()

# Define prompt structure for system and human prompts
system_template = "You are a helpful assistant at New Mexico Tax & Rev. Explain Chapter-7-NMSA-1978 clearly and accurately."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "User asks: {user_input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

@app.event("app_mention")
def handle_mentions(body, say):
    try:
        event_text = body.get("event", {}).get("text", "")
        user_input = event_text.split(">", 1)[-1].strip()

        # Check if the query is about New Mexico Statutes
        if "new mexico statute" in user_input.lower():
            response = retrieval_qa_chain.run(user_input)
            # Return the result or a fallback message if no result
            say(response['result'] if 'result' in response else "I couldn't find any information on that topic.")
        else:
            # Handle general queries with the same RetrievalQA
            response = retrieval_qa_chain.run(user_input)
            # Return the result or a fallback message
            say(response['result'] if 'result' in response else "I couldn't generate a response.")

    except Exception as e:
        # Log the error with more detail
        print("Error handling app_mention:", e)
        # Return a fallback response in case of an error
        say("Sorry, I encountered an error while processing your request. Could you rephrase or ask again?")



# In[10]:


import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Initialize the OpenAI-based language model with conversational capabilities
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a memory buffer for maintaining context in conversations
memory = ConversationBufferMemory()

# System prompt template to define AI assistant's behavior
system_template = """
You are Alphie, a conversational AI assistant at New Mexico Tax & Rev. 
Your role is to engage in meaningful conversations and help with a variety of tasks, including answering general questions, solving coding issues, and more.
Provide thoughtful and friendly responses to all user questions. 


One of your primary tasks is to analyze the Current New Mexico Statutes Annotated 1978. Make sure to do a detailed analysis. 

"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human message prompt to integrate user input into the conversation
human_template = "User says: {user_input}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create a conversational prompt with system and human messages
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Create a conversational chain with memory for maintaining context
chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)

# Define the event listener for Slack messages
@app.event("message")
def handle_message_events(event, say):
    user_text = event.get("text", "")

    # If the user text is a greeting
    if user_text.lower() in ["hi", "hello", "hey"]:
        say("Hello! I'm Alphie, your friendly AI assistant at New Mexico Tax & Rev. How can I help you today?")
        return

    # Generate the response with the conversational chain
    response = chain.run(user_input=user_text)

    # Respond with the generated response
    say(response)


# In[11]:


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# In[ ]:


# Initialize the Flask app
if __name__ == "__main__":
    flask_app.run()


# In[ ]:




