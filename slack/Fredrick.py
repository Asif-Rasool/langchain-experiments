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

# Load environment variables from .env file
load_dotenv(find_dotenv())

from fredapi import Fred
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import re
from datetime import datetime

# Remove duplicate import statements
from slack_sdk import WebClient
from dotenv import find_dotenv, load_dotenv
import os

# Load environment variables
load_dotenv(find_dotenv())

# Set Slack API credentials and FRED API key
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN_FRED")  # Using get to avoid KeyError
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET_FRED")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID_FRED")


# Initialize the Slack app and Flask app
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler


# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)
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
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN_FRED"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")



# Initialize the Slack app and Flask app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Initialize the OpenAI-based language model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create a memory buffer for conversational context
memory = ConversationBufferMemory()

# Create system and human prompts for Fredrick-specific conversations
system_template = """
You are Fredrick, an AI assistant working at the New Mexico Tax & Revenue Department. 
Your role is to assist with data-related queries, provide economic data on demand, and engage in meaningful conversations.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "User says: {user_input}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Create a conversational chain with memory for FRED-related queries
chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)

# Initialize the FRED API
fred = Fred(api_key=FRED_API_KEY)

# Function to fetch data from FRED based on a series ID
def fetch_historical_data_from_fred(series_id):
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"  # Source URL

    try:
        # Fetch the entire time series
        series = fred.get_series(series_id)
        if series.empty:
            return f"No data found for the given series. You can check the source here: {source_url}"

        # Create a dataframe and get the first 5 and last 5 rows
        table_data = pd.DataFrame(series, columns=["Value"]).reset_index()
        table_data.columns = ["Date", "Value"]

        # Get the first 5 and last 5 rows
        first_rows = table_data.head(5)
        last_rows = table_data.tail(5)

        # Format as a Slack-friendly markdown table
        table_str = "```"  # Slack code block
        table_str += "{:<15} {:>10}\n".format("Date", "Value")

        # Add the first 5 rows
        for _, row in first_rows.iterrows():
            date = str(row["Date"])  # Use raw date
            table_str += "{:<15} {:>10}\n".format(date, f"{row['Value']:.2f}")

        # Add ellipses to indicate omitted rows
        table_str += "... [omitted] ...\n"

        # Add the last 5 rows
        for _, row in last_rows.iterrows():
            date = str(row["Date"])  # Use raw date
            table_str += "{:<15} {:>10}\n".format(date, f"{row['Value']:.2f}")

        table_str += "```"

        # Return the data table along with the source link
        return f"{table_str}\nSource: {source_url}"

    except Exception as e:
        # If there's an error, return the error message along with the source URL
        return f"An error occurred while fetching data: {str(e)}. You can check the source here: {source_url}"

# Function to get series ID based on a keyword
def get_series_id_from_keyword(keyword):
    # Search for FRED series based on a keyword
    search_result = fred.search(keyword)
    if search_result.empty:
        return None
    # Get the first result from the search
    return search_result.iloc[0]["id"]

# Slack event handler for all messages
@app.event("message")
def handle_all_messages(event, say):
    user_text = event.get("text", "").lower()

    # Respond to greetings
    if user_text in ["hi", "hello", "hey"]:
        say("Hi! I'm Fredrick, your AI Assistant at New Mexico Tax & Revenue, and I specialize in managing all data requests.. How can I help you today?")
        return

    # Determine which data to fetch based on the user's request
    keyword = re.search(r"\b(\w+)\b", user_text).group(0)  # Extract the first word as the keyword
    series_id = get_series_id_from_keyword(keyword)

    if series_id:
        historical_data = fetch_historical_data_from_fred(series_id)
        say(f"Data for {keyword.upper()} from FRED:\n{historical_data}")  # Provide data with source link
    else:
        # If the series ID is not found, provide a default message with a source URL
        say("Sorry, I couldn't find any data for that query. You can check the FRED website for more information.")



# Flask endpoint for Slack events
@flask_app.route("/slack/events-FRED", methods=["POST"])
def slack_events():
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    logging.info("Flask app started")
    flask_app.run(host="0.0.0.0", port=8000)


# ngrok start --config=ngrok-instance2.yml --all



