
# 1 Function to extract text from PDF

def extract_text_from_pdf(pdf_path):
    reader = fitz.open(pdf_path)
    raw_text = ""
    for page in reader:
        text = page.get_text()
        if text:
            raw_text += text + "\n"
    return raw_text

# 2 Function to clean text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = text.strip()  # Strip leading and trailing whitespace
    return text

# 3 Function to analyze PDF content

def analyze_pdf(pdf_path, question):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n", " ", "."],
    )
    texts = text_splitter.split_text(cleaned_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    qa_chain = load_qa_chain(OpenAI(), chain_type="refine")

    docs = docsearch.similarity_search(question, k=7)  # Context for similarity search
    answer = qa_chain.run(input_documents=docs, question=question)

    return answer

# 4 Function to get the PDF path based on a keyword

# Load the mapping from the CSV file for keyword-to-PDF mapping
csv_path = "C:/Users/asifr/OneDrive - State of New Mexico\Documents/GitHub/langchain-experiments/slack/output_tables/article_titles.csv"  # Update with correct path
pdf_dir = "articles_output"  # Directory with PDFs

import pandas as pd
import os

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

def get_pdf_path_from_keyword(keyword):
    keyword = keyword.lower()
    for key in article_pdf_map:
        if keyword in key:
            return article_pdf_map[key]
    return None

# 5 Function to perform Google Search

def google_search(query):
    API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"]
    params = {
        "q": query,
        "api_key": API_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results


# 6 Function to get a Wikipedia summary

def get_wikipedia_summary(topic):
    try:
        summary = wikipedia.summary(topic, sentences=2)
        return summary
    except wikipedia.exceptions.PageError:
        return "I couldn't find any information on that topic in Wikipedia."
    except wikipedia.exceptions.DisambiguationError:
        return "This topic is ambiguous. Could you be more specific?"

# 7 Function to get bot user id

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

# 8 Function to fetch data from FRED based on a series ID

import os
import re
import pandas as pd
from fredapi import Fred

# Load environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Initialize Fred API
fred = Fred(api_key=FRED_API_KEY)

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

# 9 Function to get series ID based on a keyword

def get_series_id_from_keyword(keyword):
    # Search for FRED series based on a keyword
    search_result = fred.search(keyword)
    if search_result.empty:
        return None
    # Get the first result from the search
    return search_result.iloc[0]["id"]



