
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
# csv_path = "C:/Users/asifr/OneDrive - State of New Mexico\Documents/GitHub/langchain-experiments/slack/output_tables/article_titles.csv"  # Update with correct path
pdf_dir = "articles_output"  # Directory with PDFs

# import pandas as pd
# import os

# article_pdf_map = {}
# df = pd.read_csv(csv_path)

# # Populate the dictionary with ARTICLE numbers and short titles
# for _, row in df.iterrows():
#     article_number = row["ARTICLE Number"]
#     short_title = row["Short Title"]

#     pdf_name = f"ARTICLE {article_number}.pdf"
#     pdf_path = os.path.join(pdf_dir, pdf_name)

#     article_pdf_map[article_number] = pdf_path
#     article_pdf_map[short_title.lower()] = pdf_path

# def get_pdf_path_from_keyword(keyword):
#     keyword = keyword.lower()
#     for key in article_pdf_map:
#         if keyword in key:
#             return article_pdf_map[key]
#     return None

# Function to get the PDF path based on a keyword
def get_pdf_path_from_keyword(keyword, pdf_dir):
    keyword = keyword.lower()
    for file_name in os.listdir(pdf_dir):
        if file_name.lower().startswith("article") and file_name.lower().endswith(".pdf"):
            article_number = file_name.lower().split("article")[1].split(".pdf")[0].strip()
            if keyword in article_number:
                return os.path.join(pdf_dir, file_name)
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
        # Display a loading message while the analysis is in progress
        loading_message = "Fetching data from FRED website. Please wait..."
        print(loading_message)
        
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

#10 Function to analyze an excel file

import pandas as pd

def analyze_excel_file(file_path):
    """
    Analyzes each sheet of the Excel file located at the given file path and generates a summary of the analysis.
    
    Parameters:
        file_path (str): The path to the Excel file.
        
    Returns:
        str: A summary of the analysis for each sheet.
    """
    try:
        # Read the Excel file
        xls = pd.ExcelFile(file_path)
        
        # Initialize an empty string to store the summary
        summary_all_sheets = ""
        
        # Analyze each sheet
        for sheet_name in xls.sheet_names:
            try:
                # Display a loading message while the analysis is in progress
                loading_message = "Analyzing the Excel file. Please wait..."
                print(loading_message)
               
                # Read the current sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Check if the DataFrame has columns
                if df.empty or len(df.columns) == 0:
                    summary = f"The sheet '{sheet_name}' doesn't contain any data."
                else:
                    # Exclude non-numeric columns
                    numeric_columns = df.select_dtypes(include=['float64', 'int64'])
                    
                    # Perform analysis on numeric columns
                    summary_stats = numeric_columns.describe().to_markdown()  # Get summary statistics
                    correlation_matrix = numeric_columns.corr().to_markdown()  # Compute correlation matrix
                    missing_values = df.isnull().sum().to_frame().to_markdown()  # Count missing values

                    # Create descriptive text for each section of the analysis
                    summary_text = f"**Summary Statistics:**\n\n" \
                                   f"The summary statistics provide a descriptive overview of the numerical data in the sheet. " \
                                   f"They include measures such as mean, median, standard deviation, minimum, and maximum values.\n\n" \
                                   f"**Correlation Matrix:**\n\n" \
                                   f"The correlation matrix shows the pairwise correlations between the numerical columns. " \
                                   f"A correlation close to 1 indicates a strong positive correlation, while a correlation close to -1 indicates a strong negative correlation.\n\n" \
                                   f"**Missing Values:**\n\n" \
                                   f"The missing values table displays the number of missing values for each column in the sheet.\n\n"

                    # Create a summary of all analysis
                    summary = f"Analysis of sheet '{sheet_name}':\n\n" \
                              f"{summary_text}" \
                              f"**Summary Statistics:**\n{summary_stats}\n\n" \
                              f"**Correlation Matrix:**\n{correlation_matrix}\n\n" \
                              f"**Missing Values:**\n{missing_values}\n\n"
                
                # Append the summary for the current sheet to the overall summary
                summary_all_sheets += summary + "\n\n"
                
            except Exception as e:
                # If an error occurs during analysis of the current sheet, append an error message to the overall summary
                summary_all_sheets += f"Error occurred while analyzing the sheet '{sheet_name}': {str(e)}\n\n"

        # Return the overall summary of all sheets
        return summary_all_sheets.strip()
        
    except Exception as e:
        # If an error occurs while processing the Excel file, return an error message
        return f"Error occurred while analyzing the Excel file: {str(e)}"

# 11 Function to handle file upload

import requests
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError

def handle_file_upload(event, say):
    try:
        # Get file information from the event
        file_id = event["files"][0]["id"]
        file_name = event["files"][0]["name"]
        
        # Initialize WebClient
        client = WebClient(token=SLACK_BOT_TOKEN)

        # Get file information
        file_info = client.files_info(file=file_id)
        file_url = file_info["file"]["url_private_download"]

        # Download the file
        response = requests.get(url=file_url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"})
        file_content = response.content

        # Save the file locally
        local_file_path = f"./{file_name}"
        with open(local_file_path, "wb") as f:
            f.write(file_content)

        # Analyze the Excel file
        analysis_result = analyze_excel_file(local_file_path)

        # Respond with the analysis result
        say(f"Analysis of the uploaded Excel file '{file_name}':\n{analysis_result}")

    except Exception as e:
        say(f"Error occurred while processing the uploaded file: {str(e)}")



#12 Function to get yahoo finance data

import pandas as pd
import yfinance as yf

import pandas as pd
import yfinance as yf

def fetch_historical_data_yahoo(series_id):
    try:
        # Fetch the historical data from Yahoo Finance
        ticker = yf.Ticker(series_id)
        data = ticker.history(period="max")
        
        if not data.empty:
            return format_data(data['Close'])
        
        return f"No data found for the given series ID on Yahoo Finance."

    except Exception as e:
        # If there's an error, return the error message
        return f"An error occurred while fetching data from Yahoo Finance: {str(e)}"


def format_data(data):
    # Create a dataframe and get the first 5 and last 5 rows
    table_data = pd.DataFrame(data, columns=["Value"]).reset_index()
    table_data.columns = ["Date", "Value"]
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

    # Return the data table
    return table_str

