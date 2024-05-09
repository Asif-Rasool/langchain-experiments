from PyPDF2 import PdfReader

# Load the PDF and extract text
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

def analyze_pdf(pdf_path, question):
    # Extract the full text from the PDF
    raw_text = extract_pdf_text(pdf_path)
    
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vector store
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load the QA chain
    qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Perform a similarity search and run the QA chain
    docs = docsearch.similarity_search(question)
    response = qa_chain.run(input_documents=docs, question=question)

    return response


