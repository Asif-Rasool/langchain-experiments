import os
from dotenv import find_dotenv, load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def load_environment_variables():
    load_dotenv(find_dotenv())


def read_pdf_text(pdf_path):
    raw_text = ''
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def split_text(raw_text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)


def create_embeddings_and_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)


def answer_question(vectorstore, question):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = vectorstore.similarity_search(question)
    return chain.run(input_documents=docs, question=question)


def process_pdf_and_answer_questions(pdf_path, questions):
    load_environment_variables()
    raw_text = read_pdf_text(pdf_path)
    texts = split_text(raw_text)
    vectorstore = create_embeddings_and_vectorstore(texts)

    results = {}
    for question in questions:
        answer = answer_question(vectorstore, question)
        results[question] = answer

    return results
