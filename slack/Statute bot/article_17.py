#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install langchain
# !pip install openai
# !pip install PyPDF2
# !pip install faiss-cpu
# !pip install tiktoken


# In[3]:


# Load environment variables from .env file
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


# In[4]:


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


# In[5]:


# location of the pdf file/files. 
reader = PdfReader('Article 17.pdf')


# In[6]:


# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


# In[7]:


raw_text[:100]


# In[8]:


# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


# In[9]:


len(texts)


# In[10]:


texts[0]


# In[11]:


texts[1]


# In[13]:


# Download embeddings from OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[14]:


docsearch = FAISS.from_texts(texts, embeddings)


# In[16]:


docsearch


# In[19]:


from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI


# In[20]:


chain = load_qa_chain(OpenAI(), chain_type="stuff")


# In[21]:


query = "what is this article about?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[22]:


query = "what does alcoholic beverage mean according to article 17?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[26]:


query = "what are the ammendments in article 17?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[27]:


query = "Comapre and contrast amendments 2000 and 2008?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


# In[ ]:




