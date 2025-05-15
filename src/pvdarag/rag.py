from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI
from openai import OpenAI
import numpy as np
import getpass
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the LLM with the API key defined above
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0, # Temperature to 0 since we want it to be highly reliable
    api_key=os.getenv("OPENAI_API_KEY")  
)

# Load PDF 
loader = PyPDFLoader("/Users/bartamin/Documents/PvdA-RAG/src/Strijden-voor-de-ziel-van-de-stad-Verkiezingsprogramma-PvdA-Amsterdam-2022-2026-1.pdf")
docs = loader.load()

# Parse PDF into recursive chunks
parser = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
chunks = parser.split_documents(docs)

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Embed the parsed chunks with openAI embedding model
client = OpenAI()
texts = [chunk.page_content for chunk in chunks]
response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

# Store chunk embeddings in vector store
vector_store = {}
for i, chunk in enumerate(chunks):
    vector_store[i] = {
        'text': chunk.page_content,
        'embedding': response.data[i].embedding
    }

# Get embedding for question
def similarity_search(question, vector_store):
    """
    Gives back the 3 chunks of text that is semantically most similar to the input question,
    First it embeds the input question, and computes cosine similarity with chunk embeddings
    to retrieve the top matches

    Args:
        question: The question of the user.
        Vector_store: Vector database containing chunk embeddings.

    Returns:
        Text of the top 3 matches.
    """
    cos_sim = []
    indices = []
    response = client.embeddings.create(
        input= question,
        model="text-embedding-3-small"
    )
    embedding_q = np.array(response.data[0].embedding)
    for i, item in vector_store.items():
        embedding_i = np.array(item['embedding'])
        sim = np.dot(embedding_q, embedding_i) / (np.linalg.norm(embedding_q) * np.linalg.norm(embedding_i))
        cos_sim.append((i, sim))
    cos_sim_sorted = sorted(cos_sim, key=lambda x: x[1], reverse=True)
    top_match = cos_sim_sorted[:3]
    for item in top_match:
        indices.append(item[0])
    context = " ".join(vector_store[i]['text'] for i in indices)
    return context

prompt = ChatPromptTemplate.from_template(
    "Je bent een politicus voor PvdA Amsterdam. Potentiele kiezers zullen je vragen stellen met betrekking tot"
    "het verkiezingsprogramma van de PvdA Amsterdam voor de Gemeenteraad 2022-2026 verkiezingen. Jouw taak is om deze"
    "vragen zo goed mogelijk te beantwoorden. Gebruik de volgende context uit het PvdA verkiezingsprogramma om de" 
    "vraag te beantwoorden.\n" "Het is heel belangrijk dat je antwoorden PUUR gebaseerd zijn op de context" 
    "(het verkiezingsprogramma). Als het antwoord op de vraag niet direct van de context is af te leiden, antwoord"
    "dan dat je helaas de vraag niet kan beantwoorden\n\n"
    "Context:\n{context}\n\n"
    "Vraag: {question}\n"
    "Antwoord:"
)

def answer_question(question):
    """
    For a given question, it retrieves the chunks of text that are semantically most similar to it
    (function: similarity_search), and based on that prompts gpt-4o-mini for a response and prints it

    Args:
        Question: The question of the user.

    Returns:
        Answer of gpt-4o-mini
    """
    context = similarity_search(question, vector_store)
    messages = prompt.invoke({'question': question, 'context': context})
    answer = llm.invoke(messages).content
    print(answer)