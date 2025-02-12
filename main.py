# Updated Imports for LangChain 0.3.1+ with ChromaDB
from langchain_community.vectorstores import Chroma  # Use Chroma instead of FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings  # Use Ollama for local embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os

pdfs_directory = 'pdfs/'
chroma_db_directory = 'chroma_db/'  #  Directory to store ChromaDB

# Ensure ChromaDB directory exists
if not os.path.exists(chroma_db_directory):
    os.makedirs(chroma_db_directory)

#  Updated usage of `OllamaEmbeddings` & `OllamaLLM`
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
model = OllamaLLM(model="deepseek-r1:7b")

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user question. If you don't know the answer, say that you don't know. Use up to three sentences, keeping the answer concise. Also mention the relevant pdf name, section name, and page number used for the answer.
Question: {question} 
Context: {context} 
Answer:
"""

# Ensure directory exists before saving
def upload_pdf(file):
    """Ensure directory exists and return valid file path."""
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)

    file_path = os.path.join(pdfs_directory, file.name)

    try:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Check if file was actually saved
        if os.path.exists(file_path):
            print(f"File successfully saved at: {file_path}")
            return file_path
        else:
            print(f"File saving failed: {file_path}")
            return None

    except Exception as e:
        print(f"Error while saving file: {e}")
        return None

# Use ChromaDB instead of FAISS
def create_vector_store(file_path):
    """Process PDF, split text, and store embeddings in ChromaDB."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )

    chunked_docs = text_splitter.split_documents(documents)

    # Store documents in ChromaDB
    db = Chroma.from_documents(
        chunked_docs,
        embedding=embeddings,
        persist_directory=chroma_db_directory  # Persistent storage for ChromaDB
    )
    
    db.persist()  # Save ChromaDB for future use
    print(f"ChromaDB stored at: {chroma_db_directory}")
    return db

# Update retrieval function for ChromaDB
def retrieve_docs(db, query, k=4):
    """Retrieve top-k most relevant documents using ChromaDB."""
    results = db.similarity_search(query, k)
    print(results)  # Debugging output
    return results

# Keep the same function for querying PDF context
def question_pdf(question, documents):
    """Generate an answer using retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

def load_existing_chroma_db():
    """Load existing ChromaDB instead of recreating it."""
    db = Chroma(
        persist_directory=chroma_db_directory,
        embedding_function=embeddings  #  Ensure embeddings are used
    )
    print(" Loaded existing ChromaDB successfully.")
    return db

