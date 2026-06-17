import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

PASTA_BASE = "base"

def criar_db():
    print("🚀 Iniciando processamento...")
    documentos = carregar_documentos()
    if not documentos:
        print("⚠️ Nenhum documento encontrado.")
        return
        
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)
    print("✅ Dados enviados para o Pinecone com sucesso!")
    
def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="**/*.pdf")
    return carregador.load()

def dividir_chunks(documentos):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=500
    )
    return separador.split_documents(documentos)

def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # O PineconeVectorStore.from_documents já cria ou adiciona ao índice existente
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="seu-indice-no-pinecone"
    )

if __name__ == "__main__":
    criar_db()