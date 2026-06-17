import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = "base"
CAMINHO_DB = "db"  

def criar_db():

    documentos = carregar_documentos()
   
    chunks = dividir_chunks(documentos)

    vetorizar_chunks(chunks)
    
def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="**/*.pdf")
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=2000,     
        chunk_overlap=500,    
        length_function=len,
        add_start_index=True,
        is_separator_regex=False,
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks

def vetorizar_chunks(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(CAMINHO_DB):
        shutil.rmtree(CAMINHO_DB)

   


    # Em vez de persist_directory, você aponta para o índice criado no site do Pinecone
    vectorstore = PineconeVectorStore(
    index_name="seu-indice-no-pinecone",
    embedding=embeddings
    )

    # Para salvar os documentos:
    vectorstore.add_documents(chunks)


if __name__ == "__main__":
    criar_db()