import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore # Alterado
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Meu Assistente RAG", page_icon="🤖", layout="centered")
st.title("🤖 Pergunte ao nosso assistente")
st.write("Faça perguntas sobre os nossos cursos.")

PROMPT_TEMPLATE = """
Responda a pergunta do usuário:
{pergunta}

Com base estritamente nessas informações do documento:
{base_conhecimento}

Se você não encontrar a resposta no contexto acima, responda exatamente: "Não consegui encontrar essa informação nos documentos cadastrados."
"""

@st.cache_resource
def iniciar_sistema_rag():
    # O modelo de embedding precisa ser o mesmo usado na criação
    funcao_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Conexão com o Pinecone
    db = PineconeVectorStore(
        index_name="seu-indice-no-pinecone", # Certifique-se que o nome é igual ao que criou
        embedding=funcao_embedding
    )
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt_objeto = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return db, llm, prompt_objeto

db, llm, prompt_objeto = iniciar_sistema_rag()

# ... (resto do seu código de histórico e chat_input permanece igual)

if pergunta_usuario := st.chat_input("Digite sua dúvida aqui..."):
    # ... (exibição da mensagem do usuário)
    
    # O Pinecone também permite busca similar:
    resultados = db.similarity_search_with_score(pergunta_usuario, k=3)
    
    with st.chat_message("assistant"):
        # Ajuste a lógica de score conforme o que o Pinecone retornar
        if len(resultados) == 0: 
            resposta_final = "Não consegui encontrar informações relevantes."
            st.warning(resposta_final)
        else:
            # ... (seu código de processamento da LLM continua igual)