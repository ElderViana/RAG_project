import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Meu Assistente RAG", page_icon="🤖", layout="centered")
st.title("🤖 Pergunte ao nosso assistente")
st.write("Faça perguntas sobre os nossos cursos.")

CAMINHO_DB = "db"
PROMPT_TEMPLATE = """
Responda a pergunta do usuário:
{pergunta}

Com base estritamente nessas informações do documento:
{base_conhecimento}

Se você não encontrar a resposta no contexto acima, responda exatamente: "Não consegui encontrar essa informação nos documentos cadastrados."
"""

@st.cache_resource
def iniciar_sistema_rag():
    funcao_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt_objeto = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return db, llm, prompt_objeto

db, llm, prompt_objeto = iniciar_sistema_rag()

if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if pergunta_usuario := st.chat_input("Digite sua dúvida aqui..."):
    
    with st.chat_message("user"):
        st.write(pergunta_usuario)
    st.session_state.historico.append({"role": "user", "content": pergunta_usuario})

    resultados = db.similarity_search_with_relevance_scores(pergunta_usuario, k=3)
    
    with st.chat_message("assistant"):
        if len(resultados) == 0 or resultados[0][1] < 0.4:
            resposta_final = "Não consegui encontrar nenhuma informação relevante nos documentos cadastrados."
            st.warning(resposta_final)
        else:
            with st.spinner("Buscando nos documentos e pensando..."):
                textos_contexto = [res[0].page_content for res in resultados]
                base_conhecimento = "\n\n----\n\n".join(textos_contexto)
                
                prompt_formatado = prompt_objeto.invoke({
                    "pergunta": pergunta_usuario, 
                    "base_conhecimento": base_conhecimento
                })
                resposta_llm = llm.invoke(prompt_formatado)
                resposta_final = resposta_llm.content
                
                st.write(resposta_final)
        
        st.session_state.historico.append({"role": "assistant", "content": resposta_final})