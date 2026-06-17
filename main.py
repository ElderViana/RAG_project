import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("O script main.py foi iniciado com sucesso!")

CAMINHO_DB = "db"


prompt_template = """
Responda a pergunta do usuário:
{pergunta}

Com base nessas informações:

{base_conhecimento}

Se você não encontrar a resposta para a pergunta do usuário nessas informações,
responda: "Não sei te dizer isso."
"""

def perguntar():
    funcao_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embedding
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print(f"Documentos no banco: {db._collection.count()}")
    print("-" * 30)

    while True:
        pergunta = input("\nEscreva a sua pergunta (or 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            print("Encerrando o chat...")
            break

        resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
        
        print("\n" + "="*40)
        if len(resultados) > 0:
            print(f"-> [DEBUG] Melhor score encontrado: {resultados[0][1]:.4f}")
            print(f"-> [DEBUG] Conteúdo do pedaço mais próximo: {resultados[0][0].page_content[:100]}...")
        else:
            print("-> [DEBUG] O banco retornou zero documentos para essa busca.")
        print("="*40 + "\n")

        if len(resultados) == 0 or resultados[0][1] < 0.4:
            print("\n[Aviso]: Não consegui encontrar nenhuma informação relevante na base.")
            continue
            
        textos_resultados = []
        for resultado in resultados:
            texto = resultado[0].page_content 
            textos_resultados.append(texto)


        base_conhecimento = "\n\n----\n\n".join(textos_resultados)
        
        prompt_template_objeto = ChatPromptTemplate.from_template(prompt_template)
        prompt_formatado = prompt_template_objeto.format(
            pergunta=pergunta, 
            base_conhecimento=base_conhecimento
        )
        
        print("\nPensando...")
        
        texto_resposta = llm.invoke(prompt_formatado)

        print("Resposta da IA:\n", texto_resposta.content)
        print("-" * 30)

if __name__ == "__main__":
    perguntar()