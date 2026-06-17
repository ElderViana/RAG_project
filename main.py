def perguntar():
    funcao_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Inicialização do Pinecone
    db = PineconeVectorStore(
        index_name="seu-indice-no-pinecone",
        embedding=funcao_embedding
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print("✅ Conectado ao Pinecone com sucesso!")
    print("-" * 30)

    while True:
        pergunta = input("\nEscreva a sua pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            break

        # Nota: similarity_search_with_relevance_scores não é nativo do Pinecone 
        # da mesma forma que no Chroma. Usamos similarity_search_with_score
        resultados = db.similarity_search_with_score(pergunta, k=3)
        
        print("\n" + "="*40)
        # O Pinecone retorna score de distância (quanto menor, mais próximo)
        # Diferente do Chroma que retorna score de relevância (quanto maior, mais próximo)
        if len(resultados) > 0:
            print(f"-> [DEBUG] Score de distância: {resultados[0][1]:.4f}")
            print(f"-> [DEBUG] Conteúdo: {resultados[0][0].page_content[:100]}...")
        print("="*40 + "\n")

        # Ajuste o threshold conforme necessário para o Pinecone
        if len(resultados) == 0:
            print("\n[Aviso]: Não encontrei dados.")
            continue
            
        textos_resultados = [res[0].page_content for res in resultados]
        base_conhecimento = "\n\n----\n\n".join(textos_resultados)
        
        prompt_template_objeto = ChatPromptTemplate.from_template(prompt_template)
        prompt_formatado = prompt_template_objeto.format(
            pergunta=pergunta, 
            base_conhecimento=base_conhecimento
        )
        
        print("\nPensando...")
        texto_resposta = llm.invoke(prompt_formatado)
        print("Resposta da IA:\n", texto_resposta.content)