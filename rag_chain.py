from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain():

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS vector DB
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs_for_bm25 = list(vectorstore.docstore._dict.values())
    def get_filtered_docs(source=None):

        if source is None:
            return docs_for_bm25

        return [
            doc
            for doc in docs_for_bm25
            if doc.metadata.get("source") == source
        ]

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Use the chat history only when necessary to understand the user's question.

Do not mention the chat history in your answer.

Do not mention the context in your answer.

Answer naturally and directly.

You MUST answer only using the provided context.

If the answer is not explicitly present in the context,
respond exactly with:

"I don't have enough information to answer this question."

Do not guess.
Do not infer.
Do not use outside knowledge.

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)
    
    rewrite_prompt = ChatPromptTemplate.from_template("""
Given the chat history and latest question:

- If the latest question is already complete and understandable on its own,
  return it unchanged.

- Only rewrite when the question contains references such as:
  "it", "they", "them", "that", "those", etc.

- Do not answer the question.

- Do not add new information.

- Do not explain the question.

Chat History:
{history}

Question:
{question}

Rewritten Question:
""")
    
    multi_query_prompt = ChatPromptTemplate.from_template(
"""
Generate 2 alternative search queries.

Rules:
- One query per line
- No numbering
- No bullet points
- No explanations
- Only the query text

Question:
{question}

Queries:
"""
)

    # Load local LLM
    llm = Ollama(model="llama3")

    rewriter_chain = (rewrite_prompt | llm | StrOutputParser())

    multi_query_chain = (multi_query_prompt| llm| StrOutputParser())

    # Format retrieved docs into text
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )
    
    def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):

        rrf_scores = {}

        # Vector rankings
        for rank, doc in enumerate(vector_docs, start=1):

            content = doc.page_content

            rrf_scores[content] = rrf_scores.get(content, 0) + (
                1 / (k + rank)
            )

        # BM25 rankings
        for rank, doc in enumerate(bm25_docs, start=1):

            content = doc.page_content

            rrf_scores[content] = rrf_scores.get(content, 0) + (
                1 / (k + rank)
            )

        return rrf_scores

    # Main RAG function
    def rag_pipeline(question,history = "",source=None):

        standalone_question = question

        if history.strip():

            standalone_question = rewriter_chain.invoke({
                "history": history,
                "question": question
            })

            print("\nREWRITTEN QUESTION:")
            print(standalone_question)

        query_text = multi_query_chain.invoke({
            "question": standalone_question
        })

        queries = [standalone_question]

        print("\nRAW MULTI QUERY OUTPUT:")
        print(query_text)

        queries.extend(
            [
                q.strip("- ").strip()
                for q in query_text.split("\n")
                if q.strip()
            ]
        )

        queries = list(dict.fromkeys(queries))

        print("\nMULTI QUERIES:")
        for q in queries:
            print(q)

        filtered_docs = get_filtered_docs(source)

        # Retrieve documents
        # Vector Retrieval
        all_vector_results = []

        for query in queries:

            results = vectorstore.similarity_search_with_score(
                query,
                k=5
            )

            all_vector_results.extend(results)

        filtered_vector_results = []

        for doc, score in all_vector_results:

            if (source is None or doc.metadata.get("source") == source):
                filtered_vector_results.append((doc, score))

        vector_docs = []

        for doc, score in filtered_vector_results:
            vector_docs.append(doc)

        # BM25 Retrieval
        filtered_bm25 = BM25Retriever.from_documents(filtered_docs)

        filtered_bm25.k = 3

        all_bm25_docs = []

        for query in queries:

            docs = filtered_bm25.invoke(query)

            all_bm25_docs.extend(docs)

        bm25_docs = all_bm25_docs   

        rrf_scores = reciprocal_rank_fusion(vector_docs,bm25_docs)

        # Combine Results
        combined_docs = {}

        for doc in vector_docs + bm25_docs:

            combined_docs[doc.page_content] = doc

        ranked_docs = sorted(
            combined_docs.values(),
            key=lambda d: rrf_scores[d.page_content],
            reverse=True
)
        docs = ranked_docs[:5]

        print("\nTOP RETRIEVED DOCS")

        for i, doc in enumerate(docs, start=1):

            print(
                f"\nRank {i}"
            )

            print(
                f"Source: {doc.metadata.get('source')}"
            )

            print(
                doc.page_content[:200]
            )

        # Format context
        context = format_docs(docs)

        # Generate answer
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        print("\nCONTEXT SENT TO LLM:")
        print(context)

        answer = chain.invoke({
            "history": history,
            "context": context,
            "question": question
        })

        sources = {}

        for doc in docs:

            source = doc.metadata.get("source", "Unknown")

            page = doc.metadata.get("page",0) + 1

            if source not in sources:
                sources[source] = []

            sources[source].append(page)

        citations = []

        for source, pages in sources.items():

            pages = sorted(set(pages))

            if len(pages) == 1:

                citations.append(
                    f"{source} (Page {pages[0]})")

            else:

                page_list = ", ".join(str(page)for page in pages)

                citations.append(f"{source} (Pages {page_list})")

        # Return BOTH answer and docs
        return {
            "answer": answer,
            "docs": docs,
            "citations": citations
        }

    return rag_pipeline