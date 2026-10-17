from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about Salesforce CRM using
    only the information provided in the retrieved Salesforce documentation.

    User Query:
    {query}

    Retrieved Salesforce Documentation:
    {retrieved_chunks}

    Rules:
    1. Answer the question using only information explicitly supported by the
      retrieved documentation.

    2. Do not use outside knowledge, assumptions, or information that is not
      present in the retrieved documentation.

    3. Answer the user's specific question directly and completely.

    4. When the documentation provides configuration steps, setup instructions,
      procedures, or troubleshooting steps, present them clearly and in the
      correct order.

    5. Preserve important Salesforce-specific details such as:
      - Feature names
      - Object names
      - Field names
      - Settings
      - Permissions
      - User roles
      - Configuration requirements
      - Limitations
      - Prerequisites
      - Exceptions
      - Supported functionality

    6. Do not invent Salesforce features, configuration options, permissions,
      limitations, or procedures.

    7. If the documentation mentions prerequisites or conditions required for
      a feature or procedure, include them when relevant to the question.

    8. If the retrieved documentation does not contain enough information to
      answer the question, respond exactly with:

      "The available documents do not contain enough information to answer this question."

    9. If the retrieved documentation contains conflicting information, clearly
      identify the conflict rather than choosing an answer based on outside
      knowledge.

    10. Do not include unrelated information from the retrieved documentation.

    11. Keep the response concise, clear, and easy to understand.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
