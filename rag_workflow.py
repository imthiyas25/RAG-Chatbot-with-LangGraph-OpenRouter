import os
from typing import List, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[str]
    context: str
    response: str
    steps: List[str]


class SimpleVectorStore:
    """Simple vector store using TF-IDF for document retrieval."""

    def __init__(self, csv_path: str, text_column: str):
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path).head(1000)
        self.text_column = text_column
        self.documents = self.df[text_column].fillna("").astype(str).tolist()
        print(f"Loaded {len(self.documents)} documents")
        print("Building vector index...")
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print("Vector index ready")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]


vector_store = None


def initialize_vector_store():
    global vector_store
    csv_path = "./data/complaints.csv"
    text_column = "narrative"

    if not os.path.exists(csv_path):
        data_dir = "./data"
        data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if not data_files:
            raise FileNotFoundError("No CSV file found in ./data/")
        csv_path = os.path.join(data_dir, data_files[0])
        print(f"Using dataset: {csv_path}")
        df_temp = pd.read_csv(csv_path, nrows=5)
        print(f"Available columns: {list(df_temp.columns)}")
        text_columns = [
            col
            for col in df_temp.columns
            if df_temp[col].dtype == "object" and df_temp[col].notna().any()
        ]
        if text_columns:
            text_column = text_columns[0]

    vector_store = SimpleVectorStore(csv_path, text_column)


def process_query(state: RAGState) -> dict:
    print("Processing query...")
    return {"query": state["query"], "steps": state.get("steps", []) + ["query_processed"]}


def retrieve_documents(state: RAGState) -> dict:
    print("Retrieving relevant documents...")
    docs = vector_store.search(state["query"], top_k=3)
    context = "\n\n".join([f"Document {i+1}:\n{doc[:500]}" for i, doc in enumerate(docs)])
    return {
        "retrieved_docs": docs,
        "context": context,
        "steps": state["steps"] + ["documents_retrieved"],
    }


def generate_response(state: RAGState) -> dict:
    print("Generating response with OpenRouter...")
    prompt = f"""Based on the following context, answer the user's question.

Context:
{state['context']}

Question: {state['query']}

Provide a clear, concise answer based only on the information in the context above. If the context doesn't contain relevant information, say so."""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return {
        "response": response.choices[0].message.content,
        "steps": state["steps"] + ["response_generated"],
    }


def format_output(state: RAGState) -> dict:
    print("Formatting output...")
    formatted_response = f"""
{'=' * 60}
QUESTION: {state['query']}
{'=' * 60}

RETRIEVED DOCUMENTS:
{'-' * 60}
{state['context'][:500]}...
{'-' * 60}

ANSWER:
{state['response']}
{'=' * 60}
"""
    return {"response": formatted_response, "steps": state["steps"] + ["output_formatted"]}


def create_rag_workflow():
    workflow = StateGraph(RAGState)
    workflow.add_node("process_query", process_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("format_output", format_output)
    workflow.set_entry_point("process_query")
    workflow.add_edge("process_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", "format_output")
    workflow.add_edge("format_output", END)
    return workflow.compile()


def run_example_queries(app):
    queries = [
        "What are common customer complaints?",
        "Tell me about product issues",
    ]
    for query in queries:
        print(f"\n{'#' * 60}")
        print(f"Running RAG for: '{query}'")
        print(f"{'#' * 60}\n")
        result = app.invoke(
            {"query": query, "retrieved_docs": [], "context": "", "response": "", "steps": []}
        )
        print(result["response"])
        print(f"\nExecution steps: {' → '.join(result['steps'])}")


def interactive_mode():
    initialize_vector_store()
    app = create_rag_workflow()
    print("\n" + "=" * 60)
    print("RAG Interactive Mode (type 'quit' to exit)")
    print("=" * 60 + "\n")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        result = app.invoke(
            {"query": query, "retrieved_docs": [], "context": "", "response": "", "steps": []}
        )
        print(result["response"])


def main(interactive: bool = True):
    print("=" * 60)
    print("RAG System with Kaggle Data + LangGraph + OpenRouter")
    print("=" * 60)
    print()

    initialize_vector_store()
    print()

    print("Building RAG workflow...")
    app = create_rag_workflow()
    print("Workflow ready")
    print()

    if interactive:
        print("Launching chatbot mode...")
        print("Type 'quit' to exit.\n")
        while True:
            query = input("You: ")
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            result = app.invoke(
                {
                    "query": query,
                    "retrieved_docs": [],
                    "context": "",
                    "response": "",
                    "steps": [],
                }
            )
            print(result["response"])
    else:
        run_example_queries(app)


if __name__ == "__main__":
    main(interactive=True)

