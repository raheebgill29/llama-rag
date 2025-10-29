import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    # Absolute path to your PDF document
    pdf_path = r"C:\Users\rahee\Downloads\llama\constitution of pakistan.pdf"

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # Require OpenAI API key via environment variable (do not hardcode secrets)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set. Please set it and rerun.")
        return

    # Configure LlamaIndex to use OpenAI LLM and embeddings
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Minimal changes from your original snippet: just point the reader to the exact PDF file
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Your original example query string retained
    response = query_engine.query("what is the 5 point in constitution?")
    print(response)


if __name__ == "__main__":
    main()