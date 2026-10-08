import os
 
import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
 
 
load_dotenv()
 
 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
 
 
# ==================================================
# SPLIT TEXT INTO OVERLAPPING CHUNKS
# ==================================================
 
def split_text_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
 
    pieces = []
 
    start = 0
    text_length = len(text)
 
    while start < text_length:
 
        end = min(
            start + chunk_size,
            text_length
        )
 
        # DO NOT STRIP
        piece = text[start:end]
 
        if len(piece.strip()) >= 60:
            pieces.append(piece)
 
        if end == text_length:
            break
 
        start += chunk_size - overlap
 
    return pieces
 
 
# ==================================================
# LOAD AND CHUNK DOCUMENT
# ==================================================
 
def load_and_chunk_document(document_path):
 
    reader = PdfReader(document_path)
 
    full_text = ""
 
    for page in reader.pages:
 
        text = page.extract_text() or ""
 
        full_text += text + "\n"
 
    lines = full_text.split("\n")
 
    sections = []
 
    current_section = None
    current_text = ""
 
    for line in lines:
 
        line = line.strip()
 
        if not line:
            continue
 
        is_heading = False
 
        parts = line.split(".", 1)
 
        if len(parts) == 2:
 
            if parts[0].isdigit():
 
                heading_text = parts[1].strip()
 
                if heading_text:
 
                    first_char = heading_text[0]
 
                    if first_char.isupper():
                        is_heading = True
 
        if is_heading:
 
            if current_section is not None:
 
                sections.append(
                    {
                        "section": current_section,
                        "text": current_text.strip()
                    }
                )
 
            current_section = line
            current_text = ""
 
        else:
 
            if current_text:
                current_text += " "
 
            current_text += line
 
    if current_section is not None:
 
        sections.append(
            {
                "section": current_section,
                "text": current_text.strip()
            }
        )
 
    chunks = []
 
    for section in sections:
 
        section_chunks = split_text_into_chunks(
            section["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
 
        for chunk in section_chunks:
 
            chunks.append(
                {
                    "section": section["section"],
                    "text": chunk
                }
            )
 
    return chunks
 
 
# ==================================================
# CREATE EMBEDDING MODEL
# ==================================================
 
def get_embedding_model():
 
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
 
 
# ==================================================
# CREATE RETRIEVAL COLLECTION
# ==================================================
 
def create_retrieval_collection(
    chroma_client
):
 
    collection = chroma_client.get_or_create_collection(
        name="retrievals",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )
 
    return collection
 
 
# ==================================================
# STORE DOCUMENT CHUNKS
# ==================================================
 
def store_chunks(
    collection,
    chunks,
    embedding_model
):
 
    documents = []
    metadatas = []
    ids = []
 
    for i, chunk in enumerate(chunks):
 
        documents.append(
            chunk["text"]
        )
 
        metadatas.append(
            {
                "section": chunk["section"]
            }
        )
 
        ids.append(
            str(i)
        )
 
    embeddings = embedding_model.encode(
        documents
    )
 
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ]
    )
 
 
# ==================================================
# RETRIEVE SECTIONS
# ==================================================
 
def retrieve_sections(
    question,
    embedding_model,
    collection
):
 
    query_embedding = embedding_model.encode(
        [question]
    )[0]
 
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K
    )
 
    matches = []
 
    documents = results.get(
        "documents",
        [[]]
    )[0]
 
    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]
 
    distances = results.get(
        "distances",
        [[]]
    )[0]
 
    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
 
        matches.append(
            {
                "chunk": text,
                "section": metadata["section"],
                "similarity": float(
                    1 - distance
                )
            }
        )
 
    matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )
 
    return matches
 
 
# ==================================================
# AUGMENT PROMPT
# ==================================================
 
def augment_prompt(
    question,
    retrieved_chunks
):
 
    context = ""
 
    for item in retrieved_chunks:
 
        context += (
            f"[Section: {item['section']}]\n"
            f"{item['chunk']}\n\n"
        )
 
    prompt = f"""
You are an employee policy assistant.
 
Answer only using the context below.
 
Context:
 
{context}
 
Question:
{question}
 
Answer:
"""
 
    return prompt.strip()
 
 
# ==================================================
# GENERATE ANSWER
# ==================================================
 
def generate_answer(
    prompt,
    client
):
 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
 
    return response.text.strip()
 
 
# ==================================================
# POLICY QA PIPELINE
# ==================================================
 
def policy_qa_pipeline(
    question,
    document_path
):
 
    # Load document
    chunks = load_and_chunk_document(
        document_path
    )
 
    # Create SentenceTransformer model
    embedding_model = get_embedding_model()
 
    # Create Chroma client
    chroma_client = chromadb.EphemeralClient()
 
    # Create collection
    collection = create_retrieval_collection(
        chroma_client
    )
 
    # Store document embeddings
    store_chunks(
        collection,
        chunks,
        embedding_model
    )
 
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_sections(
        question,
        embedding_model,
        collection
    )
 
    # Build augmented prompt
    prompt = augment_prompt(
        question,
        retrieved_chunks
    )
 
    # Gemini client
    client = genai.Client(
        api_key=os.getenv(
            "GEMINI_API_KEY"
        )
    )
 
    # Generate answer
    answer = generate_answer(
        prompt,
        client
    )
 
    # Unique sources in retrieval order
    sources = []
 
    for item in retrieved_chunks:
 
        section = item["section"]
 
        if section not in sources:
            sources.append(section)
 
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "answer": answer
    }
 
 
# ==================================================
# MAIN
# ==================================================
 
def main():
 
    document_path = (
        "data/employee_policy_handbook.pdf"
    )
 
    while True:
 
        question = input(
            "\nAsk a question (exit to quit): "
        )
 
        if question.lower() == "exit":
            break
 
        try:
 
            result = policy_qa_pipeline(
                question,
                document_path
            )
 
            print("\nSources:")
            print(
                result["sources"]
            )
 
            print("\nRetrieved Chunks:")
 
            for item in result[
                "retrieved_chunks"
            ]:
 
                print(
                    f"\nSection: "
                    f"{item['section']}"
                )
 
                print(
                    f"Similarity: "
                    f"{item['similarity']:.4f}"
                )
 
                print(
                    f"Chunk: "
                    f"{item['chunk']}"
                )
 
            print("\nAnswer:")
 
            print(
                result["answer"]
            )
 
        except Exception as error:
 
            print(
                f"\nError: {error}"
            )
 
 
if __name__ == "__main__":
    main()
 