import os
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from chromadb.utils import embedding_functions

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
        overlap=CHUNK_OVERLAP):

    pieces = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # DO NOT STRIP
        piece = text[start:end]

        if len(piece.strip()) >= 60:
            pieces.append(piece)

        if end == text_length:
            break

        start += (chunk_size - overlap)

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
# RETRIEVE SECTIONS
# ==================================================

def retrieve_sections(question, embedding_model, collection):

    results = collection.query(
        query_texts=[question],
        n_results=TOP_K
    )

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        matches.append(
            {
                "chunk": text,
                "section": metadata["section"],
                "similarity": float(1 - distance)
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

def augment_prompt(question, retrieved_chunks):

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

def generate_answer(prompt, client):

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text.strip()


# ==================================================
# POLICY QA PIPELINE
# ==================================================

def policy_qa_pipeline(question, document_path):

    chunks = load_and_chunk_document(document_path)

    chroma_client = chromadb.EphemeralClient()

    default_ef = embedding_functions.DefaultEmbeddingFunction()

    collection = chroma_client.get_or_create_collection(
        name="policy_collection",
        embedding_function=default_ef,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):

        documents.append(chunk["text"])

        metadatas.append(
            {
                "section": chunk["section"]
            }
        )

        ids.append(str(i))

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    retrieved_chunks = retrieve_sections(
        question,
        None,
        collection
    )

    prompt = augment_prompt(
        question,
        retrieved_chunks
    )

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    answer = generate_answer(
        prompt,
        client
    )

    sources = []

    for item in retrieved_chunks:

        if item["section"] not in sources:
            sources.append(item["section"])

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

    document_path = "data/employee_policy_handbook.pdf"

    while True:

        question = input("\nAsk a question (exit to quit): ")

        if question.lower() == "exit":
            break

        result = policy_qa_pipeline(
            question,
            document_path
        )

        print("\nSources:")
        print(result["sources"])

        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()