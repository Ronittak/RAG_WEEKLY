import numbers
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DOCUMENT_PATH = "data/employee_policy_handbook.pdf"
HEADING_PATTERN = r"(?m)^(\d{1,2}\.\s+[A-Z].*)$"

CHUNK_SIZE = 500
CHUNK_STEP = 420
MIN_CHUNK_LENGTH = 60
EXPECTED_OVERLAP = CHUNK_SIZE - CHUNK_STEP
MIN_COVERAGE = 0.75
N_RESULTS = 4

RETRIEVAL_COLLECTION = "wa4_retrieval_check"
LEAVE_QUESTION = "How many days of earned leave can be carried forward to the next year?"
LEAVE_SECTION = "3. Leave Entitlement"
REMOTE_QUESTION = "How many days per week may eligible employees work remotely?"
REMOTE_SECTION = "4. Remote Work and Hybrid Policy"

SAMPLE_PASSAGES = [
    (LEAVE_SECTION,
     "Up to 12 days of earned leave may be carried forward to the next year. Any balance "
     "above this limit lapses on 31 December. Casual leave cannot be carried forward."),
    (REMOTE_SECTION,
     "Confirmed employees in roles approved for hybrid work may work remotely for up to "
     "2 days per week with prior manager approval."),
    ("5. Performance Appraisal and Increments",
     "Performance is reviewed twice each year, in April and October. Ratings use a "
     "5-point scale, and annual increments normally range from 0% to 12% of fixed annual pay."),
    ("7. Travel and Expense Reimbursement",
     "Business travel requires written manager approval at least 5 working days before "
     "departure. Claims must be submitted within 15 calendar days of return."),
    ("1. Employment and Probation",
     "New employees serve a probation period of 6 months from their joining date. "
     "Confirmation is issued in writing by Human Resources after satisfactory performance."),
    ("9. Information Security and Device Use",
     "Employees must use unique passwords of at least 12 characters. Lost devices must be "
     "reported to IT and the manager within 1 hour of discovery."),
]

PROMPT_QUESTION = "What happens to earned leave above the carry-forward limit?"
PROMPT_CHUNKS = [
    {"chunk": "Any earned leave balance above the carry-forward limit lapses on 31 December.",
     "section": LEAVE_SECTION, "similarity": 0.68},
    {"chunk": "Up to 12 days of earned leave may be carried forward to the next year.",
     "section": LEAVE_SECTION, "similarity": 0.61},
    {"chunk": "Claims must be submitted within 15 calendar days of return from business travel.",
     "section": "7. Travel and Expense Reimbursement", "similarity": 0.44},
]

SECOND_PROMPT_QUESTION = "How many remote work days are allowed per week?"
SECOND_PROMPT_CHUNKS = [
    {"chunk": "Eligible employees may work remotely for up to 2 days per week.",
     "section": REMOTE_SECTION, "similarity": 0.73},
]

SPY_PROMPT = ("You are an employee policy assistant. Using only the excerpt below, answer "
              "the question.\n\n[Section: 3. Leave Entitlement]\nUp to 12 days of earned leave "
              "may be carried forward.\n\nQuestion: How many earned leave days can I carry forward?")
SPY_REPLY = "\n  Up to 12 days of earned leave may be carried forward, per section 3. Leave Entitlement.  \n"

LIVE_PROMPT_A = "In one short sentence, explain what earned leave is."
LIVE_PROMPT_B = "In one short sentence, explain what business travel reimbursement is."

PIPELINE_QUESTION = LEAVE_QUESTION


def normalise(text):
    """Collapse whitespace so comparisons ignore line-break handling choices."""
    return " ".join(str(text).split())


def flatten_contents(contents):
    """Reduce whatever was passed as `contents` to plain text, list form included."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (list, tuple)):
        return " ".join(flatten_contents(part) for part in contents)
    return str(getattr(contents, "text", contents))


# --------------------------------------------------------------------------------------
# Every piece of shared setup runs through cached(). Work is done once, and any failure is
# raised inside the calling test, so each test reports PASSED or FAILED and never ERROR or
# SKIPPED. There are no skip, skipif or xfail markers anywhere in this file.
# --------------------------------------------------------------------------------------

_CACHE = {}


def cached(name, builder):
    if name not in _CACHE:
        try:
            _CACHE[name] = (True, builder())
        except Exception as exc:
            _CACHE[name] = (False, f"{type(exc).__name__}: {exc}")
    succeeded, payload = _CACHE[name]
    if not succeeded:
        pytest.fail(f"could not build '{name}' -- {payload}", pytrace=False)
    return payload


def build_document_text():
    """Independent extraction of the handbook. Ground truth, uses no student code."""
    from pypdf import PdfReader
    reader = PdfReader(DOCUMENT_PATH)
    return "".join(page.extract_text() for page in reader.pages)


def build_section_bodies():
    """Split the handbook into {heading: body} without touching student code."""
    parts = re.split(HEADING_PATTERN, cached("document_text", build_document_text))
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def build_chunks():
    from main import load_and_chunk_document
    return load_and_chunk_document(DOCUMENT_PATH)


def build_retrieval_store():
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [passage for _, passage in SAMPLE_PASSAGES]
    embeddings = embedding_model.encode(texts)

    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(name=RETRIEVAL_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=RETRIEVAL_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    collection.add(
        ids=[str(i) for i in range(len(SAMPLE_PASSAGES))],
        embeddings=[embedding.tolist() for embedding in embeddings],
        documents=texts,
        metadatas=[{"section": section} for section, _ in SAMPLE_PASSAGES],
    )
    return embedding_model, collection


def build_retrievals():
    from main import retrieve_sections
    embedding_model, collection = cached("retrieval_store", build_retrieval_store)
    return {
        "leave": retrieve_sections(LEAVE_QUESTION, embedding_model, collection),
        "remote": retrieve_sections(REMOTE_QUESTION, embedding_model, collection),
    }


def build_prompts():
    from main import augment_prompt
    return (augment_prompt(PROMPT_QUESTION, PROMPT_CHUNKS),
            augment_prompt(SECOND_PROMPT_QUESTION, SECOND_PROMPT_CHUNKS))


class _SpyResponse:
    def __init__(self, text):
        self.text = text


class _SpyModels:
    def __init__(self, owner):
        self._owner = owner

    def generate_content(self, model=None, contents=None, **kwargs):
        self._owner.received = {"model": model, "contents": contents}
        return _SpyResponse(SPY_REPLY)


class SpyClient:
    """Stands in for genai.Client so the call itself can be inspected, offline."""

    def __init__(self):
        self.received = None
        self.models = _SpyModels(self)


def build_spy_call():
    from main import generate_answer
    client = SpyClient()
    returned = generate_answer(SPY_PROMPT, client)
    return client, returned


def build_live_answers():
    from main import generate_answer
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return (generate_answer(LIVE_PROMPT_A, client),
            generate_answer(LIVE_PROMPT_B, client))


def build_pipeline_result():
    from main import policy_qa_pipeline
    return policy_qa_pipeline(PIPELINE_QUESTION, DOCUMENT_PATH)


# --------------------------------------------------------------------------------------
# load_and_chunk_document  (5 tests)
# --------------------------------------------------------------------------------------

def test_chunking_covers_the_whole_document():
    """Chunks must be a list of dicts that between them cover the handbook's own text."""
    chunks = cached("chunks", build_chunks)
    document = normalise(cached("document_text", build_document_text))

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, dict)

    # Mark the stretches of the real document that the chunks account for. Text that is
    # not found in the document contributes nothing, so invented chunks cannot inflate this.
    covered = bytearray(len(document))
    for chunk in chunks:
        text = normalise(chunk["text"])
        start = document.find(text)
        if start >= 0:
            covered[start:start + len(text)] = b"\x01" * len(text)

    assert sum(covered) >= MIN_COVERAGE * len(document)


def test_each_chunk_is_labelled_with_a_real_heading():
    """Every chunk must expose section and text, labelled with a heading from the document."""
    chunks = cached("chunks", build_chunks)
    headings = [normalise(h) for h in cached("section_bodies", build_section_bodies)]

    for chunk in chunks:
        assert "section" in chunk and "text" in chunk
        assert isinstance(chunk["section"], str) and isinstance(chunk["text"], str)
        assert len(chunk["text"].strip()) >= MIN_CHUNK_LENGTH

        label = normalise(chunk["section"])
        assert len(label) > 0
        assert any(label == h or label in h or h in label for h in headings)


def test_chunk_text_is_taken_verbatim_from_the_document():
    """Chunk text must be genuine handbook text, not paraphrased or fabricated."""
    chunks = cached("chunks", build_chunks)
    document = normalise(cached("document_text", build_document_text))

    for chunk in chunks:
        assert normalise(chunk["text"]) in document


def test_chunks_never_span_two_sections():
    """Splitting must happen before windowing, so a chunk sits inside one section body."""
    chunks = cached("chunks", build_chunks)
    bodies = cached("section_bodies", build_section_bodies)
    lookup = {normalise(heading): normalise(body) for heading, body in bodies.items()}

    assert len(lookup) > 1
    assert len(set(normalise(c["section"]) for c in chunks)) == len(lookup)
    assert len(chunks) > len(lookup)

    for chunk in chunks:
        label = normalise(chunk["section"])
        owner = [body for heading, body in lookup.items()
                 if label == heading or label in heading or heading in label]
        assert len(owner) == 1
        assert normalise(chunk["text"]) in owner[0]


def test_sliding_window_size_and_overlap_are_applied():
    """Chunks must respect the size ceiling, the length floor and the overlap step."""
    chunks = cached("chunks", build_chunks)

    for chunk in chunks:
        assert len(chunk["text"]) <= CHUNK_SIZE
        assert len(chunk["text"].strip()) >= MIN_CHUNK_LENGTH

    compared = 0
    for current, following in zip(chunks, chunks[1:]):
        if current["section"] == following["section"] and len(current["text"]) == CHUNK_SIZE:
            assert current["text"][-EXPECTED_OVERLAP:] == following["text"][:EXPECTED_OVERLAP]
            compared += 1
    assert compared > 0


# --------------------------------------------------------------------------------------
# retrieve_sections  (3 tests)
# --------------------------------------------------------------------------------------

def test_retrieval_returns_stored_passages_in_the_documented_shape():
    """Retrieval must return N_RESULTS dicts whose text really is in the collection."""
    retrieved = cached("retrievals", build_retrievals)["leave"]
    stored = {normalise(passage) for _, passage in SAMPLE_PASSAGES}

    assert isinstance(retrieved, list)
    assert len(retrieved) == N_RESULTS
    for item in retrieved:
        assert isinstance(item, dict)
        assert isinstance(item["chunk"], str)
        assert isinstance(item["section"], str)
        assert isinstance(item["similarity"], numbers.Real)
        assert not isinstance(item["similarity"], bool)
        assert normalise(item["chunk"]) in stored


def test_each_result_keeps_its_own_section_metadata():
    """A passage must come back paired with the section it was indexed under, once only."""
    retrieved = cached("retrievals", build_retrievals)["leave"]
    indexed = {normalise(passage): normalise(section) for section, passage in SAMPLE_PASSAGES}

    for item in retrieved:
        assert indexed[normalise(item["chunk"])] == normalise(item["section"])

    keys = [normalise(item["chunk"]) for item in retrieved]
    assert len(keys) == len(set(keys))


def test_ranking_follows_the_question_that_was_asked():
    """Scores must be ordered, and a different question must surface a different passage."""
    retrievals = cached("retrievals", build_retrievals)
    leave, remote = retrievals["leave"], retrievals["remote"]

    for results in (leave, remote):
        scores = [float(item["similarity"]) for item in results]
        for score in scores:
            assert -1.0 <= score <= 1.0
        assert scores == sorted(scores, reverse=True)

    assert normalise(leave[0]["section"]) == normalise(LEAVE_SECTION)
    assert normalise(remote[0]["section"]) == normalise(REMOTE_SECTION)


# --------------------------------------------------------------------------------------
# augment_prompt  (2 tests)
# --------------------------------------------------------------------------------------

def test_prompt_carries_the_question_and_every_labelled_excerpt():
    """The prompt must contain the question plus every excerpt and its section label."""
    prompt, _ = cached("prompts", build_prompts)
    flat = normalise(prompt)

    assert isinstance(prompt, str)
    assert normalise(PROMPT_QUESTION) in flat
    for item in PROMPT_CHUNKS:
        assert normalise(item["chunk"]) in flat
        assert normalise(item["section"]) in flat

    supplied = len(PROMPT_QUESTION) + sum(len(i["chunk"]) + len(i["section"]) for i in PROMPT_CHUNKS)
    assert len(prompt) > supplied


def test_prompt_is_rebuilt_from_whatever_arguments_it_receives():
    """A different question and excerpt must produce a correspondingly different prompt."""
    first, second = cached("prompts", build_prompts)
    flat_second = normalise(second)

    assert first != second
    assert normalise(SECOND_PROMPT_QUESTION) in flat_second
    assert normalise(SECOND_PROMPT_CHUNKS[0]["chunk"]) in flat_second
    assert normalise(SECOND_PROMPT_CHUNKS[0]["section"]) in flat_second
    assert normalise(PROMPT_CHUNKS[0]["chunk"]) not in flat_second
    assert normalise(PROMPT_QUESTION) not in flat_second


# --------------------------------------------------------------------------------------
# generate_answer  (2 tests)
# --------------------------------------------------------------------------------------

def test_generate_answer_sends_the_prompt_and_trims_the_reply():
    """The client must actually be called with the prompt, and its reply must be stripped."""
    client, returned = cached("spy_call", build_spy_call)

    assert client.received is not None
    assert normalise(SPY_PROMPT) in normalise(flatten_contents(client.received["contents"]))
    assert isinstance(client.received["model"], str)
    assert len(client.received["model"].strip()) > 0
    assert returned == SPY_REPLY.strip()


def test_generate_answer_returns_distinct_live_model_text():
    """Two unrelated prompts must return two different, clean, substantive answers."""
    first, second = cached("live_answers", build_live_answers)

    for answer in (first, second):
        assert isinstance(answer, str)
        assert answer == answer.strip()
        assert len(answer) > 10
    assert first != second


# --------------------------------------------------------------------------------------
# policy_qa_pipeline  (3 tests)
# --------------------------------------------------------------------------------------

def test_pipeline_returns_the_four_documented_fields():
    """The pipeline must return the documented dict, echoing the question unchanged."""
    result = cached("pipeline_result", build_pipeline_result)

    assert isinstance(result, dict)
    for key in ("question", "retrieved_chunks", "sources", "answer"):
        assert key in result

    assert result["question"] == PIPELINE_QUESTION
    assert isinstance(result["sources"], list)
    assert isinstance(result["retrieved_chunks"], list)
    assert len(result["retrieved_chunks"]) == N_RESULTS

    # What comes back must be what was indexed: the chunker's own output, not new text.
    indexed = {normalise(chunk["text"]) for chunk in cached("chunks", build_chunks)}
    for item in result["retrieved_chunks"]:
        assert len(item["chunk"].strip()) >= MIN_CHUNK_LENGTH
        assert normalise(item["chunk"]) in indexed

    assert isinstance(result["answer"], str)
    assert result["answer"] == result["answer"].strip()
    assert len(result["answer"]) > 20
    assert normalise(result["answer"]) != normalise(PIPELINE_QUESTION)


def test_pipeline_retrieved_chunks_trace_back_to_the_handbook():
    """Retrieved excerpts must be real handbook text with valid labels and scores."""
    result = cached("pipeline_result", build_pipeline_result)
    document = normalise(cached("document_text", build_document_text))

    for item in result["retrieved_chunks"]:
        assert normalise(item["chunk"]) in document
        assert normalise(item["section"]) in document
        assert isinstance(item["similarity"], numbers.Real)
        assert -1.0 <= float(item["similarity"]) <= 1.0


def test_pipeline_sources_cite_the_retrieved_sections():
    """Sources must be the unique retrieved headings, in order, and the citations must be true."""
    result = cached("pipeline_result", build_pipeline_result)
    bodies = cached("section_bodies", build_section_bodies)
    lookup = {normalise(heading): normalise(body) for heading, body in bodies.items()}
    order = [item["section"] for item in result["retrieved_chunks"]]

    expected = []
    for section in order:
        if section not in expected:
            expected.append(section)

    assert len(result["sources"]) > 0
    assert len(result["sources"]) == len(set(result["sources"]))
    assert result["sources"] == expected

    # A citation is only worth anything if the cited section really does contain the excerpt.
    for item in result["retrieved_chunks"]:
        label = normalise(item["section"])
        owner = [body for heading, body in lookup.items()
                 if label == heading or label in heading or heading in label]
        assert len(owner) == 1
        assert normalise(item["chunk"]) in owner[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
