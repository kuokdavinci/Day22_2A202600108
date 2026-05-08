"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import hashlib
from pathlib import Path

# --- Import our custom configuration loader ---
import config

# --- Import required libraries ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

# Prompt Hub names (change these to your own unique names)
PROMPT_V1_NAME = "rag-prompt-v1-kuokdavinci"
PROMPT_V2_NAME = "rag-prompt-v2-kuokdavinci"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    Use: client.push_prompt(name, object=template, description="...")
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e) or "Conflict" in str(e):
            print(f"✅ V1 on Hub is already up-to-date (no changes detected).")
        else:
            print(f"⚠️ Failed to push V1 to Hub: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e) or "Conflict" in str(e):
            print(f"✅ V2 on Hub is already up-to-date (no changes detected).")
        else:
            print(f"⚠️ Failed to push V2 to Hub: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️ Using local fallback for '{PROMPT_V1_NAME}' due to: {e}")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️ Using local fallback for '{PROMPT_V2_NAME}' due to: {e}")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    even hash → PROMPT_V1_NAME
    odd  hash → PROMPT_V2_NAME
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    text = Path("data/knowledge_base.txt").read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )
    return FAISS.from_texts(chunks, embeddings)


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}
    """
    # Retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run the chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # Create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # Push both prompts to LangSmith Hub
    print("\n[1/4] Uploading prompts to LangSmith Hub...")
    push_prompts_to_hub(client)

    # Pull both prompts from Hub
    print("\n[2/4] Retrieving prompts from LangSmith Hub...")
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever, and LLM
    print("\n[3/4] Initializing RAG assets...")
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )

    # Loop over all 50 questions with A/B routing
    from qa_pairs import QA_PAIRS
    SAMPLE_QUESTIONS = [item["question"] for item in QA_PAIRS]

    
    counts = {"v1": 0, "v2": 0}
    
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        
        counts[version_tag] += 1
        
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] Q: {question[:55]}...")
        print(f"     A: {result['answer'][:90]}...\n")

    print("=" * 60)
    print("📊 A/B Routing Summary:")
    print(f"   Prompt V1 (Concise)   : {counts['v1']} queries")
    print(f"   Prompt V2 (Structured): {counts['v2']} queries")
    print("=" * 60)
    print("✅ Step 2 execution completed successfully!")


if __name__ == "__main__":
    main()
