"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")   # suppress RAGAS/LangChain deprecation warnings

from pathlib import Path
import numpy as np

# --- Import our custom configuration loader ---
import config

# --- Import RAGAS evaluate + dataset classes ---
from ragas import evaluate, EvaluationDataset, SingleTurnSample

# --- Import the 4 metric instances (RAGAS 0.4.x) ---
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# --- Import LangChain components ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 2. QA pairs with ground-truth answers ───────────────────────────────────
# Imported from our shared module to ensure exact consistency
from qa_pairs import QA_PAIRS

# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
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

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
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


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.

    IMPORTANT: return contexts as a LIST of strings, not a joined string!
    RAGAS needs individual passage strings to compute context_recall and context_precision.

    Returns: {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # ← list of strings!
    ctx_str  = "\n\n".join(contexts)

    # Run the chain
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=config.DEFAULT_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
        temperature=0,
    )
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],   # must be list[str]
        })
        print(f"  [{i:02d}/50] Answered: {qa['question'][:50]}...")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.
    Each SingleTurnSample needs:
      user_input         → the question
      response           → the generated answer
      retrieved_contexts → list[str] of retrieved passages
      reference          → the ground-truth answer
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    # Initialize evaluator LLM and Embeddings using direct OpenAI models
    llm_eval = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    emb_eval = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config.OPENAI_API_KEY,
    )

    # Run Ragas evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Ragas returns metric scores per sample, so we extract and take the mean
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]           # list of floats
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Build vectorstore
    vectorstore = build_vectorstore()

    # Collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 RAGAS COMPARISON TABLE (V1 vs V2):")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1 Better" if s1 > s2 else "← V2 Better"
        print(f"  {metric:20s}: V1 = {s1:.4f} | V2 = {s2:.4f} | {winner}")
    print("=" * 60)

    # Check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"✅ Target met: best faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️ Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # Save JSON report to data/ragas_report.json
    # Create parent directories if they don't exist
    os.makedirs("data", exist_ok=True)
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    
    # Save to both standard report locations
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
    print("💾 Saved data/ragas_report.json")
    
    os.makedirs("evidence", exist_ok=True)
    Path("evidence/03_ragas_report.json").write_text(json.dumps(report, indent=2))
    print("💾 Saved evidence/03_ragas_report.json")


if __name__ == "__main__":
    main()
