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

⏰ NOTE: This step takes ~20-30 minutes. Start it early!
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")   # suppress RAGAS deprecation warnings

from pathlib import Path

# ── 1. Imports ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
import numpy as np


# ── 2. QA pairs with ground-truth answers ───────────────────────────────────
# Each entry has a "question" and a "reference" (ground-truth answer).
# You need BOTH to compute context_recall.
QA_PAIRS = [
    {"question": "What are the three main types of machine learning?",
     "reference": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."},
    {"question": "What is overfitting in machine learning?",
     "reference": "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on new data."},
    {"question": "Explain the bias-variance tradeoff.",
     "reference": "High bias means underfitting; high variance means overfitting. The optimal model balances both to minimize total error."},
    {"question": "How does regularization prevent overfitting?",
     "reference": "L1 and L2 regularization add penalty terms to the loss function to discourage overly complex models."},
    {"question": "What is cross-validation?",
     "reference": "Cross-validation splits data into multiple folds to estimate model performance more reliably."},
    {"question": "What is backpropagation?",
     "reference": "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule of calculus."},
    {"question": "What are Convolutional Neural Networks primarily used for?",
     "reference": "CNNs are specialized for processing grid-like data such as images."},
    {"question": "How do LSTM networks address the vanishing gradient problem?",
     "reference": "LSTMs use gating mechanisms that control information flow through the network."},
    {"question": "What activation functions are commonly used in neural networks?",
     "reference": "Common activation functions include ReLU, sigmoid, and tanh."},
    {"question": "What is the role of pooling layers in CNNs?",
     "reference": "Pooling layers reduce spatial dimensions, decreasing computational complexity while retaining important features."},
    {"question": "What is the transformer architecture?",
     "reference": "The transformer uses self-attention mechanisms and processes entire sequences in parallel, introduced in 'Attention Is All You Need' (2017)."},
    {"question": "What are word embeddings?",
     "reference": "Word embeddings represent words as dense vectors where semantically similar words are geometrically close."},
    {"question": "What is transfer learning in NLP?",
     "reference": "Pre-training large models on massive corpora then fine-tuning on downstream tasks."},
    {"question": "How does BERT handle language understanding?",
     "reference": "BERT uses bidirectional transformer training with masked language modeling and next sentence prediction."},
    {"question": "What is self-attention in transformers?",
     "reference": "Self-attention allows models to weigh the importance of different words relative to each other in a sequence."},
    {"question": "What is GPT and how is it trained?",
     "reference": "GPT uses autoregressive training to predict the next token given previous tokens, trained on massive text datasets."},
    {"question": "What is instruction tuning?",
     "reference": "Fine-tuning pre-trained LLMs on instruction-following datasets to improve alignment with human intent."},
    {"question": "What is RLHF?",
     "reference": "Reinforcement Learning from Human Feedback uses human preferences to align LLMs to be helpful, harmless, and honest."},
    {"question": "What is chain-of-thought prompting?",
     "reference": "Chain-of-thought prompting encourages LLMs to show reasoning step by step, improving performance on complex tasks."},
    {"question": "What is the context length of GPT-4?",
     "reference": "GPT-4 supports up to 128K tokens of context."},
    {"question": "What is Retrieval-Augmented Generation?",
     "reference": "RAG combines generative LLMs with retrieval from external knowledge bases to produce grounded, up-to-date answers."},
    {"question": "What are the main components of a RAG pipeline?",
     "reference": "A retriever that searches a document store and a generator (LLM) that produces answers from the query and retrieved passages."},
    {"question": "What is dense retrieval?",
     "reference": "Dense retrieval uses neural embeddings to encode queries and documents, determining relevance by cosine similarity."},
    {"question": "Why is chunking strategy important in RAG?",
     "reference": "Chunking affects retrieval precision and context window fit; options include fixed-size, semantic, and hierarchical chunking."},
    {"question": "What advanced RAG techniques exist beyond basic retrieval?",
     "reference": "Re-ranking, query expansion, HyDE, and iterative retrieval for multi-hop questions."},
    {"question": "What are vector databases used for?",
     "reference": "Storing and querying high-dimensional vector embeddings for fast similarity search."},
    {"question": "What is FAISS?",
     "reference": "FAISS is a library for efficient similarity search supporting exact and approximate nearest neighbor algorithms."},
    {"question": "How do text embeddings capture semantic meaning?",
     "reference": "Text embeddings convert text into numerical vectors where semantically similar texts produce geometrically close vectors."},
    {"question": "What is HNSW?",
     "reference": "HNSW builds a hierarchical graph for logarithmic-complexity approximate nearest neighbor search."},
    {"question": "What is hybrid search in vector databases?",
     "reference": "Hybrid search combines dense vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion."},
    {"question": "What is LangChain?",
     "reference": "LangChain is an open-source framework for building LLM applications with abstractions for chains, agents, and data connections."},
    {"question": "What is LangChain Expression Language (LCEL)?",
     "reference": "LCEL is a declarative way to compose chains using the pipe operator, supporting streaming, async, and batching."},
    {"question": "What is LangGraph?",
     "reference": "LangGraph extends LangChain for stateful multi-actor applications as directed graphs, supporting cycles."},
    {"question": "What memory types does LangChain support?",
     "reference": "ConversationBufferMemory, ConversationSummaryMemory, ConversationWindowMemory, and vector store memory."},
    {"question": "What are LangChain retrievers?",
     "reference": "Retrievers fetch relevant documents from a data source given a query, supporting vector stores and BM25."},
    {"question": "What is LangSmith?",
     "reference": "LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications through automatic tracing."},
    {"question": "What information do LangSmith traces capture?",
     "reference": "Inputs, outputs, latency, token usage, and errors for every component in a chain."},
    {"question": "What is the LangSmith Prompt Hub?",
     "reference": "A repository for storing, versioning, and sharing prompt templates, supporting A/B testing and rollback."},
    {"question": "How does LangSmith help monitor production LLM applications?",
     "reference": "LangSmith provides latency percentiles, error rates, token costs, and feedback annotations."},
    {"question": "What are LangSmith datasets used for?",
     "reference": "Systematic evaluation using example inputs and expected outputs to compare model versions or prompt changes."},
    {"question": "What is RAGAS?",
     "reference": "RAGAS is an open-source framework for evaluating RAG pipelines using LLM-based reference-free metrics."},
    {"question": "How does RAGAS compute faithfulness?",
     "reference": "By extracting claims from the answer and checking whether each claim can be inferred from the retrieved context."},
    {"question": "What is answer relevancy in RAGAS?",
     "reference": "Measures how well the answer addresses the original question by generating synthetic questions from the answer."},
    {"question": "What is context recall in RAGAS?",
     "reference": "Measures how well the retrieved context covers the information needed, evaluated against a ground truth reference."},
    {"question": "What inputs does RAGAS evaluation require?",
     "reference": "User queries, generated answers, retrieved contexts (list of passages), and optionally reference answers."},
    {"question": "What is Guardrails AI?",
     "reference": "An open-source framework for adding validation and safety checks to LLM outputs with configurable on-fail actions."},
    {"question": "What is PII and why is it important to detect in LLM responses?",
     "reference": "PII is Personally Identifiable Information; exposing it can violate GDPR and HIPAA privacy regulations."},
    {"question": "What does structured output validation ensure?",
     "reference": "That LLM responses conform to expected schemas such as JSON, detecting and fixing invalid formats."},
    {"question": "What is Constitutional AI?",
     "reference": "A technique where a model is given principles to follow and uses self-critique to improve its responses."},
    {"question": "What are common AI safety concerns with LLMs?",
     "reference": "Hallucination, toxicity, bias, PII leakage, and jailbreaking attacks."},
]


# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)

PROMPTS = {
    "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")]),
    "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")]),
}


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
def build_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    text = (Path(__file__).parent / "data" / "dataset.txt").read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.

    IMPORTANT: return contexts as a LIST of strings, not a joined string!
    RAGAS needs individual passage strings to compute context_recall.

    Returns: {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # ← list of strings!
    ctx_str  = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt    = PROMPTS[prompt_version]

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
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
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

    llm_eval = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
    )
    emb_eval = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

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

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).resolve().parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {report_path}")



if __name__ == "__main__":
    main()
