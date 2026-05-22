"""
NovaDesk AI Support System — Hybrid Retrieval Pipeline

Production-style retrieval orchestration combining:
  • Deterministic FAQ exact matching
  • BM25 keyword retrieval  
  • Semantic vector retrieval (sentence-transformers + FAISS)
  • Personalized memory-aware retrieval
  • Confidence-based evaluation

Designed for reliability, bounded probabilistic behavior, and hallucination reduction.
LangGraph-compatible modular architecture.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import difflib

import numpy as np

# Third-party dependencies (required in requirements.txt)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  WARNING: sentence-transformers not installed. Install via: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  WARNING: faiss-cpu not installed. Install via: pip install faiss-cpu")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  WARNING: rank-bm25 not installed. Install via: pip install rank-bm25")


# ──────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.65
TOP_K_BM25 = 5
TOP_K_SEMANTIC = 5
TOP_K_MEMORY = 3
BM25_MIN_SCORE = 0.1


# ──────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────────────

class RetrievalSource(str, Enum):
    """Possible sources of retrieved context."""
    FAQ_EXACT = "faq_exact"
    BM25 = "bm25"
    SEMANTIC = "semantic"
    MEMORY = "memory"
    MULTI = "multi"
    NONE = "none"


class DocumentType(str, Enum):
    """Document source types."""
    FAQ = "faq"
    PRICING = "pricing"
    REFUND_POLICY = "refund_policy"
    ESCALATION_RULE = "escalation_rule"
    EDGE_CASE = "edge_case"
    CONVERSATION_FLOW = "conversation_flow"


@dataclass
class Document:
    """Normalized document structure for unified retrieval."""
    id: str
    source_type: DocumentType
    category: str
    content: str
    metadata: Dict[str, Any]
    tokens: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding numpy arrays."""
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d.pop("tokens", None)
        d.pop("embedding", None)
        return d


@dataclass
class RetrievalResult:
    """Structured output from hybrid retrieval."""
    query: str
    retrieval_source: str
    retrieval_confidence: float
    requires_clarification: bool
    retrieved_documents: List[Document]
    memory_context: List[str]
    final_context: str
    retrieval_scores: Dict[str, List[float]]
    evaluation_details: Dict[str, Any]


# ──────────────────────────────────────────────────────────────
# DOCUMENT LOADING LAYER
# ──────────────────────────────────────────────────────────────

def load_json_file(filepath: Path) -> Dict:
    """Safely load and validate JSON files."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Data file not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error in {filepath}: {e}")
        return {}


def normalize_faq_documents() -> List[Document]:
    """Load FAQs and normalize to Document structure."""
    filepath = DATA_DIR / "faqs.json"
    data = load_json_file(filepath)
    docs = []
    
    for faq in data.get("faqs", []):
        doc = Document(
            id=faq.get("id", ""),
            source_type=DocumentType.FAQ,
            category=faq.get("category", "general"),
            content=f"Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}",
            metadata={
                "question": faq.get("question", ""),
                "answer": faq.get("answer", ""),
                "tags": faq.get("tags", []),
                "priority": faq.get("priority", "medium"),
            }
        )
        docs.append(doc)
    
    return docs


def normalize_pricing_documents() -> List[Document]:
    """Load pricing plans and normalize to Document structure."""
    filepath = DATA_DIR / "pricing_plans.json"
    data = load_json_file(filepath)
    docs = []
    
    for plan in data.get("plans", []):
        content = f"Plan: {plan.get('name', '')}\n{plan.get('description', '')}\n"
        content += f"Monthly: ${plan.get('pricing', {}).get('monthly', 0)}\n"
        content += f"Features: {', '.join(plan.get('features', [])[:5])}"
        
        doc = Document(
            id=plan.get("id", ""),
            source_type=DocumentType.PRICING,
            category="pricing",
            content=content,
            metadata={
                "plan_name": plan.get("name", ""),
                "monthly_price": plan.get("pricing", {}).get("monthly", 0),
                "annual_price": plan.get("pricing", {}).get("annual", 0),
                "features": plan.get("features", []),
            }
        )
        docs.append(doc)
    
    return docs


def normalize_refund_policy_documents() -> List[Document]:
    """Load refund policies and normalize to Document structure."""
    filepath = DATA_DIR / "refund_policies.json"
    data = load_json_file(filepath)
    docs = []
    
    for policy in data.get("refund_policies", []):
        content = f"Policy: {policy.get('policy_name', '')}\n"
        content += f"{policy.get('description', '')}\n"
        content += f"Refund Window: {policy.get('refund_window_days', 0)} days\n"
        content += f"Conditions: {'; '.join(policy.get('conditions', [])[:3])}"
        
        doc = Document(
            id=policy.get("id", ""),
            source_type=DocumentType.REFUND_POLICY,
            category="refund",
            content=content,
            metadata={
                "policy_name": policy.get("policy_name", ""),
                "refund_window_days": policy.get("refund_window_days", 0),
                "max_refund_amount": policy.get("max_refund_amount", 0),
                "escalation_required": policy.get("escalation_required", False),
            }
        )
        docs.append(doc)
    
    return docs


def normalize_escalation_documents() -> List[Document]:
    """Load escalation rules and normalize to Document structure."""
    filepath = DATA_DIR / "escalation_rules.json"
    data = load_json_file(filepath)
    docs = []
    
    for rule in data.get("escalation_rules", []):
        content = f"Rule: {rule.get('rule_name', '')}\n"
        content += f"{rule.get('description', '')}\n"
        content += f"Severity: {rule.get('severity', 'unknown')}\n"
        content += f"Action: {rule.get('action', '')[:200]}"
        
        doc = Document(
            id=rule.get("id", ""),
            source_type=DocumentType.ESCALATION_RULE,
            category="escalation",
            content=content,
            metadata={
                "rule_name": rule.get("rule_name", ""),
                "severity": rule.get("severity", "medium"),
                "trigger_conditions": rule.get("trigger_conditions", []),
                "sla_minutes": rule.get("sla_minutes", 0),
                "auto_escalate": rule.get("auto_escalate", False),
            }
        )
        docs.append(doc)
    
    return docs


def load_all_documents() -> List[Document]:
    """Load and normalize all document types into unified structure."""
    all_docs = []
    all_docs.extend(normalize_faq_documents())
    all_docs.extend(normalize_pricing_documents())
    all_docs.extend(normalize_refund_policy_documents())
    all_docs.extend(normalize_escalation_documents())
    
    print(f"✓ Loaded {len(all_docs)} documents across all sources")
    return all_docs


# ──────────────────────────────────────────────────────────────
# FAQ EXACT MATCH LAYER — Deterministic Retrieval
# ──────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization with lowercase normalization."""
    text = text.lower()
    # Remove punctuation and split
    tokens = re.sub(r'[^\w\s]', '', text).split()
    return [t for t in tokens if len(t) > 2]  # Filter short tokens


def calculate_faq_match_score(query: str, faq_question: str) -> float:
    """
    Calculate FAQ match score using fuzzy matching + keyword overlap.
    Returns score in [0, 1].
    """
    query_lower = query.lower()
    question_lower = faq_question.lower()
    
    # Fuzzy string similarity (SequenceMatcher)
    fuzzy_score = difflib.SequenceMatcher(None, query_lower, question_lower).ratio()
    
    # Token overlap score
    query_tokens = set(tokenize_text(query_lower))
    question_tokens = set(tokenize_text(question_lower))
    
    if not question_tokens:
        overlap_score = 0.0
    else:
        overlap_score = len(query_tokens & question_tokens) / len(question_tokens)
    
    # Weighted combination
    match_score = 0.4 * fuzzy_score + 0.6 * overlap_score
    return match_score


def faq_exact_match(query: str, faq_docs: List[Document], threshold: float = 0.65) -> Optional[Document]:
    """
    Deterministic FAQ retrieval: return exact match if above threshold.
    Returns single best FAQ or None if no strong match.
    """
    best_score = 0.0
    best_doc = None
    
    for doc in faq_docs:
        if doc.source_type != DocumentType.FAQ:
            continue
        
        question = doc.metadata.get("question", "")
        score = calculate_faq_match_score(query, question)
        
        if score > best_score:
            best_score = score
            best_doc = doc
    
    if best_score >= threshold:
        print(f"  ✓ FAQ exact match found: {best_doc.metadata['question'][:60]}... (score: {best_score:.3f})")
        return best_doc
    
    return None


# ──────────────────────────────────────────────────────────────
# BM25 RETRIEVAL LAYER — Keyword-based Retrieval
# ──────────────────────────────────────────────────────────────

def build_bm25_index(documents: List[Document]) -> Tuple[BM25Okapi, List[str]]:
    """Build BM25 index from document corpus."""
    if not BM25_AVAILABLE:
        print("⚠️  BM25 unavailable")
        return None, []
    
    # Tokenize all documents
    tokenized_docs = []
    doc_ids = []
    
    for doc in documents:
        tokens = tokenize_text(doc.content)
        tokenized_docs.append(tokens)
        doc_ids.append(doc.id)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    print(f"✓ BM25 index built with {len(documents)} documents")
    return bm25, doc_ids


def bm25_retrieve(
    query: str,
    documents: List[Document],
    bm25_index: BM25Okapi,
    doc_ids: List[str],
    top_k: int = TOP_K_BM25
) -> List[Tuple[Document, float]]:
    """
    BM25 keyword retrieval: tokenize query, score all docs, return top-k.
    Returns list of (Document, score) tuples.
    """
    if bm25_index is None:
        return []
    
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []
    
    # Score documents
    scores = bm25_index.get_scores(query_tokens)
    
    # Sort by score
    scored_docs = []
    for doc_id, score in zip(doc_ids, scores):
        if score > BM25_MIN_SCORE:
            doc = next((d for d in documents if d.id == doc_id), None)
            if doc:
                scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    result = scored_docs[:top_k]
    
    if result:
        print(f"  ✓ BM25 retrieved {len(result)} documents")
    
    return result


# ──────────────────────────────────────────────────────────────
# SEMANTIC RETRIEVAL LAYER — Vector-based Retrieval
# ──────────────────────────────────────────────────────────────

class SemanticRetriever:
    """Semantic retrieval using sentence-transformers + FAISS."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.documents = []
        self.document_ids = []
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✓ Embedding model loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
    
    def build_index(self, documents: List[Document]) -> bool:
        """Build FAISS index from document corpus."""
        if self.model is None or not FAISS_AVAILABLE:
            return False
        
        self.documents = documents
        
        # Embed all documents
        embeddings = []
        for doc in documents:
            try:
                embedding = self.model.encode(doc.content, convert_to_numpy=True)
                embeddings.append(embedding)
                doc.embedding = embedding
            except Exception as e:
                print(f"❌ Failed to embed document {doc.id}: {e}")
                embeddings.append(np.zeros((384,)))  # Fallback embedding
        
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        # Build FAISS index
        try:
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings_array)
            print(f"✓ FAISS index built with {len(documents)} documents (dimension: {dimension})")
            return True
        except Exception as e:
            print(f"❌ Failed to build FAISS index: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = TOP_K_SEMANTIC) -> List[Tuple[Document, float]]:
        """
        Semantic similarity search: embed query, find nearest neighbors in FAISS.
        Returns list of (Document, similarity_score) tuples.
        """
        if self.model is None or self.faiss_index is None:
            return []
        
        try:
            # Embed query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            query_embedding = np.array([query_embedding]).astype(np.float32)
            
            # Search FAISS index (L2 distance)
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Convert L2 distance to similarity score (0-1)
            # Smaller L2 distance = higher similarity
            result = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    # Normalize L2 distance to similarity (exponential decay)
                    similarity = np.exp(-dist)
                    result.append((doc, similarity))
            
            if result:
                print(f"  ✓ Semantic retrieval found {len(result)} documents")
            
            return result
        except Exception as e:
            print(f"❌ Semantic retrieval error: {e}")
            return []


# ──────────────────────────────────────────────────────────────
# MEMORY RETRIEVAL LAYER — Personalized Context
# ──────────────────────────────────────────────────────────────

def extract_memory_patterns(conversation_history: List[Dict[str, str]]) -> List[str]:
    """
    Extract recurring patterns and issues from conversation history.
    Returns list of memory context strings.
    """
    if not conversation_history:
        return []
    
    patterns = []
    
    # Look for repeated topics
    all_text = " ".join([msg.get("content", "") for msg in conversation_history])
    
    # Simple pattern extraction
    billing_keywords = ["charge", "billing", "payment", "invoice", "refund", "bill"]
    if any(kw in all_text.lower() for kw in billing_keywords):
        patterns.append("MEMORY: User has billing-related concerns in history")
    
    escalation_keywords = ["urgent", "angry", "complaint", "not working", "broken", "fail"]
    if any(kw in all_text.lower() for kw in escalation_keywords):
        patterns.append("MEMORY: User has expressed urgency or escalation sentiment")
    
    # Count message frequency (proxy for engagement level)
    msg_count = len(conversation_history)
    if msg_count > 5:
        patterns.append(f"MEMORY: Extended conversation ({msg_count} messages)")
    
    return patterns


def memory_retrieve(
    conversation_history: List[Dict[str, str]],
    semantic_retriever: Optional[SemanticRetriever] = None,
    top_k: int = TOP_K_MEMORY
) -> Tuple[List[str], List[Tuple[Document, float]]]:
    """
    Personalized memory retrieval from conversation history.
    Returns (memory_context_strings, retrieved_memory_documents).
    """
    memory_context = extract_memory_patterns(conversation_history)
    memory_docs = []
    
    # If semantic retriever available, retrieve similar historical context
    if semantic_retriever and conversation_history:
        # Build query from recent conversation
        recent_msgs = [msg.get("content", "") for msg in conversation_history[-3:]]
        if recent_msgs:
            combined_query = " ".join(recent_msgs)
            memory_docs = semantic_retriever.retrieve(combined_query, top_k=top_k)
    
    if memory_context or memory_docs:
        print(f"  ✓ Memory retrieval: {len(memory_context)} patterns, {len(memory_docs)} documents")
    
    return memory_context, memory_docs


# ──────────────────────────────────────────────────────────────
# RETRIEVAL EVALUATOR — Confidence Scoring
# ──────────────────────────────────────────────────────────────

def evaluate_retrieval(
    query: str,
    faq_score: Optional[float],
    bm25_results: List[Tuple[Document, float]],
    semantic_results: List[Tuple[Document, float]],
    memory_results: List[Tuple[Document, float]]
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Evaluate retrieval quality and assign confidence score.
    Returns (confidence_score, retrieval_source, evaluation_details).
    
    Confidence scoring logic:
    - FAQ exact match (confidence > 0.9): deterministic, high confidence
    - Multi-source agreement (BM25 + semantic): high confidence (0.75-0.85)
    - Single strong source (BM25 OR semantic): medium confidence (0.55-0.75)
    - Weak/no matches: low confidence (< 0.55)
    """
    evaluation = {
        "faq_match_score": faq_score,
        "bm25_count": len(bm25_results),
        "semantic_count": len(semantic_results),
        "memory_count": len(memory_results),
        "source_agreement": [],
    }
    
    confidence = 0.0
    source = RetrievalSource.NONE.value
    
    # FAQ exact match is deterministic high-confidence
    if faq_score is not None and faq_score >= 0.65:
        confidence = 0.95
        source = RetrievalSource.FAQ_EXACT.value
        evaluation["rationale"] = "FAQ exact match (deterministic)"
    
    # Multi-source agreement (BM25 + semantic)
    elif bm25_results and semantic_results:
        # Check if top results overlap (same source)
        bm25_sources = {doc.id for doc, _ in bm25_results[:3]}
        semantic_sources = {doc.id for doc, _ in semantic_results[:3]}
        overlap = bm25_sources & semantic_sources
        
        if overlap:
            confidence = 0.82
            source = RetrievalSource.MULTI.value
            evaluation["rationale"] = f"Multi-source agreement ({len(overlap)} overlapping docs)"
        else:
            # Different sources but both present
            avg_bm25 = np.mean([s for _, s in bm25_results[:3]])
            avg_semantic = np.mean([s for _, s in semantic_results[:3]])
            confidence = 0.7 + 0.1 * min(avg_bm25, avg_semantic)
            source = RetrievalSource.MULTI.value
            evaluation["rationale"] = "Multi-source (no overlap)"
    
    # Strong BM25-only retrieval
    elif bm25_results:
        top_bm25_score = bm25_results[0][1]
        confidence = 0.5 + 0.25 * min(top_bm25_score / 10, 1.0)  # Normalize BM25 score
        source = RetrievalSource.BM25.value
        evaluation["rationale"] = f"BM25 retrieval (top score: {top_bm25_score:.3f})"
    
    # Strong semantic-only retrieval
    elif semantic_results:
        top_semantic_score = semantic_results[0][1]
        confidence = 0.55 + 0.3 * top_semantic_score
        source = RetrievalSource.SEMANTIC.value
        evaluation["rationale"] = f"Semantic retrieval (similarity: {top_semantic_score:.3f})"
    
    # Memory-only retrieval
    elif memory_results:
        confidence = 0.45
        source = RetrievalSource.MEMORY.value
        evaluation["rationale"] = "Memory retrieval only (low confidence)"
    
    # No retrieval
    else:
        confidence = 0.2
        source = RetrievalSource.NONE.value
        evaluation["rationale"] = "No strong retrieval match"
    
    # Flag for clarification if confidence is too low
    requires_clarification = confidence < CONFIDENCE_THRESHOLD
    evaluation["requires_clarification"] = requires_clarification
    
    return confidence, source, evaluation


# ──────────────────────────────────────────────────────────────
# CONTEXT MERGING & DEDUPLICATION
# ──────────────────────────────────────────────────────────────

def merge_retrieved_context(
    faq_doc: Optional[Document],
    bm25_results: List[Tuple[Document, float]],
    semantic_results: List[Tuple[Document, float]],
    memory_context: List[str],
    memory_docs: List[Tuple[Document, float]]
) -> Tuple[str, List[Document]]:
    """
    Merge and deduplicate retrieved context into coherent final context.
    Returns (merged_context_string, all_retrieved_documents).
    """
    context_parts = []
    all_docs = []
    seen_ids = set()
    
    # 1. FAQ exact match (if present, prioritize)
    if faq_doc:
        context_parts.append(f"[FAQ MATCH]\n{faq_doc.content}")
        all_docs.append(faq_doc)
        seen_ids.add(faq_doc.id)
    
    # 2. BM25 results
    for doc, score in bm25_results:
        if doc.id not in seen_ids:
            context_parts.append(f"[BM25 - relevance: {score:.3f}]\n{doc.content}")
            all_docs.append(doc)
            seen_ids.add(doc.id)
    
    # 3. Semantic results
    for doc, score in semantic_results:
        if doc.id not in seen_ids:
            context_parts.append(f"[SEMANTIC - similarity: {score:.3f}]\n{doc.content}")
            all_docs.append(doc)
            seen_ids.add(doc.id)
    
    # 4. Memory patterns
    if memory_context:
        context_parts.append(f"[MEMORY PATTERNS]\n" + "\n".join(memory_context))
    
    # 5. Memory documents
    for doc, score in memory_docs:
        if doc.id not in seen_ids:
            context_parts.append(f"[MEMORY DOC - relevance: {score:.3f}]\n{doc.content}")
            all_docs.append(doc)
            seen_ids.add(doc.id)
    
    final_context = "\n\n".join(context_parts)
    
    return final_context, all_docs


# ──────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION FUNCTION
# ──────────────────────────────────────────────────────────────

class HybridRetriever:
    """Production-style hybrid retrieval orchestrator."""
    
    def __init__(self):
        self.all_documents = []
        self.faq_documents = []
        self.bm25_index = None
        self.bm25_doc_ids = []
        self.semantic_retriever = SemanticRetriever()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Load documents and build indexes."""
        print("\n" + "="*70)
        print("HYBRID RETRIEVER INITIALIZATION")
        print("="*70)
        
        # Load documents
        self.all_documents = load_all_documents()
        self.faq_documents = [d for d in self.all_documents if d.source_type == DocumentType.FAQ]
        
        # Build BM25 index
        if BM25_AVAILABLE:
            self.bm25_index, self.bm25_doc_ids = build_bm25_index(self.all_documents)
        
        # Build semantic index
        if EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
            self.semantic_retriever.build_index(self.all_documents)
        
        self.initialized = True
        print("="*70 + "\n")
        return True
    
    def retrieve(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RetrievalResult:
        """
        Main hybrid retrieval orchestration function.
        
        Pipeline:
          1. FAQ exact match (deterministic)
          2. BM25 retrieval (keyword)
          3. Semantic retrieval (vector)
          4. Memory retrieval (personalized)
          5. Evaluation (confidence scoring)
          6. Context merging
        
        Args:
            query: User query string
            conversation_history: List of previous messages for context
        
        Returns:
            RetrievalResult with structured output
        """
        if not self.initialized:
            self.initialize()
        
        if conversation_history is None:
            conversation_history = []
        
        print(f"\n{'─'*70}")
        print(f"🔍 RETRIEVAL QUERY: {query[:80]}")
        print(f"{'─'*70}")
        
        retrieval_scores = {
            "faq": [],
            "bm25": [],
            "semantic": [],
            "memory": [],
        }
        
        # Step 1: FAQ Exact Match (Deterministic)
        print("\n1️⃣  FAQ EXACT MATCH (Deterministic)")
        faq_doc = faq_exact_match(query, self.faq_documents)
        faq_score = calculate_faq_match_score(query, faq_doc.metadata.get("question", "")) if faq_doc else None
        if faq_score:
            retrieval_scores["faq"].append(faq_score)
        
        # Step 2: BM25 Retrieval (Keyword)
        print("\n2️⃣  BM25 KEYWORD RETRIEVAL")
        bm25_results = bm25_retrieve(query, self.all_documents, self.bm25_index, self.bm25_doc_ids)
        retrieval_scores["bm25"] = [score for _, score in bm25_results]
        
        # Step 3: Semantic Retrieval (Vector)
        print("\n3️⃣  SEMANTIC VECTOR RETRIEVAL")
        semantic_results = self.semantic_retriever.retrieve(query)
        retrieval_scores["semantic"] = [score for _, score in semantic_results]
        
        # Step 4: Memory Retrieval (Personalized)
        print("\n4️⃣  MEMORY-AWARE RETRIEVAL")
        memory_context, memory_docs = memory_retrieve(
            conversation_history,
            self.semantic_retriever
        )
        retrieval_scores["memory"] = [score for _, score in memory_docs]
        
        # Step 5: Retrieval Evaluation
        print("\n5️⃣  RETRIEVAL EVALUATION")
        confidence, source, evaluation_details = evaluate_retrieval(
            query, faq_score, bm25_results, semantic_results, memory_docs
        )
        print(f"  → Confidence: {confidence:.3f}")
        print(f"  → Source: {source}")
        print(f"  → Rationale: {evaluation_details.get('rationale', 'N/A')}")
        
        # Step 6: Context Merging
        print("\n6️⃣  CONTEXT MERGING & DEDUPLICATION")
        final_context, all_retrieved_docs = merge_retrieved_context(
            faq_doc, bm25_results, semantic_results, memory_context, memory_docs
        )
        print(f"  → Merged {len(all_retrieved_docs)} unique documents")
        
        # Flag for clarification
        requires_clarification = confidence < CONFIDENCE_THRESHOLD
        if requires_clarification:
            print(f"\n⚠️  CLARIFICATION REQUIRED (confidence: {confidence:.3f} < {CONFIDENCE_THRESHOLD})")
        
        result = RetrievalResult(
            query=query,
            retrieval_source=source,
            retrieval_confidence=confidence,
            requires_clarification=requires_clarification,
            retrieved_documents=all_retrieved_docs,
            memory_context=memory_context,
            final_context=final_context,
            retrieval_scores=retrieval_scores,
            evaluation_details=evaluation_details,
        )
        
        return result


# ──────────────────────────────────────────────────────────────
# LOGGING & DEMO UTILITIES
# ──────────────────────────────────────────────────────────────

def print_retrieval_result(result: RetrievalResult):
    """Pretty-print retrieval result."""
    print(f"\n{'═'*70}")
    print(f"📊 RETRIEVAL RESULT")
    print(f"{'═'*70}")
    print(f"Query: {result.query}")
    print(f"Source: {result.retrieval_source.upper()}")
    print(f"Confidence: {result.retrieval_confidence:.3f}")
    print(f"Requires Clarification: {result.requires_clarification}")
    print(f"\nDocuments Retrieved: {len(result.retrieved_documents)}")
    for i, doc in enumerate(result.retrieved_documents[:3], 1):
        print(f"  {i}. [{doc.source_type.value.upper()}] {doc.id}")
        print(f"     Category: {doc.category}")
        print(f"     Content: {doc.content[:100]}...")
    
    if result.memory_context:
        print(f"\nMemory Context:")
        for mem in result.memory_context:
            print(f"  • {mem}")
    
    print(f"\nRetrieval Scores:")
    for source_type, scores in result.retrieval_scores.items():
        if scores:
            print(f"  {source_type}: {scores}")
    
    print(f"{'═'*70}\n")


# ──────────────────────────────────────────────────────────────
# DEMO TEST CASES
# ──────────────────────────────────────────────────────────────

def run_demo():
    """Run demo test cases demonstrating hybrid retrieval."""
    
    print("\n" + "█"*70)
    print("█ NOVADESK AI — HYBRID RETRIEVAL SYSTEM DEMO")
    print("█"*70)
    
    # Initialize retriever
    retriever = HybridRetriever()
    retriever.initialize()
    
    # Demo queries
    test_queries = [
        "Why was I charged twice?",
        "My bill suddenly increased.",
        "I still have the same invoice issue.",
        "Can I get a refund after heavy API usage?",
        "I will take legal action.",
    ]
    
    conversation_history = [
        {"role": "user", "content": "I have a billing issue."},
        {"role": "assistant", "content": "I'm here to help with billing issues."},
    ]
    
    print("\n" + "▼"*70)
    print("RUNNING 5 DEMO TEST QUERIES")
    print("▼"*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"# DEMO {i}/5: {query}")
        print(f"{'#'*70}")
        
        result = retriever.retrieve(query, conversation_history)
        print_retrieval_result(result)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({
            "role": "assistant",
            "content": f"Based on my retrieval, I found relevant documents with {result.retrieval_confidence:.1%} confidence."
        })


if __name__ == "__main__":
    run_demo()
