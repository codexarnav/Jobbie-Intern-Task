

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  WARNING: sentence-transformers not installed")

from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROUNDING_THRESHOLD = 0.55
HALLUCINATION_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.65
DIVERSITY_THRESHOLD = 0.70
SEMANTIC_SIMILARITY_THRESHOLD = 0.80

# Unsupported claim patterns
UNSUPPORTED_PATTERNS = [
    r"we (will|always|guarantee) (refund|return)",  # unsupported refund guarantees
    r"your (plan|account) (will|must) (change|upgrade)",  # unsupported upgrades
    r"we (never|always) (charge|deduct)",  # absolute claims
    r"this is (definitely|certainly|definitely)" if "pricing" in "" else "",  # overconfident pricing
]

HALLUCINATION_KEYWORDS = [
    "enterprise plan includes", "guaranteed", "always", "never", "your account will be",
    "we promise", "you are entitled to", "we will definitely",
]


# ──────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """Normalized confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Validation-triggered recovery actions."""
    CONTINUE_NORMAL = "continue_normal_flow"
    REFRESH_RETRIEVAL = "refresh_retrieval"
    REGENERATE_RESPONSE = "regenerate_response"
    CLARIFICATION_REQUEST = "clarification_request"
    ESCALATE_TO_HUMAN = "escalate_to_human"


@dataclass
class GroundingValidation:
    """Grounding validation result."""
    grounded: bool
    grounding_score: float
    supporting_documents: List[str]  # document IDs
    reason: str
    unsupported_segments: List[str]


@dataclass
class HallucinationValidation:
    """Hallucination detection result."""
    hallucination_detected: bool
    hallucination_risk: float
    reason: str
    unsupported_segments: List[str]
    confidence_penalty: float


@dataclass
class ConfidenceValidation:
    """Combined confidence validation."""
    confidence_score: float
    confidence_level: str
    component_scores: Dict[str, float]
    rationale: str


@dataclass
class ResponseDiversityValidation:
    """Response diversity and loop prevention."""
    diversity_score: float
    repetition_detected: bool
    reason: str
    similar_previous_responses: List[int]  # indices


@dataclass
class ValidationResult:
    """Complete validation output for orchestration."""
    valid: bool
    confidence_score: float
    confidence_level: str

    # Validation components
    grounding: Dict[str, Any]
    hallucination: Dict[str, Any]
    loop_validation: Dict[str, Any]

    # Integration signals
    retrieval_confidence_input: float
    loop_risk_input: float

    # Escalation logic
    requires_escalation: bool
    escalation_reason: str
    recovery_action: str

    # System state
    validation_summary: str
    system_awareness: Dict[str, Any]


# ──────────────────────────────────────────────────────────────
# EMBEDDINGS & SEMANTIC UTILITIES
# ──────────────────────────────────────────────────────────────

class SemanticValidator:
    """Semantic similarity validation using embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if self.model is None:
            return 0.0
        
        try:
            emb1 = self.model.encode(text1, convert_to_numpy=True)
            emb2 = self.model.encode(text2, convert_to_numpy=True)
            sim = cosine_similarity([emb1], [emb2])[0][0]
            return float(sim)
        except Exception:
            return 0.0

    def batch_similarity(self, text1: str, text_list: List[str]) -> List[float]:
        """Compute similarity between one text and a list of texts."""
        if self.model is None:
            return [0.0] * len(text_list)
        
        try:
            emb1 = self.model.encode(text1, convert_to_numpy=True)
            embs_list = self.model.encode(text_list, convert_to_numpy=True)
            similarities = cosine_similarity([emb1], embs_list)[0]
            return [float(s) for s in similarities]
        except Exception:
            return [0.0] * len(text_list)


# ──────────────────────────────────────────────────────────────
# GROUNDING VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_grounding(
    generated_response: str,
    retrieved_context: str,
    semantic_validator: Optional[SemanticValidator] = None,
    retrieval_source: Optional[str] = None
) -> GroundingValidation:
    """
    Validate that generated response is grounded in retrieved documents.
    
    High grounding score = response semantically overlaps with retrieved context.
    Low grounding score = response diverges from retrieval (hallucination risk).
    """
    if not retrieved_context:
        return GroundingValidation(
            grounded=False,
            grounding_score=0.0,
            supporting_documents=[],
            reason="No retrieved context available",
            unsupported_segments=[]
        )

    # Text-based overlap heuristic
    response_words = set(generated_response.lower().split())
    context_words = set(retrieved_context.lower().split())
    
    if not context_words:
        overlap_score = 0.0
    else:
        word_overlap = len(response_words & context_words) / len(context_words)
        overlap_score = word_overlap

    # Semantic similarity (if available)
    semantic_score = 0.0
    if semantic_validator:
        semantic_score = semantic_validator.similarity(generated_response, retrieved_context)

    # Weighted combination
    grounding_score = 0.3 * overlap_score + 0.7 * semantic_score

    # Determine grounding status
    grounded = grounding_score >= GROUNDING_THRESHOLD

    # FAQ exact matches are inherently grounded
    if retrieval_source == "faq_exact":
        grounded = True
        grounding_score = min(1.0, grounding_score + 0.2)

    reason = f"Grounding score: {grounding_score:.3f} ({'supported' if grounded else 'weak'})"

    print(f"  ✓ Grounding validation: {grounding_score:.3f} {'(grounded)' if grounded else '(weak)'}")

    return GroundingValidation(
        grounded=grounded,
        grounding_score=grounding_score,
        supporting_documents=[],  # Could add doc IDs if available
        reason=reason,
        unsupported_segments=[]
    )


# ──────────────────────────────────────────────────────────────
# HALLUCINATION DETECTION
# ──────────────────────────────────────────────────────────────

def detect_hallucination(
    generated_response: str,
    retrieved_context: str,
    retrieval_confidence: float,
    semantic_validator: Optional[SemanticValidator] = None,
) -> HallucinationValidation:
    """
    Detect unsupported or fabricated content in generated response.
    
    Hallucination risk increases when:
    - Low semantic overlap with retrieved context
    - Low retrieval confidence
    - Unsupported claim patterns detected
    - Overconfident language used
    """
    hallucination_risk = 0.0
    unsupported_segments = []
    confidence_penalty = 0.0

    # Signal 1: Low semantic overlap
    if semantic_validator:
        overlap = semantic_validator.similarity(generated_response, retrieved_context)
        if overlap < 0.55:
            hallucination_risk += 0.4 * (1 - overlap)
    else:
        # Fallback text overlap
        response_words = set(generated_response.lower().split())
        context_words = set(retrieved_context.lower().split())
        if context_words:
            word_overlap = len(response_words & context_words) / len(context_words)
            if word_overlap < 0.55:
                hallucination_risk += 0.3 * (1 - word_overlap)

    # Signal 2: Low retrieval confidence
    if retrieval_confidence < 0.65:
        hallucination_risk += 0.3 * (1 - retrieval_confidence)
        confidence_penalty += 0.1

    # Signal 3: Unsupported claim patterns
    response_lower = generated_response.lower()
    for keyword in HALLUCINATION_KEYWORDS:
        if keyword in response_lower:
            hallucination_risk += 0.15
            unsupported_segments.append(keyword)

    # Signal 4: Overconfident absolute statements
    absolute_keywords = ["always", "never", "guaranteed", "definitely", "certainly"]
    for keyword in absolute_keywords:
        if keyword in response_lower and retrieval_confidence < 0.8:
            hallucination_risk += 0.1
            unsupported_segments.append(keyword)

    # Normalize risk to [0, 1]
    hallucination_risk = min(1.0, hallucination_risk)

    hallucination_detected = hallucination_risk >= HALLUCINATION_THRESHOLD

    reason = f"Hallucination risk: {hallucination_risk:.3f} ({'high risk' if hallucination_detected else 'acceptable'})"

    print(f"  ✓ Hallucination detection: {hallucination_risk:.3f} {'(HIGH RISK)' if hallucination_detected else '(acceptable)'}")

    return HallucinationValidation(
        hallucination_detected=hallucination_detected,
        hallucination_risk=hallucination_risk,
        reason=reason,
        unsupported_segments=unsupported_segments,
        confidence_penalty=confidence_penalty
    )


# ──────────────────────────────────────────────────────────────
# CONFIDENCE VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_confidence(
    retrieval_confidence: float,
    grounding_score: float,
    hallucination_risk: float,
    loop_risk_score: float = 0.0,
    response_diversity_score: float = 1.0,
) -> ConfidenceValidation:
    """
    Combine multiple confidence signals into final system confidence.
    
    Inputs from:
    - retrieval.py (retrieval_confidence)
    - grounding validation (grounding_score)
    - hallucination detection (hallucination_risk)
    - loop_detection.py (loop_risk_score)
    
    Produces normalized confidence in [0, 1].
    """
    
    # Component scores
    component_scores = {
        "retrieval": retrieval_confidence,
        "grounding": grounding_score,
        "hallucination": 1.0 - hallucination_risk,
        "loop_risk": 1.0 - loop_risk_score,
        "response_diversity": response_diversity_score,
    }

    # Weighted combination
    weights = {
        "retrieval": 0.25,
        "grounding": 0.25,
        "hallucination": 0.25,
        "loop_risk": 0.15,
        "response_diversity": 0.10,
    }

    confidence_score = sum(
        component_scores[key] * weights[key]
        for key in component_scores
    )
    confidence_score = min(1.0, max(0.0, confidence_score))

    # Classify confidence level
    if confidence_score >= 0.85:
        confidence_level = ConfidenceLevel.HIGH.value
    elif confidence_score >= 0.65:
        confidence_level = ConfidenceLevel.MEDIUM.value
    elif confidence_score >= 0.45:
        confidence_level = ConfidenceLevel.LOW.value
    else:
        confidence_level = ConfidenceLevel.CRITICAL.value

    rationale = (
        f"Confidence: {confidence_score:.3f} ({confidence_level}) | "
        f"Retrieval: {retrieval_confidence:.2f}, "
        f"Grounding: {grounding_score:.2f}, "
        f"Hallucination: {1-hallucination_risk:.2f}"
    )

    print(f"  ✓ Confidence validation: {confidence_score:.3f} ({confidence_level})")

    return ConfidenceValidation(
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        component_scores=component_scores,
        rationale=rationale
    )


# ──────────────────────────────────────────────────────────────
# RESPONSE DIVERSITY VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_response_diversity(
    generated_response: str,
    assistant_history: List[str],
    semantic_validator: Optional[SemanticValidator] = None,
) -> ResponseDiversityValidation:
    """
    Validate that response is not a repetition of previous responses.
    
    Integrates with loop_detection.py to prevent conversational stagnation.
    """
    if not assistant_history:
        return ResponseDiversityValidation(
            diversity_score=1.0,
            repetition_detected=False,
            reason="No prior responses to compare",
            similar_previous_responses=[]
        )

    similar_indices = []
    max_similarity = 0.0

    if semantic_validator:
        similarities = semantic_validator.batch_similarity(generated_response, assistant_history)
        for idx, sim in enumerate(similarities):
            max_similarity = max(max_similarity, sim)
            if sim > 0.85:  # Threshold for "similar"
                similar_indices.append(idx)
    else:
        # Fallback: exact substring matching
        for idx, prev_response in enumerate(assistant_history):
            if generated_response.lower() in prev_response.lower() or \
               prev_response.lower() in generated_response.lower():
                similar_indices.append(idx)
                max_similarity = 1.0

    # Diversity score: 1 - max_similarity
    diversity_score = 1.0 - max_similarity

    repetition_detected = max_similarity > DIVERSITY_THRESHOLD

    reason = f"Diversity score: {diversity_score:.3f}, Max similarity to history: {max_similarity:.3f}"

    if repetition_detected:
        print(f"  ⚠️  Response diversity: {diversity_score:.3f} (REPETITION DETECTED)")
    else:
        print(f"  ✓ Response diversity: {diversity_score:.3f} (diverse)")

    return ResponseDiversityValidation(
        diversity_score=diversity_score,
        repetition_detected=repetition_detected,
        reason=reason,
        similar_previous_responses=similar_indices
    )


# ──────────────────────────────────────────────────────────────
# ESCALATION LOGIC
# ──────────────────────────────────────────────────────────────

def decide_escalation(
    confidence_score: float,
    hallucination_risk: float,
    grounding_score: float,
    loop_risk_score: float,
    repetition_detected: bool,
    retrieval_confidence: float,
) -> Tuple[bool, str, str]:
    """
    Decide whether to escalate to human based on validation signals.
    
    Returns: (requires_escalation, escalation_reason, recovery_action)
    """
    
    escalation_signals = []

    # Signal 1: Overall confidence too low
    if confidence_score < 0.45:
        escalation_signals.append("confidence_critical")

    # Signal 2: Hallucination risk too high
    if hallucination_risk > 0.70:
        escalation_signals.append("hallucination_critical")

    # Signal 3: Grounding very weak
    if grounding_score < 0.40:
        escalation_signals.append("grounding_weak")

    # Signal 4: Loop risk critical
    if loop_risk_score > 0.80:
        escalation_signals.append("loop_critical")

    # Signal 5: Response repetition
    if repetition_detected and loop_risk_score > 0.5:
        escalation_signals.append("response_repetition")

    # Signal 6: Retrieval completely failed
    if retrieval_confidence < 0.30:
        escalation_signals.append("retrieval_failed")

    requires_escalation = len(escalation_signals) > 0

    if requires_escalation:
        escalation_reason = f"Escalation triggered: {', '.join(escalation_signals)}"
        recovery_action = RecoveryAction.ESCALATE_TO_HUMAN.value
        print(f"  🚨 ESCALATION RECOMMENDED: {escalation_reason}")
    else:
        escalation_reason = "No escalation needed"
        
        # Suggest recovery action based on signals
        if repetition_detected:
            recovery_action = RecoveryAction.REGENERATE_RESPONSE.value
        elif hallucination_risk > 0.60:
            recovery_action = RecoveryAction.REFRESH_RETRIEVAL.value
        elif grounding_score < 0.55:
            recovery_action = RecoveryAction.CLARIFICATION_REQUEST.value
        else:
            recovery_action = RecoveryAction.CONTINUE_NORMAL.value

    return requires_escalation, escalation_reason, recovery_action


# ──────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION FUNCTION
# ──────────────────────────────────────────────────────────────

def validate_response(
    generated_response: str,
    retrieval_result: Dict[str, Any],
    loop_detection_result: Optional[Dict[str, Any]] = None,
    assistant_history: Optional[List[str]] = None,
    system_state: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Main validation orchestration function.
    
    Integrates with:
    - retrieval.py (RetrievalResult)
    - loop_detection.py (PostLoopResult)
    
    Produces ValidationResult with structured escalation decisions.
    
    Args:
        generated_response: AI-generated response text
        retrieval_result: Output from retrieval.py (RetrievalResult dict)
        loop_detection_result: Output from loop_detection.py (PostLoopResult dict)
        assistant_history: List of previous assistant responses
        system_state: Lightweight system state tracking
    
    Returns:
        ValidationResult with validation summary and escalation recommendation
    """
    
    if assistant_history is None:
        assistant_history = []
    if system_state is None:
        system_state = {}

    print(f"\n{'─'*70}")
    print(f"🔍 VALIDATION ORCHESTRATION")
    print(f"{'─'*70}")

    # Extract inputs from retrieval result
    retrieval_confidence = retrieval_result.get("retrieval_confidence", 0.5)
    retrieved_context = retrieval_result.get("final_context", "")
    retrieval_source = retrieval_result.get("retrieval_source", "none")

    # Extract inputs from loop detection result
    loop_risk_score = 0.0
    if loop_detection_result:
        loop_risk_score = loop_detection_result.get("post_loop_score", 0.0)

    # Initialize semantic validator
    semantic_validator = None
    if EMBEDDINGS_AVAILABLE:
        semantic_validator = SemanticValidator()

    # Step 1: Grounding Validation
    print("\n1️⃣  GROUNDING VALIDATION")
    grounding_result = validate_grounding(
        generated_response,
        retrieved_context,
        semantic_validator,
        retrieval_source
    )

    # Step 2: Hallucination Detection
    print("\n2️⃣  HALLUCINATION DETECTION")
    hallucination_result = detect_hallucination(
        generated_response,
        retrieved_context,
        retrieval_confidence,
        semantic_validator
    )

    # Step 3: Response Diversity Validation
    print("\n3️⃣  RESPONSE DIVERSITY CHECK")
    diversity_result = validate_response_diversity(
        generated_response,
        assistant_history,
        semantic_validator
    )

    # Step 4: Confidence Validation
    print("\n4️⃣  CONFIDENCE VALIDATION")
    confidence_result = validate_confidence(
        retrieval_confidence,
        grounding_result.grounding_score,
        hallucination_result.hallucination_risk,
        loop_risk_score,
        diversity_result.diversity_score
    )

    # Step 5: Escalation Logic
    print("\n5️⃣  ESCALATION LOGIC")
    requires_escalation, escalation_reason, recovery_action = decide_escalation(
        confidence_result.confidence_score,
        hallucination_result.hallucination_risk,
        grounding_result.grounding_score,
        loop_risk_score,
        diversity_result.repetition_detected,
        retrieval_confidence
    )

    # Determine if response is valid for delivery
    valid = (
        confidence_result.confidence_score >= CONFIDENCE_THRESHOLD and
        not hallucination_result.hallucination_detected and
        not requires_escalation
    )

    # Build validation summary
    validation_summary = (
        f"Response confidence: {confidence_result.confidence_score:.3f} ({confidence_result.confidence_level}) | "
        f"Grounding: {grounding_result.grounding_score:.3f} | "
        f"Hallucination risk: {hallucination_result.hallucination_risk:.3f} | "
        f"Diversity: {diversity_result.diversity_score:.3f} | "
        f"Valid: {valid}"
    )

    # System awareness (for debugging and orchestration)
    system_awareness = {
        "retrieval_source": retrieval_source,
        "retrieval_confidence": retrieval_confidence,
        "loop_risk_score": loop_risk_score,
        "repetition_detected": diversity_result.repetition_detected,
        "unsupported_claims": hallucination_result.unsupported_segments,
    }

    result = ValidationResult(
        valid=valid,
        confidence_score=confidence_result.confidence_score,
        confidence_level=confidence_result.confidence_level,
        grounding=asdict(grounding_result),
        hallucination=asdict(hallucination_result),
        loop_validation=asdict(diversity_result),
        retrieval_confidence_input=retrieval_confidence,
        loop_risk_input=loop_risk_score,
        requires_escalation=requires_escalation,
        escalation_reason=escalation_reason,
        recovery_action=recovery_action,
        validation_summary=validation_summary,
        system_awareness=system_awareness,
    )

    print(f"\n{'─'*70}")
    print(f"✅ VALIDATION RESULT: {'VALID' if valid else 'INVALID'}")
    print(f"{'─'*70}\n")

    return result


# # ──────────────────────────────────────────────────────────────
# # DEMO TEST CASES
# # ──────────────────────────────────────────────────────────────

# def run_demo():
#     """Run demo test cases for validation system."""

#     print("\n" + "█"*70)
#     print("█ NOVADESK AI — VALIDATION SYSTEM DEMO")
#     print("█"*70)

#     # Initialize semantic validator
#     semantic_validator = SemanticValidator() if EMBEDDINGS_AVAILABLE else None

#     # Demo 1: Properly grounded billing response
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 1: Properly Grounded Billing Response")
#     print("▌"*70)

#     retrieval_1 = {
#         "retrieval_confidence": 0.92,
#         "retrieval_source": "faq_exact",
#         "final_context": "FAQ: Why was I charged twice? Answer: Duplicate charges can occur if a payment initially failed and was retried. Please check your billing history under Settings → Billing → Invoices. Contact support within 14 days for full refund.",
#     }

#     response_1 = "Based on our FAQ, duplicate charges can occur if a payment failed and was retried. I recommend checking your billing history under Settings → Billing → Invoices and contacting our support team within 14 days for a full refund."

#     result_1 = validate_response(response_1, retrieval_1)
#     print(result_1.validation_summary)
#     print(f"Escalation: {result_1.requires_escalation}")

#     # Demo 2: Unsupported refund promise
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 2: Unsupported Refund Promise (HALLUCINATION)")
#     print("▌"*70)

#     retrieval_2 = {
#         "retrieval_confidence": 0.45,
#         "retrieval_source": "semantic",
#         "final_context": "Usage-based charges are non-refundable once API calls have been consumed. Refund requests must follow our policy.",
#     }

#     response_2 = "We will definitely process a full refund for your usage charges immediately. Your account is guaranteed to be credited within 24 hours."

#     result_2 = validate_response(response_2, retrieval_2)
#     print(result_2.validation_summary)
#     print(f"Escalation: {result_2.requires_escalation}")
#     print(f"Hallucination Risk: {result_2.hallucination['hallucination_risk']:.3f}")

#     # Demo 3: Weak semantic retrieval
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 3: Low Retrieval Confidence")
#     print("▌"*70)

#     retrieval_3 = {
#         "retrieval_confidence": 0.35,
#         "retrieval_source": "none",
#         "final_context": "",
#     }

#     response_3 = "I don't have clear information about your specific issue. Let me connect you with a specialist."

#     result_3 = validate_response(response_3, retrieval_3)
#     print(result_3.validation_summary)
#     print(f"Escalation: {result_3.requires_escalation}")

#     # Demo 4: Repeated assistant response
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 4: Response Repetition Detection")
#     print("▌"*70)

#     history = [
#         "To resolve this, please check your billing settings and verify your payment method.",
#         "You may need to update your account information.",
#     ]

#     retrieval_4 = {
#         "retrieval_confidence": 0.70,
#         "retrieval_source": "semantic",
#         "final_context": "Billing settings can be updated from your dashboard.",
#     }

#     response_4 = "To resolve this, please check your billing settings and verify your payment method. This should address the issue."

#     result_4 = validate_response(response_4, retrieval_4, assistant_history=history)
#     print(result_4.validation_summary)
#     print(f"Repetition Detected: {result_4.loop_validation['repetition_detected']}")

#     # Demo 5: High loop risk with low grounding
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 5: High Loop Risk + Low Grounding = ESCALATION")
#     print("▌"*70)

#     loop_result = {"post_loop_score": 0.85}

#     retrieval_5 = {
#         "retrieval_confidence": 0.50,
#         "retrieval_source": "bm25",
#         "final_context": "Partial context available.",
#     }

#     response_5 = "I understand your concern. Let me help you further."

#     result_5 = validate_response(response_5, retrieval_5, loop_result)
#     print(result_5.validation_summary)
#     print(f"Escalation Required: {result_5.requires_escalation}")
#     print(f"Escalation Reason: {result_5.escalation_reason}")

#     print("\n\n" + "█"*70)
#     print("█ DEMO COMPLETE")
#     print("█"*70 + "\n")


# if __name__ == "__main__":
#     run_demo()
