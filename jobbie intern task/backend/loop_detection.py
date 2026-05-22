
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import numpy as np

# Third-party dependencies
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  WARNING: sentence-transformers not installed")

from sklearn.metrics.pairwise import cosine_similarity




EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEMANTIC_SIMILARITY_THRESHOLD = 0.88
FRUSTRATION_SIMILARITY_THRESHOLD = 0.75
POST_LOOP_SIMILARITY_THRESHOLD = 0.85
CLARIFICATION_ATTEMPT_THRESHOLD = 3
UNRESOLVED_TURN_THRESHOLD = 4
MAX_CONSECUTIVE_LOOPS = 2

# Frustration indicators
FRUSTRATION_KEYWORDS = [
    "still not working", "same problem", "you already said that", "not fixed",
    "again", "still broken", "repeated", "useless", "not helping", "frustrated",
    "angry", "unacceptable", "ridiculous", "enough", "give up", "waste of time",
    "legal action", "lawsuit", "sue", "complaint", "report", "terrible service",
    "completely broken", "still the same", "nothing changed", "still stuck",
]

CLARIFICATION_KEYWORDS = [
    "can you clarify", "what do you mean", "can you explain", "i don't understand",
    "need more details", "specifically", "exactly", "what exactly", "be more specific"
]




class LoopRiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recommended recovery actions."""
    CONTINUE_NORMAL = "continue_normal_flow"
    CLARIFICATION_REQUEST = "clarification_request"
    RETRIEVAL_REFRESH = "retrieval_refresh"
    ESCALATION_RECOMMENDED = "escalation_recommended"
    REGENERATE_RESPONSE = "regenerate_response"
    REFRESH_RETRIEVAL = "refresh_retrieval"
    ASK_FOR_CONTEXT = "ask_for_specific_context"
    ESCALATE_TO_HUMAN = "escalate_to_human"


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    turn_id: int
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None
    topic: Optional[str] = None
    resolved: bool = False
    retrieval_source: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class PreLoopResult:
    """Pre-loop detection result."""
    loop_detected: bool
    loop_risk_score: float
    loop_risk_level: str
    matched_turns: List[int]
    reason: str
    recovery_action: str
    requires_escalation: bool
    frustration_signals: List[str]
    similarity_scores: Dict[str, float]
    state_summary: Dict[str, Any]


@dataclass
class PostLoopResult:
    """Post-loop detection result."""
    post_loop_detected: bool
    post_loop_score: float
    loop_risk_level: str
    reason: str
    response_diversity_score: float
    recovery_action: str
    requires_escalation: bool
    repeated_patterns: List[str]
    similarity_scores: Dict[str, float]
    state_summary: Dict[str, Any]


# ──────────────────────────────────────────────────────────────
# CONVERSATION STATE MANAGEMENT
# ──────────────────────────────────────────────────────────────

class ConversationState:
    """Lightweight stateful tracking of conversation progress."""
    
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.active_issue: Optional[str] = None
        self.unresolved_turn_count: int = 0
        self.clarification_attempts: int = 0
        self.repeated_topics: List[str] = []
        self.escalation_history: List[str] = []
        self.retrieval_sources_used: List[str] = []
        self.previous_responses: List[str] = []
        self.loop_detection_history: List[Dict[str, Any]] = []
    
    def add_turn(self, role: str, content: str, topic: Optional[str] = None,
                 resolved: bool = False, retrieval_source: Optional[str] = None,
                 confidence: Optional[float] = None) -> ConversationTurn:
        """Add a new turn to conversation state."""
        turn = ConversationTurn(
            turn_id=len(self.turns),
            role=role,
            content=content,
            topic=topic,
            resolved=resolved,
            retrieval_source=retrieval_source,
            confidence=confidence
        )
        self.turns.append(turn)
        
        # Update state
        if role == "user":
            if not resolved:
                self.unresolved_turn_count += 1
            if topic:
                if topic not in self.repeated_topics:
                    self.repeated_topics.append(topic)
        elif role == "assistant":
            self.previous_responses.append(content)
            if retrieval_source:
                self.retrieval_sources_used.append(retrieval_source)
        
        return turn
    
    def record_clarification_attempt(self):
        """Track clarification attempts."""
        self.clarification_attempts += 1
    
    def record_escalation(self, reason: str):
        """Track escalation events."""
        self.escalation_history.append(reason)
    
    def reset_unresolved_count(self):
        """Reset when issue is resolved."""
        self.unresolved_turn_count = 0
    
    def get_recent_turns(self, window: int = 5, role: Optional[str] = None) -> List[ConversationTurn]:
        """Get recent turns, optionally filtered by role."""
        recent = self.turns[-window:]
        if role:
            return [t for t in recent if t.role == role]
        return recent
    
    def get_summary(self) -> Dict[str, Any]:
        """Get state summary for logging."""
        return {
            "total_turns": len(self.turns),
            "active_issue": self.active_issue,
            "unresolved_turn_count": self.unresolved_turn_count,
            "clarification_attempts": self.clarification_attempts,
            "repeated_topics": self.repeated_topics,
            "escalation_history": self.escalation_history,
            "retrieval_sources_used": list(set(self.retrieval_sources_used)),
        }




class SemanticMatcher:
    """Semantic similarity matching using embeddings."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"✓ SemanticMatcher initialized with {model_name}")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
    
    def embed(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Embed text using sentence-transformers."""
        if self.model is None:
            return None
        
        # Check cache
        if use_cache and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            if use_cache:
                self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        if emb1 is None or emb2 is None:
            # Fallback: character-level similarity
            return self._fallback_similarity(text1, text2)
        
        # Normalize and compute cosine similarity
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        return float(np.dot(emb1_norm, emb2_norm))
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using character overlap."""
        s1 = set(text1.lower().split())
        s2 = set(text2.lower().split())
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# FRUSTRATION & SIGNAL DETECTION
# ──────────────────────────────────────────────────────────────

def detect_frustration_signals(text: str) -> Tuple[bool, List[str]]:
    """
    Detect frustration indicators in user message.
    Returns (has_frustration, signal_list).
    """
    text_lower = text.lower()
    detected_signals = []
    
    for keyword in FRUSTRATION_KEYWORDS:
        if keyword in text_lower:
            detected_signals.append(keyword)
    
    has_frustration = len(detected_signals) > 0
    return has_frustration, detected_signals


def detect_clarification_request(text: str) -> bool:
    """Detect if user is asking for clarification."""
    text_lower = text.lower()
    for keyword in CLARIFICATION_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def extract_topic(text: str) -> str:
    """Extract probable topic from text."""
    text_lower = text.lower()
    
    topics = {
        "billing": ["charge", "billing", "payment", "invoice", "refund", "bill", "price"],
        "api": ["api", "request", "rate limit", "quota", "endpoint", "token"],
        "account": ["account", "login", "password", "email", "verification", "reset"],
        "technical": ["error", "bug", "broken", "not working", "crash", "fail"],
        "feature": ["feature", "function", "capability", "upgrade", "downgrade"],
    }
    
    for topic, keywords in topics.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    
    return "general"


# ──────────────────────────────────────────────────────────────
# PRE-LOOP DETECTION
# ──────────────────────────────────────────────────────────────

def detect_pre_loop(
    user_query: str,
    conversation_state: ConversationState,
    matcher: SemanticMatcher,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> PreLoopResult:
    """
    Pre-loop detection: Detect stuck user conversations BEFORE generation.
    
    Analyzes:
      • Semantic similarity with recent user queries
      • Frustration signal accumulation
      • Topic stagnation
      • Unresolved turn count
    
    Returns PreLoopResult with risk assessment and recovery action.
    """
    print(f"\n{'─'*70}")
    print(f"🔄 PRE-LOOP DETECTION: {user_query[:60]}...")
    print(f"{'─'*70}")
    
    loop_detected = False
    loop_risk_score = 0.0
    matched_turns = []
    reasons = []
    similarity_scores = {}
    
    # Signal 1: Semantic similarity with recent user queries
    print("\n1️⃣  Semantic Repetition Detection")
    recent_user_turns = conversation_state.get_recent_turns(window=6, role="user")
    
    for turn in recent_user_turns[:-1]:  # Exclude current query
        sim = matcher.similarity(user_query, turn.content)
        similarity_scores[f"turn_{turn.turn_id}"] = sim
        
        if sim > similarity_threshold:
            loop_detected = True
            matched_turns.append(turn.turn_id)
            loop_risk_score += 0.35
            reasons.append(f"Semantic match with turn {turn.turn_id} (sim: {sim:.3f})")
            print(f"  ⚠️  HIGH similarity with turn {turn.turn_id}: {sim:.3f}")
    
    # Signal 2: Frustration signals
    print("\n2️⃣  Frustration Signal Detection")
    has_frustration, frustration_signals = detect_frustration_signals(user_query)
    
    if has_frustration:
        loop_risk_score += 0.25
        reasons.append(f"Frustration signals detected: {', '.join(frustration_signals)}")
        print(f"  ⚠️  Frustration signals: {frustration_signals}")
    
    # Signal 3: Topic stagnation
    print("\n3️⃣  Topic Stagnation Detection")
    current_topic = extract_topic(user_query)
    
    if conversation_state.repeated_topics:
        if current_topic == conversation_state.repeated_topics[-1]:
            stagnation_count = conversation_state.repeated_topics.count(current_topic)
            if stagnation_count >= 2:
                loop_risk_score += 0.2
                reasons.append(f"Topic '{current_topic}' repeated {stagnation_count} times")
                print(f"  ⚠️  Topic stagnation: '{current_topic}' appears {stagnation_count} times")
    
    # Signal 4: Unresolved turn accumulation
    print("\n4️⃣  Unresolved Turn Accumulation")
    if conversation_state.unresolved_turn_count >= UNRESOLVED_TURN_THRESHOLD:
        loop_risk_score += 0.2
        reasons.append(f"{conversation_state.unresolved_turn_count} unresolved turns")
        print(f"  ⚠️  Unresolved turns: {conversation_state.unresolved_turn_count}")
    
    # Signal 5: Repeated clarification failures
    print("\n5️⃣  Clarification Failure Detection")
    if conversation_state.clarification_attempts >= CLARIFICATION_ATTEMPT_THRESHOLD:
        loop_risk_score += 0.15
        reasons.append(f"Multiple clarification attempts ({conversation_state.clarification_attempts})")
        print(f"  ⚠️  Clarification attempts: {conversation_state.clarification_attempts}")
    
    # Determine risk level
    loop_risk_level = _score_to_risk_level(loop_risk_score)
    
    # Determine recovery action
    requires_escalation = loop_risk_level in [LoopRiskLevel.HIGH.value, LoopRiskLevel.CRITICAL.value]
    recovery_action = _determine_recovery_action_pre(
        loop_detected, loop_risk_score, has_frustration, 
        conversation_state.unresolved_turn_count
    )
    
    # Build result
    print(f"\n✓ Pre-loop Risk Score: {loop_risk_score:.3f} ({loop_risk_level.upper()})")
    print(f"✓ Recovery Action: {recovery_action}")
    if requires_escalation:
        print(f"🚨 ESCALATION RECOMMENDED")
    
    result = PreLoopResult(
        loop_detected=loop_detected,
        loop_risk_score=loop_risk_score,
        loop_risk_level=loop_risk_level,
        matched_turns=matched_turns,
        reason="; ".join(reasons) if reasons else "No loop patterns detected",
        recovery_action=recovery_action,
        requires_escalation=requires_escalation,
        frustration_signals=frustration_signals,
        similarity_scores=similarity_scores,
        state_summary=conversation_state.get_summary(),
    )
    
    return result


def _score_to_risk_level(score: float) -> str:
    """Convert numerical score (0-1) to risk level."""
    if score < 0.3:
        return LoopRiskLevel.LOW.value
    elif score < 0.6:
        return LoopRiskLevel.MEDIUM.value
    elif score < 0.85:
        return LoopRiskLevel.HIGH.value
    else:
        return LoopRiskLevel.CRITICAL.value


def _determine_recovery_action_pre(
    loop_detected: bool,
    risk_score: float,
    has_frustration: bool,
    unresolved_count: int
) -> str:
    """Determine recovery action for pre-loop."""
    if risk_score >= 0.85:
        return RecoveryAction.ESCALATION_RECOMMENDED.value
    elif risk_score >= 0.6:
        if has_frustration:
            return RecoveryAction.ESCALATION_RECOMMENDED.value
        else:
            return RecoveryAction.RETRIEVAL_REFRESH.value
    elif risk_score >= 0.3:
        if has_frustration or unresolved_count >= 3:
            return RecoveryAction.CLARIFICATION_REQUEST.value
        else:
            return RecoveryAction.RETRIEVAL_REFRESH.value
    else:
        return RecoveryAction.CONTINUE_NORMAL.value


# ──────────────────────────────────────────────────────────────
# POST-LOOP DETECTION
# ──────────────────────────────────────────────────────────────

def detect_post_loop(
    generated_response: str,
    conversation_state: ConversationState,
    matcher: SemanticMatcher,
    similarity_threshold: float = POST_LOOP_SIMILARITY_THRESHOLD
) -> PostLoopResult:
    """
    Post-loop detection: Detect repeated assistant responses AFTER generation.
    
    Analyzes:
      • Semantic similarity with recent assistant responses
      • Response diversity
      • Repeated clarification patterns
      • Repeated retrieval context reuse
    
    Returns PostLoopResult with risk assessment and recovery action.
    """
    print(f"\n{'─'*70}")
    print(f"🔄 POST-LOOP DETECTION: {generated_response[:60]}...")
    print(f"{'─'*70}")
    
    post_loop_detected = False
    post_loop_score = 0.0
    repeated_patterns = []
    similarity_scores = {}
    
    # Signal 1: Semantic similarity with recent assistant responses
    print("\n1️⃣  Response Repetition Detection")
    recent_assistant_turns = conversation_state.get_recent_turns(window=5, role="assistant")
    
    max_similarity = 0.0
    for turn in recent_assistant_turns:
        sim = matcher.similarity(generated_response, turn.content)
        similarity_scores[f"turn_{turn.turn_id}"] = sim
        max_similarity = max(max_similarity, sim)
        
        if sim > similarity_threshold:
            post_loop_detected = True
            post_loop_score += 0.4
            repeated_patterns.append(f"Similar to turn {turn.turn_id} (sim: {sim:.3f})")
            print(f"  ⚠️  HIGH similarity with turn {turn.turn_id}: {sim:.3f}")
    
    response_diversity_score = 1.0 - max_similarity
    
    # Signal 2: Repeated clarification patterns
    print("\n2️⃣  Repeated Clarification Pattern Detection")
    clarification_keywords_in_response = sum(
        1 for kw in CLARIFICATION_KEYWORDS 
        if kw in generated_response.lower()
    )
    
    recent_clarification_count = sum(
        1 for turn in recent_assistant_turns 
        if any(kw in turn.content.lower() for kw in CLARIFICATION_KEYWORDS)
    )
    
    if clarification_keywords_in_response > 0 and recent_clarification_count >= 2:
        post_loop_score += 0.25
        repeated_patterns.append("Repeated clarification pattern")
        print(f"  ⚠️  Clarification being repeated (recent: {recent_clarification_count})")
    
    # Signal 3: Repeated retrieval context reuse
    print("\n3️⃣  Retrieval Context Reuse Detection")
    if conversation_state.retrieval_sources_used:
        last_source = conversation_state.retrieval_sources_used[-1] if conversation_state.retrieval_sources_used else None
        repeated_source_count = conversation_state.retrieval_sources_used.count(last_source)
        
        if repeated_source_count >= 3:
            post_loop_score += 0.2
            repeated_patterns.append(f"Retrieval source '{last_source}' reused {repeated_source_count} times")
            print(f"  ⚠️  Retrieval context reused {repeated_source_count} times from '{last_source}'")
    
    # Signal 4: Consecutive recovery attempts
    print("\n4️⃣  Consecutive Recovery Attempt Detection")
    if conversation_state.loop_detection_history:
        recent_loops = conversation_state.loop_detection_history[-3:]
        recovery_attempts = sum(1 for l in recent_loops if l.get("recovery_attempted"))
        
        if recovery_attempts >= 2:
            post_loop_score += 0.15
            repeated_patterns.append(f"Multiple recovery attempts ({recovery_attempts})")
            print(f"  ⚠️  Multiple recovery attempts detected ({recovery_attempts})")
    
    # Determine risk level
    loop_risk_level = _score_to_risk_level(post_loop_score)
    
    # Determine recovery action
    requires_escalation = loop_risk_level in [LoopRiskLevel.HIGH.value, LoopRiskLevel.CRITICAL.value]
    recovery_action = _determine_recovery_action_post(
        post_loop_detected, post_loop_score, recent_clarification_count
    )
    
    print(f"\n✓ Post-loop Risk Score: {post_loop_score:.3f} ({loop_risk_level.upper()})")
    print(f"✓ Response Diversity Score: {response_diversity_score:.3f}")
    print(f"✓ Recovery Action: {recovery_action}")
    if requires_escalation:
        print(f"🚨 ESCALATION RECOMMENDED")
    
    result = PostLoopResult(
        post_loop_detected=post_loop_detected,
        post_loop_score=post_loop_score,
        loop_risk_level=loop_risk_level,
        reason="; ".join(repeated_patterns) if repeated_patterns else "No repeated patterns detected",
        response_diversity_score=response_diversity_score,
        recovery_action=recovery_action,
        requires_escalation=requires_escalation,
        repeated_patterns=repeated_patterns,
        similarity_scores=similarity_scores,
        state_summary=conversation_state.get_summary(),
    )
    
    return result


def _determine_recovery_action_post(
    post_loop_detected: bool,
    risk_score: float,
    clarification_count: int
) -> str:
    """Determine recovery action for post-loop."""
    if risk_score >= 0.85:
        return RecoveryAction.ESCALATE_TO_HUMAN.value
    elif risk_score >= 0.6:
        if clarification_count >= 2:
            return RecoveryAction.ESCALATE_TO_HUMAN.value
        else:
            return RecoveryAction.REFRESH_RETRIEVAL.value
    elif risk_score >= 0.3:
        return RecoveryAction.REGENERATE_RESPONSE.value
    else:
        return RecoveryAction.CONTINUE_NORMAL.value


# ──────────────────────────────────────────────────────────────
# COMBINED LOOP RISK ENGINE
# ──────────────────────────────────────────────────────────────

def compute_combined_loop_risk(
    pre_loop_result: Optional[PreLoopResult],
    post_loop_result: Optional[PostLoopResult],
    state: ConversationState
) -> Tuple[float, str, bool]:
    """
    Compute combined loop risk score across pre and post detection.
    Returns (risk_score, risk_level, requires_escalation).
    """
    combined_score = 0.0
    reasons = []
    
    if pre_loop_result:
        combined_score += 0.5 * pre_loop_result.loop_risk_score
        if pre_loop_result.requires_escalation:
            reasons.append("Pre-loop escalation triggered")
    
    if post_loop_result:
        combined_score += 0.5 * post_loop_result.post_loop_score
        if post_loop_result.requires_escalation:
            reasons.append("Post-loop escalation triggered")
    
    # Add weight for escalation history
    if state.escalation_history:
        combined_score += 0.1 * len(state.escalation_history)
    
    # Normalize to [0, 1]
    combined_score = min(combined_score, 1.0)
    
    risk_level = _score_to_risk_level(combined_score)
    requires_escalation = combined_score >= 0.65
    
    return combined_score, risk_level, requires_escalation

# ──────────────────────────────────────────────────────────────
# ORCHESTRATION WRAPPER — append this to the END of loop_detection.py
# ──────────────────────────────────────────────────────────────

class LoopDetector:
    def __init__(self):
        self.matcher = SemanticMatcher()

    def initialize(self) -> bool:
        return True

    def detect_pre_loop(self, query: str, conversation_history: list) -> dict:
        state = ConversationState()
        for msg in conversation_history:
            state.add_turn(msg.get("role", "user"), msg.get("content", ""))
        result = detect_pre_loop(query, state, self.matcher)
        return asdict(result)

    def detect_post_loop(self, generated_response: str, assistant_history: list) -> dict:
        state = ConversationState()
        for content in assistant_history:
            state.add_turn("assistant", content)
        result = detect_post_loop(generated_response, state, self.matcher)
        return asdict(result)
# ──────────────────────────────────────────────────────────────
# LOGGING & RESULT FORMATTING
# ──────────────────────────────────────────────────────────────

def print_pre_loop_result(result: PreLoopResult):
    """Pretty-print pre-loop detection result."""
    print(f"\n{'═'*70}")
    print(f"📊 PRE-LOOP DETECTION RESULT")
    print(f"{'═'*70}")
    print(f"Loop Detected: {result.loop_detected}")
    print(f"Risk Score: {result.loop_risk_score:.3f}")
    print(f"Risk Level: {result.loop_risk_level.upper()}")
    print(f"Requires Escalation: {result.requires_escalation}")
    print(f"Recovery Action: {result.recovery_action}")
    print(f"\nReason: {result.reason}")
    
    if result.frustration_signals:
        print(f"\nFrustration Signals:")
        for signal in result.frustration_signals:
            print(f"  • {signal}")
    
    if result.matched_turns:
        print(f"\nMatched Turns: {result.matched_turns}")
    
    if result.similarity_scores:
        print(f"\nSimilarity Scores:")
        for turn_id, score in result.similarity_scores.items():
            print(f"  {turn_id}: {score:.3f}")
    
    print(f"{'═'*70}\n")


def print_post_loop_result(result: PostLoopResult):
    """Pretty-print post-loop detection result."""
    print(f"\n{'═'*70}")
    print(f"📊 POST-LOOP DETECTION RESULT")
    print(f"{'═'*70}")
    print(f"Loop Detected: {result.post_loop_detected}")
    print(f"Risk Score: {result.post_loop_score:.3f}")
    print(f"Risk Level: {result.loop_risk_level.upper()}")
    print(f"Requires Escalation: {result.requires_escalation}")
    print(f"Recovery Action: {result.recovery_action}")
    print(f"Response Diversity Score: {result.response_diversity_score:.3f}")
    print(f"\nReason: {result.reason}")
    
    if result.repeated_patterns:
        print(f"\nRepeated Patterns:")
        for pattern in result.repeated_patterns:
            print(f"  • {pattern}")
    
    if result.similarity_scores:
        print(f"\nSimilarity Scores:")
        for turn_id, score in result.similarity_scores.items():
            print(f"  {turn_id}: {score:.3f}")
    
    print(f"{'═'*70}\n")


# # ──────────────────────────────────────────────────────────────
# # DEMO TEST CASES
# # ──────────────────────────────────────────────────────────────

# def run_demo():
#     """Run comprehensive demo of loop detection system."""
    
#     print("\n" + "█"*70)
#     print("█ NOVADESK AI — LOOP DETECTION SYSTEM DEMO")
#     print("█"*70)
    
#     # Initialize
#     state = ConversationState()
#     matcher = SemanticMatcher()
    
#     # Demo 1: Repeated billing issue
#     print("\n\n" + "▼"*70)
#     print("DEMO 1: REPEATED BILLING ISSUE")
#     print("▼"*70)
    
#     state.add_turn("user", "Why was I charged twice this month?", topic="billing", resolved=False)
#     state.add_turn("assistant", "Let me help with your billing issue. Can you provide your account ID?", 
#                    retrieval_source="faq_exact", confidence=0.95)
#     state.add_turn("user", "I still don't understand. Why the duplicate charge?", topic="billing", resolved=False)
    
#     pre_result1 = detect_pre_loop(
#         "Why am I still being charged twice?",
#         state,
#         matcher
#     )
#     print_pre_loop_result(pre_result1)
    
#     state.add_turn("user", "Why am I still being charged twice?", topic="billing", resolved=False)
#     state.record_clarification_attempt()
    
#     # Demo 2: User expressing frustration
#     print("\n\n" + "▼"*70)
#     print("DEMO 2: ESCALATING FRUSTRATION")
#     print("▼"*70)
    
#     state.add_turn("assistant", "I understand. Let me check the billing records.", 
#                    retrieval_source="bm25", confidence=0.72)
#     state.add_turn("user", "This is still not working! I'm extremely frustrated!", topic="billing", resolved=False)
    
#     pre_result2 = detect_pre_loop(
#         "I will take legal action if this is not resolved immediately!",
#         state,
#         matcher
#     )
#     print_pre_loop_result(pre_result2)
    
#     if pre_result2.requires_escalation:
#         state.record_escalation("High frustration + legal threat detected in pre-loop")
    
#     # Demo 3: Post-loop detection - repeated response
#     print("\n\n" + "▼"*70)
#     print("DEMO 3: REPEATED ASSISTANT RESPONSES")
#     print("▼"*70)
    
#     state.add_turn("user", "Still broken. Can you clarify the refund process?", topic="billing", resolved=False)
#     state.record_clarification_attempt()
    
#     response1 = "I appreciate your patience. Could you provide more specific details about the charge?"
#     state.add_turn("assistant", response1, retrieval_source="semantic", confidence=0.58)
    
#     response2 = "I understand your concern. Could you provide more specific details about when the charge occurred?"
#     pre_result3 = detect_pre_loop(
#         "Still the same problem!",
#         state,
#         matcher
#     )
#     print_pre_loop_result(pre_result3)
    
#     post_result1 = detect_post_loop(response2, state, matcher)
#     print_post_loop_result(post_result1)
    
#     if post_result1.requires_escalation:
#         state.record_escalation("Repeated clarification pattern detected in post-loop")
    
#     # Demo 4: Recovery through retrieval refresh
#     print("\n\n" + "▼"*70)
#     print("DEMO 4: RECOVERY THROUGH RETRIEVAL REFRESH")
#     print("▼"*70)
    
#     state.add_turn("assistant", 
#                    "Let me get fresh information from our refund policy database...",
#                    retrieval_source="refresh", confidence=0.62)
#     state.add_turn("user", "Okay, I'm willing to give this another chance.", topic="billing", resolved=True)
#     state.reset_unresolved_count()
    
#     response3 = "Thank you for your patience. Based on our refund policy, here's what we can do..."
#     state.add_turn("assistant", response3, retrieval_source="refund_policy", confidence=0.88)
    
#     pre_result4 = detect_pre_loop(
#         "What are my options?",
#         state,
#         matcher
#     )
#     print_pre_loop_result(pre_result4)
    
#     print("✓ Loop successfully resolved through recovery orchestration")
    
#     # Demo 5: Extended conversation requiring escalation
#     print("\n\n" + "▼"*70)
#     print("DEMO 5: EXTENDED CONVERSATION REQUIRING ESCALATION")
#     print("▼"*70)
    
#     state2 = ConversationState()
#     state2.add_turn("user", "My account is broken.", topic="technical", resolved=False)
#     state2.add_turn("assistant", "Let me help troubleshoot. Can you describe the issue?", 
#                     retrieval_source="faq_exact", confidence=0.70)
#     state2.add_turn("user", "My account is still broken.", topic="technical", resolved=False)
#     state2.record_clarification_attempt()
#     state2.add_turn("assistant", "Can you provide more details about what's broken?", 
#                     retrieval_source="semantic", confidence=0.55)
#     state2.add_turn("user", "My account is STILL broken after your help!", topic="technical", resolved=False)
#     state2.record_clarification_attempt()
#     state2.add_turn("user", "This is ridiculous. My account has been broken for days!", topic="technical", resolved=False)
#     state2.record_clarification_attempt()
    
#     pre_result5 = detect_pre_loop(
#         "I need immediate escalation. This is unacceptable.",
#         state2,
#         matcher
#     )
#     print_pre_loop_result(pre_result5)
    
#     combined_score, risk_level, escalate = compute_combined_loop_risk(pre_result5, None, state2)
    
#     print(f"\n{'─'*70}")
#     print(f"COMBINED RISK ASSESSMENT")
#     print(f"{'─'*70}")
#     print(f"Combined Risk Score: {combined_score:.3f}")
#     print(f"Risk Level: {risk_level.upper()}")
#     print(f"Requires Escalation: {escalate}")
#     print(f"{'─'*70}\n")
    
#     if escalate:
#         print("🚨 ESCALATION INITIATED")
#         print("   → Route to Senior Support Agent")
#         print("   → Attach full conversation history")
#         print("   → Flag for engineering review")


# if __name__ == "__main__":
#     run_demo()
