from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
from datetime import datetime


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

LEGAL_THREAT_KEYWORDS = [
    "lawyer", "attorney", "lawsuit", "legal action", "sue", "court",
    "regulatory complaint", "ftc", "consumer protection", "report to",
    "compliance", "violation", "attorney general"
]

ESCALATION_KEYWORDS = [
    "urgent", "emergency", "critical", "immediate", "asap",
    "unacceptable", "ridiculous", "enough", "fed up", "angry"
]

REFUND_DISPUTE_KEYWORDS = [
    "refund", "money back", "charge back", "dispute", "fraud",
    "unauthorized", "stolen"
]

ENTERPRISE_KEYWORDS = [
    "enterprise", "sla", "msa", "contract", "agreement", "dedicated",
    "support engineer", "critical account"
]


# ──────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────────────

class EscalationPriority(str, Enum):
    """Escalation priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationStage(str, Enum):
    """When escalation triggers."""
    PRE_GENERATION = "pre_generation"
    POST_VALIDATION = "post_validation"


class HandoffType(str, Enum):
    """Type of human escalation needed."""
    BILLING_SUPPORT = "billing_support"
    LEGAL_COMPLIANCE = "legal_compliance"
    ENTERPRISE_SUPPORT = "enterprise_support"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_SUPPORT = "general_support"


@dataclass
class PreEscalationResult:
    """Pre-generation escalation evaluation."""
    requires_escalation: bool
    escalation_stage: str
    priority: str
    reason: str
    handoff_type: str
    confidence: float


@dataclass
class PostEscalationResult:
    """Post-validation escalation evaluation."""
    requires_escalation: bool
    escalation_stage: str
    priority: str
    reason: str
    recovery_summary: str
    confidence: float


@dataclass
class HumanHandoff:
    """Human escalation ticket."""
    ticket_id: str
    escalation_reason: str
    priority: str
    conversation_summary: str
    active_issue: str
    handoff_context: Dict[str, Any]
    assigned_queue: str
    recommended_agent_type: str


@dataclass
class EscalationResult:
    """Complete escalation orchestration result."""
    escalation_required: bool
    escalation_stage: str
    priority: str
    reason: str
    recommended_action: str
    human_handoff: Optional[Dict[str, Any]]
    recovery_attempted: bool
    system_summary: Dict[str, float]


# ──────────────────────────────────────────────────────────────
# RISK DETECTION UTILITIES
# ──────────────────────────────────────────────────────────────

def detect_legal_threat(query: str, conversation_history: List[Dict]) -> float:
    """Detect legal threat indicators. Returns risk [0, 1]."""
    text = (query + " " + " ".join([m.get("content", "") for m in conversation_history])).lower()
    
    threat_count = sum(1 for kw in LEGAL_THREAT_KEYWORDS if kw in text)
    if threat_count == 0:
        return 0.0
    
    return min(1.0, 0.3 + 0.2 * threat_count)


def detect_refund_dispute(query: str) -> float:
    """Detect refund dispute indicators. Returns risk [0, 1]."""
    text = query.lower()
    dispute_count = sum(1 for kw in REFUND_DISPUTE_KEYWORDS if kw in text)
    return min(1.0, 0.2 * dispute_count) if dispute_count > 0 else 0.0


def detect_enterprise_customer(conversation_state: Dict) -> bool:
    """Check if customer is enterprise tier."""
    return conversation_state.get("customer_tier", "").lower() == "enterprise"


def detect_frustration(query: str, conversation_history: List[Dict]) -> float:
    """Detect user frustration signals. Returns score [0, 1]."""
    text = (query + " " + " ".join([m.get("content", "") for m in conversation_history[-5:]])).lower()
    
    escalation_count = sum(1 for kw in ESCALATION_KEYWORDS if kw in text)
    repetition = sum(1 for m in conversation_history[-5:] if "still" in m.get("content", "").lower())
    
    frustration = min(1.0, 0.15 * escalation_count + 0.1 * repetition)
    return frustration


def calculate_priority(
    legal_threat: float,
    refund_dispute: float,
    loop_risk: float,
    hallucination_risk: float,
    frustration: float,
    unresolved_turns: int
) -> str:
    """Calculate escalation priority from multiple signals."""
    
    # Critical signals
    if legal_threat > 0.7 or (refund_dispute > 0.5 and frustration > 0.6):
        return EscalationPriority.CRITICAL.value
    
    # High signals
    if legal_threat > 0.4 or loop_risk > 0.8 or hallucination_risk > 0.75 or unresolved_turns > 5:
        return EscalationPriority.HIGH.value
    
    # Medium signals
    if hallucination_risk > 0.60 or frustration > 0.5 or unresolved_turns > 3:
        return EscalationPriority.MEDIUM.value
    
    return EscalationPriority.LOW.value


# ──────────────────────────────────────────────────────────────
# STAGE 1: PRE-GENERATION ESCALATION
# ──────────────────────────────────────────────────────────────

def evaluate_pre_generation_escalation(
    query: str,
    intent_data: Optional[Dict] = None,
    loop_data: Optional[Dict] = None,
    conversation_state: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> PreEscalationResult:
    """
    Early escalation detection BEFORE retrieval/generation.
    Prevents unsafe automated responses before they happen.
    """
    
    if intent_data is None:
        intent_data = {}
    if loop_data is None:
        loop_data = {}
    if conversation_state is None:
        conversation_state = {}
    if conversation_history is None:
        conversation_history = []

    print(f"\n1️⃣  PRE-GENERATION ESCALATION CHECK")

    # Detect risks
    legal_threat = detect_legal_threat(query, conversation_history)
    refund_dispute = detect_refund_dispute(query)
    frustration = detect_frustration(query, conversation_history)
    pre_loop_risk = loop_data.get("loop_risk_score", 0.0)
    unresolved_turns = conversation_state.get("unresolved_turn_count", 0)

    # Determine escalation
    requires_escalation = False
    reason = ""
    handoff_type = HandoffType.GENERAL_SUPPORT.value
    confidence = 0.0

    # Signal 1: Legal threat
    if legal_threat > 0.6:
        requires_escalation = True
        reason = "Legal threat detected"
        handoff_type = HandoffType.LEGAL_COMPLIANCE.value
        confidence = legal_threat

    # Signal 2: Refund dispute + frustration
    elif refund_dispute > 0.5 and frustration > 0.5:
        requires_escalation = True
        reason = "Refund dispute with elevated frustration"
        handoff_type = HandoffType.BILLING_SUPPORT.value
        confidence = (refund_dispute + frustration) / 2

    # Signal 3: Critical loop risk pre-generation
    elif pre_loop_risk > 0.85:
        requires_escalation = True
        reason = "Critical pre-loop detection - conversation stuck"
        handoff_type = HandoffType.GENERAL_SUPPORT.value
        confidence = pre_loop_risk

    # Signal 4: Enterprise escalation
    elif detect_enterprise_customer(conversation_state) and (refund_dispute > 0.3 or legal_threat > 0.3):
        requires_escalation = True
        reason = "Enterprise customer with dispute"
        handoff_type = HandoffType.ENTERPRISE_SUPPORT.value
        confidence = 0.75

    # Signal 5: Repeated unresolved turns + frustration
    elif unresolved_turns > 5 and frustration > 0.5:
        requires_escalation = True
        reason = f"Repeated unresolved issue ({unresolved_turns} turns) with frustration"
        handoff_type = HandoffType.TECHNICAL_SUPPORT.value
        confidence = min(1.0, 0.5 + frustration)

    if requires_escalation:
        print(f"  🚨 PRE-ESCALATION TRIGGERED: {reason}")
        print(f"     Handoff: {handoff_type}, Confidence: {confidence:.3f}")
    else:
        print(f"  ✓ No pre-escalation needed")

    return PreEscalationResult(
        requires_escalation=requires_escalation,
        escalation_stage=EscalationStage.PRE_GENERATION.value,
        priority=calculate_priority(legal_threat, refund_dispute, pre_loop_risk, 0.0, frustration, unresolved_turns),
        reason=reason,
        handoff_type=handoff_type,
        confidence=confidence
    )


# ──────────────────────────────────────────────────────────────
# STAGE 2: POST-VALIDATION ESCALATION
# ──────────────────────────────────────────────────────────────

def evaluate_post_validation_escalation(
    validation_data: Dict[str, Any],
    retrieval_data: Optional[Dict] = None,
    loop_data: Optional[Dict] = None,
    conversation_state: Optional[Dict] = None
) -> PostEscalationResult:
    """
    Late escalation after validation layer detects reliability issues.
    Escalates when AI response becomes unsafe to deliver.
    """
    
    if retrieval_data is None:
        retrieval_data = {}
    if loop_data is None:
        loop_data = {}
    if conversation_state is None:
        conversation_state = {}

    print(f"\n2️⃣  POST-VALIDATION ESCALATION CHECK")

    # Extract validation signals
    confidence_score = validation_data.get("confidence_score", 0.5)
    grounding_score = validation_data.get("grounding", {}).get("grounding_score", 0.5)
    hallucination_risk = validation_data.get("hallucination", {}).get("hallucination_risk", 0.0)
    repetition_detected = validation_data.get("loop_validation", {}).get("repetition_detected", False)

    # Extract retrieval signals
    retrieval_confidence = retrieval_data.get("retrieval_confidence", 0.5)
    retrieval_source = retrieval_data.get("retrieval_source", "none")

    # Extract loop signals
    post_loop_risk = loop_data.get("post_loop_score", 0.0)
    loop_matched_turns = len(loop_data.get("matched_turns", []))

    # Determine escalation
    requires_escalation = False
    reason = ""
    recovery_summary = ""
    confidence = 0.0

    # Signal 1: Hallucination risk too high
    if hallucination_risk > 0.70:
        requires_escalation = True
        reason = "Hallucination risk critical"
        recovery_summary = "Attempted grounding validation; failed. Escalating."
        confidence = hallucination_risk

    # Signal 2: Grounding very weak
    elif grounding_score < 0.40:
        requires_escalation = True
        reason = "Response grounding failed"
        recovery_summary = "Attempted retrieval validation; no supporting context found."
        confidence = 1.0 - grounding_score

    # Signal 3: Overall confidence too low
    elif confidence_score < 0.45:
        requires_escalation = True
        reason = "System confidence critical"
        recovery_summary = f"System confidence {confidence_score:.2f} below safe threshold"
        confidence = confidence_score

    # Signal 4: Response repetition detected + high loop risk
    elif repetition_detected and post_loop_risk > 0.70:
        requires_escalation = True
        reason = "Response repetition + high loop risk"
        recovery_summary = f"Attempted regeneration; detected {loop_matched_turns} repeated turns"
        confidence = min(1.0, post_loop_risk + 0.1)

    # Signal 5: Retrieval completely failed
    elif retrieval_source == "none" and retrieval_confidence < 0.30:
        requires_escalation = True
        reason = "Retrieval system failed"
        recovery_summary = "No relevant documents retrieved; cannot ground response"
        confidence = 0.8

    # Signal 6: Post-loop risk critical
    elif post_loop_risk > 0.85:
        requires_escalation = True
        reason = "Conversational loop at critical level"
        recovery_summary = f"Conversation stuck in loop ({loop_matched_turns} matched turns)"
        confidence = post_loop_risk

    if requires_escalation:
        print(f"  🚨 POST-ESCALATION TRIGGERED: {reason}")
        print(f"     Confidence: {confidence:.3f}, Recovery: {recovery_summary}")
    else:
        print(f"  ✓ Validation passed; no escalation needed")

    return PostEscalationResult(
        requires_escalation=requires_escalation,
        escalation_stage=EscalationStage.POST_VALIDATION.value,
        priority=calculate_priority(0.0, 0.0, post_loop_risk, hallucination_risk, 0.0, 
                                   conversation_state.get("unresolved_turn_count", 0)),
        reason=reason,
        recovery_summary=recovery_summary,
        confidence=confidence
    )


# ──────────────────────────────────────────────────────────────
# RECOVERY DECISION ENGINE
# ──────────────────────────────────────────────────────────────

def decide_recovery_action(
    validation_data: Dict[str, Any],
    loop_data: Dict[str, Any],
    retrieval_data: Dict[str, Any],
    conversation_state: Dict[str, Any]
) -> str:
    """
    Before escalating, intelligently decide if recovery is still possible.
    Returns recovery action recommendation.
    """
    
    confidence_score = validation_data.get("confidence_score", 0.5)
    grounding_score = validation_data.get("grounding", {}).get("grounding_score", 0.5)
    hallucination_risk = validation_data.get("hallucination", {}).get("hallucination_risk", 0.0)
    repetition_detected = validation_data.get("loop_validation", {}).get("repetition_detected", False)
    
    retrieval_confidence = retrieval_data.get("retrieval_confidence", 0.5)
    post_loop_risk = loop_data.get("post_loop_score", 0.0)
    
    clarification_attempts = conversation_state.get("clarification_attempts", 0)

    # Try regeneration if response is repetitive
    if repetition_detected and post_loop_risk < 0.80:
        return "regenerate_response"

    # Try clarification if confidence borderline
    if confidence_score >= 0.45 and clarification_attempts < 3:
        return "clarification_request"

    # Try retrieval refresh if confidence weak but grounding okay
    if grounding_score > 0.45 and retrieval_confidence < 0.60 and post_loop_risk < 0.70:
        return "retrieval_refresh"

    # Otherwise escalate
    return "escalate_to_human"


# ──────────────────────────────────────────────────────────────
# HUMAN HANDOFF ORCHESTRATION
# ──────────────────────────────────────────────────────────────

def trigger_human_handoff(
    escalation_data: Dict[str, Any],
    conversation_state: Dict[str, Any],
    retrieval_data: Dict[str, Any],
    validation_data: Dict[str, Any],
    conversation_history: Optional[List[Dict]] = None
) -> HumanHandoff:
    """
    Simulate production-grade human-in-the-loop escalation.
    Generate escalation ticket with full context transfer.
    """
    
    if conversation_history is None:
        conversation_history = []

    # Generate ticket ID
    ticket_id = f"TICK-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Build active issue summary
    active_issue = conversation_state.get("active_issue", "Unspecified issue")
    unresolved_turns = conversation_state.get("unresolved_turn_count", 0)

    # Build conversation summary
    recent_msgs = conversation_history[-5:] if conversation_history else []
    conversation_summary = "\n".join([
        f"[{m.get('role', 'unknown').upper()}]: {m.get('content', '')[:100]}..."
        for m in recent_msgs
    ])

    # Determine queue and agent type
    escalation_stage = escalation_data.get("escalation_stage", "post_validation")
    handoff_type = escalation_data.get("handoff_type", "general_support")
    priority = escalation_data.get("priority", "medium")

    queue_mapping = {
        "billing_support": "BILLING",
        "legal_compliance": "LEGAL",
        "enterprise_support": "ENTERPRISE",
        "technical_support": "TECHNICAL",
        "general_support": "GENERAL",
    }
    assigned_queue = queue_mapping.get(handoff_type, "GENERAL")

    # Build handoff context
    handoff_context = {
        "escalation_reason": escalation_data.get("reason", ""),
        "escalation_stage": escalation_stage,
        "retrieval_source": retrieval_data.get("retrieval_source", "none"),
        "retrieval_confidence": retrieval_data.get("retrieval_confidence", 0.0),
        "grounding_score": validation_data.get("grounding", {}).get("grounding_score", 0.0),
        "hallucination_risk": validation_data.get("hallucination", {}).get("hallucination_risk", 0.0),
        "validation_confidence": validation_data.get("confidence_score", 0.0),
        "unresolved_turns": unresolved_turns,
        "customer_tier": conversation_state.get("customer_tier", "standard"),
    }

    print(f"\n3️⃣  HUMAN HANDOFF TRIGGERED")
    print(f"  Ticket ID: {ticket_id}")
    print(f"  Queue: {assigned_queue}")
    print(f"  Priority: {priority.upper()}")
    print(f"  Issue: {active_issue}")

    return HumanHandoff(
        ticket_id=ticket_id,
        escalation_reason=escalation_data.get("reason", ""),
        priority=priority,
        conversation_summary=conversation_summary,
        active_issue=active_issue,
        handoff_context=handoff_context,
        assigned_queue=assigned_queue,
        recommended_agent_type=handoff_type.replace("_", " ").title()
    )


# ──────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION FUNCTION
# ──────────────────────────────────────────────────────────────

def orchestrate_escalation(
    query: str,
    retrieval_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    loop_result: Optional[Dict[str, Any]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict]] = None,
    pre_escalation_data: Optional[Dict[str, Any]] = None
) -> EscalationResult:
    """
    Main escalation orchestration combining pre and post stages.
    
    Integrates with:
    - retrieval.py (RetrievalResult)
    - validation.py (ValidationResult)
    - loop_detection.py (PostLoopResult)
    
    Returns complete escalation decision with human handoff if needed.
    """
    
    if loop_result is None:
        loop_result = {}
    if conversation_state is None:
        conversation_state = {}
    if conversation_history is None:
        conversation_history = []

    print(f"\n{'─'*70}")
    print(f"🚨 ESCALATION ORCHESTRATION")
    print(f"{'─'*70}")

    # Stage 1: Pre-generation check (if not already done)
    if pre_escalation_data is None:
        intent_data = {}
        pre_result = evaluate_pre_generation_escalation(
            query, intent_data, loop_result, conversation_state, conversation_history
        )
    else:
        pre_result = PreEscalationResult(**asdict(pre_escalation_data) if hasattr(pre_escalation_data, '__dataclass_fields__') else pre_escalation_data)

    if pre_result.requires_escalation:
        handoff = trigger_human_handoff(
            asdict(pre_result), conversation_state, retrieval_result, validation_result, conversation_history
        )
        return EscalationResult(
            escalation_required=True,
            escalation_stage=pre_result.escalation_stage,
            priority=pre_result.priority,
            reason=pre_result.reason,
            recommended_action="escalate_to_human",
            human_handoff=asdict(handoff),
            recovery_attempted=False,
            system_summary={
                "loop_risk": loop_result.get("post_loop_score", 0.0),
                "retrieval_confidence": retrieval_result.get("retrieval_confidence", 0.0),
                "grounding_score": validation_result.get("grounding", {}).get("grounding_score", 0.0),
                "hallucination_risk": validation_result.get("hallucination", {}).get("hallucination_risk", 0.0),
            }
        )

    # Stage 2: Post-validation escalation
    post_result = evaluate_post_validation_escalation(
        validation_result, retrieval_result, loop_result, conversation_state
    )

    if post_result.requires_escalation:
        # Decide recovery before escalating
        recovery_action = decide_recovery_action(
            validation_result, loop_result, retrieval_result, conversation_state
        )
        
        # If recovery not viable, escalate
        if recovery_action == "escalate_to_human":
            handoff = trigger_human_handoff(
                asdict(post_result), conversation_state, retrieval_result, validation_result, conversation_history
            )
            return EscalationResult(
                escalation_required=True,
                escalation_stage=post_result.escalation_stage,
                priority=post_result.priority,
                reason=post_result.reason,
                recommended_action="escalate_to_human",
                human_handoff=asdict(handoff),
                recovery_attempted=False,
                system_summary={
                    "loop_risk": loop_result.get("post_loop_score", 0.0),
                    "retrieval_confidence": retrieval_result.get("retrieval_confidence", 0.0),
                    "grounding_score": validation_result.get("grounding", {}).get("grounding_score", 0.0),
                    "hallucination_risk": validation_result.get("hallucination", {}).get("hallucination_risk", 0.0),
                }
            )
        else:
            # Recovery action attempted
            return EscalationResult(
                escalation_required=False,
                escalation_stage="no_escalation",
                priority="low",
                reason="Recovery action selected",
                recommended_action=recovery_action,
                human_handoff=None,
                recovery_attempted=True,
                system_summary={
                    "loop_risk": loop_result.get("post_loop_score", 0.0),
                    "retrieval_confidence": retrieval_result.get("retrieval_confidence", 0.0),
                    "grounding_score": validation_result.get("grounding", {}).get("grounding_score", 0.0),
                    "hallucination_risk": validation_result.get("hallucination", {}).get("hallucination_risk", 0.0),
                }
            )

    # No escalation needed
    print(f"\n✅ No escalation required")
    return EscalationResult(
        escalation_required=False,
        escalation_stage="none",
        priority="low",
        reason="System confidence sufficient",
        recommended_action="continue_normal_flow",
        human_handoff=None,
        recovery_attempted=False,
        system_summary={
            "loop_risk": loop_result.get("post_loop_score", 0.0),
            "retrieval_confidence": retrieval_result.get("retrieval_confidence", 0.0),
            "grounding_score": validation_result.get("grounding", {}).get("grounding_score", 0.0),
            "hallucination_risk": validation_result.get("hallucination", {}).get("hallucination_risk", 0.0),
        }
    )


# # ──────────────────────────────────────────────────────────────
# # DEMO TEST CASES
# # ──────────────────────────────────────────────────────────────

# def run_demo():
#     """Run demo escalation scenarios."""

#     print("\n" + "█"*70)
#     print("█ NOVADESK AI — ESCALATION SYSTEM DEMO")
#     print("█"*70)

#     # Demo 1: Legal threat
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 1: Legal Threat Escalation")
#     print("▌"*70)

#     pre_result_1 = evaluate_pre_generation_escalation(
#         "I will take legal action against your company",
#         conversation_history=[{"role": "user", "content": "Charged twice"}]
#     )
#     print(f"Pre-escalation required: {pre_result_1.requires_escalation}")
#     print(f"Priority: {pre_result_1.priority}")
#     print(f"Reason: {pre_result_1.reason}")

#     # Demo 2: Refund dispute
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 2: Refund Dispute Escalation")
#     print("▌"*70)

#     pre_result_2 = evaluate_pre_generation_escalation(
#         "I want a refund and I'm furious about this",
#         loop_data={"loop_risk_score": 0.3}
#     )
#     print(f"Pre-escalation required: {pre_result_2.requires_escalation}")
#     print(f"Priority: {pre_result_2.priority}")

#     # Demo 3: Hallucination escalation
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 3: Hallucination Risk Escalation")
#     print("▌"*70)

#     validation_3 = {
#         "confidence_score": 0.35,
#         "grounding": {"grounding_score": 0.30},
#         "hallucination": {"hallucination_risk": 0.80},
#         "loop_validation": {"repetition_detected": False}
#     }

#     post_result_3 = evaluate_post_validation_escalation(validation_3)
#     print(f"Post-escalation required: {post_result_3.requires_escalation}")
#     print(f"Reason: {post_result_3.reason}")

#     # Demo 4: Loop escalation
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 4: Conversation Loop Escalation")
#     print("▌"*70)

#     loop_4 = {"post_loop_score": 0.88, "matched_turns": [2, 4, 6, 8]}
    
#     post_result_4 = evaluate_post_validation_escalation(
#         {"confidence_score": 0.55, "grounding": {}, "hallucination": {}, "loop_validation": {"repetition_detected": True}},
#         loop_data=loop_4
#     )
#     print(f"Post-escalation required: {post_result_4.requires_escalation}")
#     print(f"Recovery summary: {post_result_4.recovery_summary}")

#     # Demo 5: Enterprise escalation
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 5: Enterprise Customer Escalation")
#     print("▌"*70)

#     pre_result_5 = evaluate_pre_generation_escalation(
#         "This billing issue is unacceptable",
#         conversation_state={"customer_tier": "enterprise"}
#     )
#     print(f"Pre-escalation required: {pre_result_5.requires_escalation}")
#     print(f"Handoff type: {pre_result_5.handoff_type}")

#     print("\n\n" + "█"*70)
#     print("█ DEMO COMPLETE")
#     print("█"*70 + "\n")


# if __name__ == "__main__":
#     run_demo()
