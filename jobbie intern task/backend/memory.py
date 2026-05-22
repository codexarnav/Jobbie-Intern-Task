from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

MAX_STM_TURNS = 8
MAX_LTM_ISSUES = 10
MAX_ESCALATION_HISTORY = 5


# ──────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ──────────────────────────────────────────────────────────────

class IssueStatus(str, Enum):
    """Issue resolution status."""
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    RECURRING = "recurring"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class STMMessage:
    """Single STM turn."""
    role: str
    content: str
    timestamp: str
    topic: Optional[str] = None
    sentiment: Optional[str] = None


@dataclass
class ShortTermMemory:
    """Recent conversation context."""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    active_topic: str = ""
    clarification_attempts: int = 0
    loop_risk_score: float = 0.0
    last_retrieval_source: str = ""
    last_validation_score: float = 0.0
    current_sentiment: str = ""
    last_updated: str = ""


@dataclass
class LTMIssue:
    """Long-term memory issue record."""
    issue_id: str
    issue_type: str
    first_mention: str
    last_mention: str
    mention_count: int
    resolution_attempts: int
    escalated: bool


@dataclass
class LongTermMemory:
    """Recurring patterns and history."""
    user_id: str
    recurring_issues: List[Dict[str, Any]] = field(default_factory=list)
    previous_escalations: List[Dict[str, Any]] = field(default_factory=list)
    historical_summaries: List[str] = field(default_factory=list)
    retrieval_failures: List[Dict[str, Any]] = field(default_factory=list)
    billing_history_summary: str = ""
    preferred_support_path: str = ""


@dataclass
class ActiveIssue:
    """Current unresolved issue state."""
    issue_id: str
    issue_type: str
    resolution_status: str
    unresolved_turn_count: int
    escalation_attempted: bool
    last_recovery_action: str
    active_risk_level: str
    last_updated: str


@dataclass
class ConversationState:
    """Unified conversation state for orchestration."""
    user_id: str
    stm: Dict[str, Any]
    ltm: Dict[str, Any]
    active_issue: Optional[Dict[str, Any]]
    retrieval_history: List[Dict[str, Any]]
    validation_history: List[Dict[str, Any]]
    escalation_history: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────
# SHORT-TERM MEMORY (STM)
# ──────────────────────────────────────────────────────────────

def init_stm() -> Dict[str, Any]:
    """Initialize empty STM."""
    return {
        "recent_messages": [],
        "active_topic": "",
        "clarification_attempts": 0,
        "loop_risk_score": 0.0,
        "last_retrieval_source": "",
        "last_validation_score": 0.0,
        "current_sentiment": "",
        "last_updated": datetime.now().isoformat()
    }


def update_stm(stm: Dict[str, Any], message: Dict[str, str], topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Update STM with new message. Keep only recent turns.
    Maintains conversational continuity and state.
    """
    msg_entry = {
        "role": message.get("role", "user"),
        "content": message.get("content", ""),
        "timestamp": datetime.now().isoformat(),
        "topic": topic
    }
    
    stm["recent_messages"].append(msg_entry)
    
    # Keep only recent turns
    if len(stm["recent_messages"]) > MAX_STM_TURNS:
        stm["recent_messages"] = stm["recent_messages"][-MAX_STM_TURNS:]
    
    if topic:
        stm["active_topic"] = topic
    
    stm["last_updated"] = datetime.now().isoformat()
    
    print(f"  ✓ STM updated: {message.get('role')} | Topic: {topic}")
    
    return stm


def retrieve_stm_context(stm: Dict[str, Any]) -> Dict[str, Any]:
    """Extract STM context for orchestration."""
    return {
        "recent_turns": len(stm.get("recent_messages", [])),
        "active_topic": stm.get("active_topic", ""),
        "clarification_attempts": stm.get("clarification_attempts", 0),
        "loop_risk_score": stm.get("loop_risk_score", 0.0),
        "last_retrieval_source": stm.get("last_retrieval_source", ""),
        "validation_score": stm.get("last_validation_score", 0.0),
    }


def update_stm_clarification(stm: Dict[str, Any]) -> Dict[str, Any]:
    """Increment clarification attempt counter."""
    stm["clarification_attempts"] += 1
    print(f"  ✓ Clarification attempt #{stm['clarification_attempts']}")
    return stm


def reset_stm_clarification(stm: Dict[str, Any]) -> Dict[str, Any]:
    """Reset clarification counter on successful resolution."""
    stm["clarification_attempts"] = 0
    return stm


# ──────────────────────────────────────────────────────────────
# LONG-TERM MEMORY (LTM)
# ──────────────────────────────────────────────────────────────

def init_ltm(user_id: str) -> Dict[str, Any]:
    """Initialize LTM for user."""
    return {
        "user_id": user_id,
        "recurring_issues": [],
        "previous_escalations": [],
        "historical_summaries": [],
        "retrieval_failures": [],
        "billing_history_summary": "",
        "preferred_support_path": ""
    }


def update_ltm_issue(
    ltm: Dict[str, Any],
    issue_type: str,
    issue_details: str
) -> Dict[str, Any]:
    """
    Track recurring issue in LTM.
    Helps identify patterns for escalation and retrieval personalization.
    """
    existing = next(
        (i for i in ltm["recurring_issues"] if i.get("issue_type") == issue_type),
        None
    )
    
    if existing:
        existing["mention_count"] += 1
        existing["last_mention"] = datetime.now().isoformat()
    else:
        ltm["recurring_issues"].append({
            "issue_type": issue_type,
            "first_mention": datetime.now().isoformat(),
            "last_mention": datetime.now().isoformat(),
            "mention_count": 1,
            "details": issue_details
        })
    
    # Keep only recent issues
    if len(ltm["recurring_issues"]) > MAX_LTM_ISSUES:
        ltm["recurring_issues"] = sorted(
            ltm["recurring_issues"],
            key=lambda x: x["mention_count"],
            reverse=True
        )[:MAX_LTM_ISSUES]
    
    print(f"  ✓ LTM issue tracked: {issue_type}")
    
    return ltm


def update_ltm_escalation(
    ltm: Dict[str, Any],
    escalation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Record escalation in LTM for continuity."""
    escalation_record = {
        "timestamp": datetime.now().isoformat(),
        "reason": escalation_data.get("reason", ""),
        "priority": escalation_data.get("priority", "medium"),
        "resolved": False,
    }
    
    ltm["previous_escalations"].append(escalation_record)
    
    if len(ltm["previous_escalations"]) > MAX_ESCALATION_HISTORY:
        ltm["previous_escalations"] = ltm["previous_escalations"][-MAX_ESCALATION_HISTORY:]
    
    print(f"  ✓ LTM escalation recorded: {escalation_data.get('reason', 'unknown')}")
    
    return ltm


def update_ltm_retrieval_failure(
    ltm: Dict[str, Any],
    query: str,
    retrieval_source: str
) -> Dict[str, Any]:
    """Track retrieval failures to inform personalization."""
    ltm["retrieval_failures"].append({
        "timestamp": datetime.now().isoformat(),
        "query": query[:100],
        "source": retrieval_source
    })
    
    if len(ltm["retrieval_failures"]) > 20:
        ltm["retrieval_failures"] = ltm["retrieval_failures"][-20:]
    
    return ltm


def retrieve_ltm_context(ltm: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Retrieve LTM context for personalization.
    Helps retrieval.py prioritize relevant docs.
    """
    # Find relevant recurring issues
    relevant_issues = [
        issue for issue in ltm.get("recurring_issues", [])
        if issue.get("mention_count", 0) > 1
    ]
    
    # Escalation frequency
    escalation_count = len(ltm.get("previous_escalations", []))
    
    # Retrieval failure context
    recent_failures = ltm.get("retrieval_failures", [])[-3:]
    
    context = {
        "recurring_issues": relevant_issues,
        "escalation_history": ltm.get("previous_escalations", []),
        "escalation_frequency": escalation_count,
        "retrieval_failures": recent_failures,
        "billing_context": ltm.get("billing_history_summary", ""),
        "preferred_path": ltm.get("preferred_support_path", "")
    }
    
    if relevant_issues or escalation_count > 0:
        print(f"  ✓ LTM personalization context retrieved: {len(relevant_issues)} recurring issues")
    
    return context


# ──────────────────────────────────────────────────────────────
# ACTIVE ISSUE TRACKER
# ──────────────────────────────────────────────────────────────

def init_active_issue(issue_type: str) -> Dict[str, Any]:
    """Initialize active issue tracking."""
    issue_id = f"ISSUE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "issue_id": issue_id,
        "issue_type": issue_type,
        "resolution_status": IssueStatus.UNRESOLVED.value,
        "unresolved_turn_count": 0,
        "escalation_attempted": False,
        "last_recovery_action": "",
        "active_risk_level": RiskLevel.LOW.value,
        "last_updated": datetime.now().isoformat()
    }


def increment_unresolved_turns(active_issue: Dict[str, Any]) -> Dict[str, Any]:
    """Increment unresolved turn counter. Used for escalation decisions."""
    active_issue["unresolved_turn_count"] += 1
    active_issue["last_updated"] = datetime.now().isoformat()
    
    # Update risk level based on unresolved turns
    turns = active_issue["unresolved_turn_count"]
    if turns > 7:
        active_issue["active_risk_level"] = RiskLevel.CRITICAL.value
    elif turns > 5:
        active_issue["active_risk_level"] = RiskLevel.HIGH.value
    elif turns > 3:
        active_issue["active_risk_level"] = RiskLevel.MEDIUM.value
    
    print(f"  ✓ Unresolved turns incremented: {turns}")
    
    return active_issue


def track_recovery_action(
    active_issue: Dict[str, Any],
    recovery_action: str
) -> Dict[str, Any]:
    """Track recovery attempt for bounded escalation logic."""
    active_issue["last_recovery_action"] = recovery_action
    active_issue["last_updated"] = datetime.now().isoformat()
    
    print(f"  ✓ Recovery action tracked: {recovery_action}")
    
    return active_issue


def mark_escalation_attempted(active_issue: Dict[str, Any]) -> Dict[str, Any]:
    """Mark that escalation has been attempted."""
    active_issue["escalation_attempted"] = True
    active_issue["resolution_status"] = IssueStatus.ESCALATED.value
    active_issue["last_updated"] = datetime.now().isoformat()
    
    print(f"  ✓ Escalation marked as attempted")
    
    return active_issue


def mark_issue_resolved(active_issue: Dict[str, Any]) -> Dict[str, Any]:
    """Mark issue as resolved. Resets state."""
    active_issue["resolution_status"] = IssueStatus.RESOLVED.value
    active_issue["last_updated"] = datetime.now().isoformat()
    
    print(f"  ✓ Issue marked as RESOLVED")
    
    return active_issue


# ──────────────────────────────────────────────────────────────
# UNIFIED STATE ORCHESTRATION
# ──────────────────────────────────────────────────────────────

def init_conversation_state(user_id: str) -> Dict[str, Any]:
    """Initialize unified conversation state."""
    return {
        "user_id": user_id,
        "stm": init_stm(),
        "ltm": init_ltm(user_id),
        "active_issue": None,
        "retrieval_history": [],
        "validation_history": [],
        "escalation_history": []
    }


def update_retrieval_history(
    conv_state: Dict[str, Any],
    retrieval_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Track retrieval operations for orchestration."""
    conv_state["retrieval_history"].append({
        "timestamp": datetime.now().isoformat(),
        "source": retrieval_data.get("retrieval_source", ""),
        "confidence": retrieval_data.get("retrieval_confidence", 0.0),
        "document_count": len(retrieval_data.get("retrieved_documents", []))
    })
    
    # Update STM with latest retrieval
    conv_state["stm"]["last_retrieval_source"] = retrieval_data.get("retrieval_source", "")
    
    # Track retrieval failures in LTM
    if retrieval_data.get("retrieval_confidence", 0.0) < 0.40:
        conv_state["ltm"] = update_ltm_retrieval_failure(
            conv_state["ltm"],
            "",  # query not available here
            retrieval_data.get("retrieval_source", "")
        )
    
    return conv_state


def update_validation_history(
    conv_state: Dict[str, Any],
    validation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Track validation operations."""
    conv_state["validation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "confidence": validation_data.get("confidence_score", 0.0),
        "grounding": validation_data.get("grounding", {}).get("grounding_score", 0.0),
        "hallucination_risk": validation_data.get("hallucination", {}).get("hallucination_risk", 0.0)
    })
    
    # Update STM
    conv_state["stm"]["last_validation_score"] = validation_data.get("confidence_score", 0.0)
    
    return conv_state


def update_escalation_history(
    conv_state: Dict[str, Any],
    escalation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Track escalation events."""
    conv_state["escalation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "reason": escalation_data.get("reason", ""),
        "priority": escalation_data.get("priority", "medium"),
        "stage": escalation_data.get("escalation_stage", "")
    })
    
    # Update LTM
    conv_state["ltm"] = update_ltm_escalation(conv_state["ltm"], escalation_data)
    
    # Update active issue if exists
    if conv_state["active_issue"]:
        conv_state["active_issue"] = mark_escalation_attempted(conv_state["active_issue"])
    
    return conv_state


# ──────────────────────────────────────────────────────────────
# CONVERSATION SUMMARIZATION
# ──────────────────────────────────────────────────────────────

def summarize_conversation(conversation_history: List[Dict[str, str]]) -> str:
    """
    Generate concise conversation summary for escalation handoff.
    Used for human-in-the-loop context transfer.
    """
    if not conversation_history:
        return "No conversation history."
    
    # Extract key info
    user_msgs = [m for m in conversation_history if m.get("role") == "user"]
    assistant_msgs = [m for m in conversation_history if m.get("role") == "assistant"]
    
    summary = f"Conversation ({len(conversation_history)} turns):\n"
    summary += f"User messages: {len(user_msgs)}\n"
    summary += f"Assistant responses: {len(assistant_msgs)}\n"
    
    # Recent messages
    if user_msgs:
        summary += f"\nLatest user concern: {user_msgs[-1].get('content', '')[:150]}...\n"
    
    # Issue detection
    issues = []
    all_text = " ".join([m.get("content", "").lower() for m in user_msgs])
    
    if "refund" in all_text or "charge" in all_text:
        issues.append("Billing")
    if "error" in all_text or "broken" in all_text:
        issues.append("Technical")
    if "frustrated" in all_text or "angry" in all_text:
        issues.append("Escalation/Sentiment")
    
    if issues:
        summary += f"Detected issues: {', '.join(issues)}"
    
    return summary


# ──────────────────────────────────────────────────────────────
# MEMORY-AWARE ORCHESTRATION HELPERS
# ──────────────────────────────────────────────────────────────

def should_prioritize_retrieval_refresh(
    stm: Dict[str, Any],
    conv_state: Dict[str, Any]
) -> bool:
    """
    Determine if retrieval should be refreshed based on memory state.
    Supports bounded recovery orchestration.
    """
    # Trigger refresh if multiple clarification attempts
    if stm.get("clarification_attempts", 0) > 2:
        return True
    
    # Trigger if recent retrieval confidence low
    recent_retrievals = conv_state.get("retrieval_history", [])[-3:]
    if recent_retrievals:
        avg_confidence = sum(r.get("confidence", 0.0) for r in recent_retrievals) / len(recent_retrievals)
        if avg_confidence < 0.50:
            return True
    
    return False


def get_escalation_recommendation_from_memory(
    active_issue: Optional[Dict[str, Any]],
    ltm: Dict[str, Any]
) -> str:
    """
    Use memory state to recommend escalation timing.
    Bounded recovery orchestration.
    """
    if not active_issue:
        return "no_escalation"
    
    unresolved_turns = active_issue.get("unresolved_turn_count", 0)
    escalation_history_count = len(ltm.get("previous_escalations", []))
    
    # Escalate faster if repeated issue + previous escalations
    if unresolved_turns > 4 and escalation_history_count > 1:
        return "escalate_immediately"
    
    # Normal escalation path
    if unresolved_turns > 5:
        return "escalate_soon"
    
    # Try recovery
    if unresolved_turns > 2:
        return "attempt_recovery"
    
    return "continue_normal"


# # ──────────────────────────────────────────────────────────────
# # DEMO TEST CASES
# # ──────────────────────────────────────────────────────────────

# def run_demo():
#     """Run memory system demo."""

#     print("\n" + "█"*70)
#     print("█ NOVADESK AI — MEMORY SYSTEM DEMO")
#     print("█"*70)

#     # Demo 1: STM with clarification tracking
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 1: Short-Term Memory & Clarification Tracking")
#     print("▌"*70)

#     conv_state = init_conversation_state("user_123")
#     stm = conv_state["stm"]

#     stm = update_stm(stm, {"role": "user", "content": "Why was I charged twice?"}, topic="billing")
#     stm = update_stm(stm, {"role": "assistant", "content": "Let me check your invoices."}, topic="billing")
#     stm = update_stm_clarification(stm)
#     stm = update_stm_clarification(stm)

#     print(f"STM State: {stm['clarification_attempts']} clarifications, Topic: {stm['active_topic']}")

#     # Demo 2: LTM tracking recurring issues
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 2: Long-Term Memory - Recurring Issues")
#     print("▌"*70)

#     ltm = conv_state["ltm"]
#     ltm = update_ltm_issue(ltm, "billing_dispute", "Duplicate charges on account")
#     ltm = update_ltm_issue(ltm, "billing_dispute", "Invoice discrepancy")
#     ltm = update_ltm_issue(ltm, "api_quota", "Rate limiting issues")

#     ltm_context = retrieve_ltm_context(ltm, "billing")
#     print(f"Recurring issues: {len(ltm_context['recurring_issues'])}")
#     for issue in ltm_context['recurring_issues']:
#         print(f"  - {issue['issue_type']}: {issue['mention_count']} mentions")

#     # Demo 3: Active issue tracking
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 3: Active Issue Tracking")
#     print("▌"*70)

#     active_issue = init_active_issue("billing_dispute")
#     conv_state["active_issue"] = active_issue

#     for i in range(5):
#         conv_state["active_issue"] = increment_unresolved_turns(conv_state["active_issue"])

#     print(f"Active issue: {active_issue['issue_id']}")
#     print(f"Unresolved turns: {conv_state['active_issue']['unresolved_turn_count']}")
#     print(f"Risk level: {conv_state['active_issue']['active_risk_level'].upper()}")

#     # Demo 4: Escalation history continuity
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 4: Escalation History & Continuity")
#     print("▌"*70)

#     escalation_data = {
#         "reason": "Legal threat detected",
#         "priority": "critical",
#         "escalation_stage": "pre_generation"
#     }

#     conv_state = update_escalation_history(conv_state, escalation_data)

#     print(f"Escalation history: {len(conv_state['escalation_history'])} events")
#     print(f"LTM escalations recorded: {len(conv_state['ltm']['previous_escalations'])}")

#     # Demo 5: Memory-informed recovery
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 5: Memory-Informed Recovery Orchestration")
#     print("▌"*70)

#     recommendation = get_escalation_recommendation_from_memory(
#         conv_state["active_issue"],
#         conv_state["ltm"]
#     )

#     print(f"Escalation recommendation: {recommendation.upper()}")

#     retrieval_refresh = should_prioritize_retrieval_refresh(conv_state["stm"], conv_state)
#     print(f"Retrieval refresh recommended: {retrieval_refresh}")

#     # Demo 6: Conversation summary for handoff
#     print("\n\n" + "▌"*70)
#     print("▌ DEMO 6: Escalation Handoff Summary")
#     print("▌"*70)

#     conversation = [
#         {"role": "user", "content": "I was charged twice on my account"},
#         {"role": "assistant", "content": "Let me investigate your billing"},
#         {"role": "user", "content": "This is frustrating, still not resolved"},
#     ]

#     summary = summarize_conversation(conversation)
#     print(summary)

#     print("\n\n" + "█"*70)
#     print("█ DEMO COMPLETE")
#     print("█"*70 + "\n")


# if __name__ == "__main__":
#     run_demo()
