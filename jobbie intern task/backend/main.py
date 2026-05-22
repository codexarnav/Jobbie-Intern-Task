import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Project modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    from retrieval import HybridRetriever, RetrievalResult
    from loop_detection import LoopDetector
    from validation import validate_response
    from escalation import orchestrate_escalation, evaluate_pre_generation_escalation
    from memory import (
        init_conversation_state, update_stm, retrieve_stm_context,
        update_ltm_issue, retrieve_ltm_context, init_active_issue,
        increment_unresolved_turns, update_retrieval_history,
        update_validation_history, update_escalation_history,
        summarize_conversation, get_escalation_recommendation_from_memory
    )
    MODULES_AVAILABLE = True
    print("✅ All orchestration modules loaded")
except Exception as e:
    print(f"❌ Module import failure: {e}")
    MODULES_AVAILABLE = False

# LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: langchain-google-genai not installed")
    GEMINI_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.0-flash"


# ──────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────

@dataclass
class OrchestrationResult:
    """Complete orchestration output."""
    status: str  # "success" or "escalation"
    response: str
    sources: List[str]
    confidence: float
    requires_escalation: bool
    escalation_ticket: Optional[Dict[str, Any]]
    orchestration_log: List[str]


# ──────────────────────────────────────────────────────────────
# INTENT + RISK CLASSIFICATION
# ──────────────────────────────────────────────────────────────

def classify_intent_and_risk(query: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight intent and risk classification."""
    query_lower = query.lower()

    # Intent detection
    intent = "general_inquiry"
    if any(w in query_lower for w in ["refund", "money back", "charge back"]):
        intent = "refund_request"
    elif any(w in query_lower for w in ["billing", "invoice", "charge", "payment", "bill"]):
        intent = "billing_issue"
    elif any(w in query_lower for w in ["pricing", "plan", "cost", "price"]):
        intent = "pricing_question"
    elif any(w in query_lower for w in ["api", "rate", "quota", "limit"]):
        intent = "api_usage"
    elif any(w in query_lower for w in ["lawyer", "legal", "lawsuit", "sue", "attorney"]):
        intent = "legal_escalation"

    # Risk detection
    risk_level = "low"
    escalation_probability = 0.0

    if "legal" in query_lower or "lawsuit" in query_lower:
        risk_level = "critical"
        escalation_probability = 0.95
    elif intent == "refund_request" and any(w in query_lower for w in ["frustrated", "angry", "upset"]):
        risk_level = "high"
        escalation_probability = 0.75
    elif conversation_state.get("unresolved_turn_count", 0) > 3:
        risk_level = "medium"
        escalation_probability = 0.5

    return {
        "intent": intent,
        "risk_level": risk_level,
        "escalation_probability": escalation_probability
    }


# ──────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ──────────────────────────────────────────────────────────────

def build_llm_context(
    retrieval_result: Dict[str, Any],
    stm_context: Dict[str, Any],
    ltm_context: Dict[str, Any],
    active_issue: Optional[Dict[str, Any]]
) -> str:
    """Build structured context for Gemini prompt."""

    context_parts = []

    # Retrieved documents
    if retrieval_result.get("final_context"):
        context_parts.append("=== RETRIEVED SUPPORT CONTEXT ===")
        context_parts.append(retrieval_result["final_context"][:2000])

    # STM context
    stm_info = f"\nRECENT CONVERSATION CONTEXT:\n"
    stm_info += f"- Active topic: {stm_context.get('active_topic', 'N/A')}\n"
    stm_info += f"- Recent turns: {stm_context.get('recent_turns', 0)}\n"
    stm_info += f"- Clarification attempts: {stm_context.get('clarification_attempts', 0)}"
    context_parts.append(stm_info)

    # LTM context
    if ltm_context.get("recurring_issues"):
        ltm_info = f"\nCUSTOMER HISTORY:\n"
        for issue in ltm_context["recurring_issues"][:3]:
            ltm_info += f"- {issue.get('issue_type', 'unknown')}: {issue.get('mention_count', 1)} mentions\n"
        context_parts.append(ltm_info)

    # Active issue
    if active_issue:
        issue_info = f"\nACTIVE ISSUE:\n"
        issue_info += f"- Type: {active_issue.get('issue_type', 'unknown')}\n"
        issue_info += f"- Unresolved for {active_issue.get('unresolved_turn_count', 0)} turns\n"
        issue_info += f"- Status: {active_issue.get('resolution_status', 'unknown')}"
        context_parts.append(issue_info)

    return "\n".join(context_parts)


# ──────────────────────────────────────────────────────────────
# GEMINI RESPONSE GENERATION
# ──────────────────────────────────────────────────────────────

def generate_llm_response(
    query: str,
    context: str,
    conversation_history: List[Dict[str, str]],
    stm_context: Dict[str, Any],
    ltm_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate response using Google Gemini with reliability constraints."""

    if not GEMINI_AVAILABLE:
        return {
            "response": "I don't have the AI service available. Please contact support.",
            "sources": [],
            "needs_clarification": True,
            "escalation_flag": True
        }

    try:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Build ONE structured orchestration prompt
        system_prompt = f"""=== SYSTEM ROLE ===
You are NovaDesk AI Support Agent — a bounded conversational reliability system.

CRITICAL BEHAVIOR RULES:
1. REMAIN GROUNDED: Only use information from the provided retrieval context below.
2. AVOID FABRICATION: Never invent policies, pricing, or refund terms.
3. AVOID UNSUPPORTED PROMISES: Never promise refunds, SLA changes, or guarantees not in documentation.
4. ESCALATE UNCERTAINTY: If you cannot ground a response, recommend escalation rather than guessing.
5. PREFER CLARIFICATION: When intent is ambiguous, ask a clarifying question over making assumptions.
6. PROFESSIONAL TONE: Maintain support-focused, empathetic communication at all times.

=== RETRIEVAL CONTEXT ===
{context}

=== STM CONTEXT (Short-Term Memory) ===
Active Topic: {stm_context.get('active_topic', 'N/A')}
Recent Turns: {stm_context.get('recent_turns', 0)}
Clarification Attempts: {stm_context.get('clarification_attempts', 0)}

=== LTM CONTEXT (Long-Term Memory) ===
Recurring Issues: {json.dumps(ltm_context.get('recurring_issues', [])[:3], indent=2)}

=== VALIDATION RULES ===
- Every factual claim must be traceable to the retrieved context above.
- If no supporting document exists for a claim, say: "I don't have clear information on that."
- If the user raises legal or regulatory concerns, immediately flag for escalation.
- Do not provide specific pricing, billing amounts, or refund figures unless sourced from context.

=== CURRENT USER QUERY ===
Customer: {query}"""

        messages = [SystemMessage(content=system_prompt)]

        # Add recent conversation history
        for msg in conversation_history[-4:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=query))
        # Orchestration visibility log
        context_length = len(system_prompt)
        stm_injected = stm_context.get('recent_turns', 0) > 0
        ltm_injected = bool(ltm_context.get('recurring_issues'))

        print(f"\n  {'─'*60}")
        print(f"  🤖 GEMINI ORCHESTRATION")
        print(f"  {'─'*60}")
        print(f"    Model              : {"gemini-2.0-flash"}")
        print(f"    Temperature        : 0.3")
        print(f"    Context Length     : {context_length} chars")
        print(f"    STM Injected       : {'✅ Yes' if stm_injected else '❌ No'}")
        print(f"    LTM Injected       : {'✅ Yes' if ltm_injected else '❌ No'}")
        print(f"    Retrieval Grounded : ✅ Active")
        print(f"  {'─'*60}")

        # Generate response
        result = llm.invoke(messages)
        response_text = result.content

        print(f"  ✅ Gemini response generated")
        print(f"  {'─'*60}")

        return {
            "response": response_text,
            "sources": ["gemini-grounded"],
            "needs_clarification": False,
            "escalation_flag": False
        }

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return {
            "response": "I encountered an issue generating a response. Let me connect you with support.",
            "sources": [],
            "needs_clarification": True,
            "escalation_flag": True
        }


# ──────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION ENGINE
# ──────────────────────────────────────────────────────────────

class NovaDesk:
    """Central orchestration engine."""

    def __init__(self, user_id: str = "user_001"):
        if not MODULES_AVAILABLE:
            raise RuntimeError(
                "❌ Cannot start NovaDesk — one or more orchestration modules "
                "failed to import. Check the error printed above."
            )
        self.user_id = user_id
        self.conversation_state = init_conversation_state(user_id)
        self.retriever = HybridRetriever()
        self.retriever.initialize()
        self.loop_detector = LoopDetector()
        self.loop_detector.initialize()
        self.log = []

    def _log(self, stage: str, message: str):
        """Add orchestration log entry."""
        log_entry = f"[{stage}] {message}"
        self.log.append(log_entry)
        print(f"  {log_entry}")

    def process(self, query: str) -> OrchestrationResult:
        """
        Main orchestration flow.
        Implements complete bounded probabilistic behavior system.
        """
        self.log = []
        print(f"\n{'─'*70}")
        print(f"🎯 ORCHESTRATION: {query[:60]}")
        print(f"{'─'*70}")

        # Extract conversation history
        conv_history = self.conversation_state.get("stm", {}).get("recent_messages", [])
        conv_history_dicts = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in conv_history
        ]

        active_issue = self.conversation_state.get("active_issue")

        # STAGE 1: Memory — Update STM with user message FIRST
        self._log("MEMORY", "Updating STM with incoming user message")
        self.conversation_state["stm"] = update_stm(
            self.conversation_state["stm"],
            {"role": "user", "content": query},
            "pending"  # intent not yet classified; will be updated after
        )

        # Retrieve updated context post-STM-write
        stm_context = retrieve_stm_context(self.conversation_state["stm"])
        ltm_context = retrieve_ltm_context(self.conversation_state["ltm"], query)
        self._log("MEMORY", "STM/LTM context retrieved")

        # STAGE 2: Pre-Loop Detection
        self._log("PRE-LOOP", "Checking for conversational loops")
        pre_loop_result = self.loop_detector.detect_pre_loop(query, conv_history_dicts)
        if pre_loop_result.get("loop_detected"):
            self._log("PRE-LOOP", f"Loop risk: {pre_loop_result.get('loop_risk_score', 0):.2f}")
        else:
            self._log("PRE-LOOP", "No loop detected")

        # STAGE 3: Intent + Risk Classification
        self._log("INTENT", "Classifying intent and risk")
        intent_data = classify_intent_and_risk(
            query,
            {"unresolved_turn_count": active_issue.get("unresolved_turn_count", 0) if active_issue else 0}
        )
        self._log("INTENT", f"Intent: {intent_data['intent']}, Risk: {intent_data['risk_level']}, Esc. Prob: {intent_data['escalation_probability']:.2f}")

        # STAGE 4: Pre-Generation Escalation Check (always via escalation engine)
        self._log("ESCALATION", "Pre-generation escalation evaluation")

        if not active_issue:
            active_issue = init_active_issue(intent_data["intent"])
            self.conversation_state["active_issue"] = active_issue

        pre_escalation_result = evaluate_pre_generation_escalation(
            query,
            intent_data,
            pre_loop_result,
            {"unresolved_turn_count": active_issue.get("unresolved_turn_count", 0)},
            conv_history_dicts
        )

        if pre_escalation_result.requires_escalation:
            self._log("ESCALATION", f"🚨 PRE-GENERATION ESCALATION: {pre_escalation_result.reason}")
            return self._escalate_handoff(
                pre_escalation_result,
                query,
                intent_data,
                conv_history_dicts
            )
        else:
            self._log("ESCALATION", "Pre-generation escalation not required")

        # STAGE 5: Hybrid Retrieval
        self._log("RETRIEVAL", "Running hybrid retrieval pipeline")
        retrieval_result = self.retriever.retrieve(query, conv_history_dicts)
        self._log("RETRIEVAL", f"Source: {retrieval_result.retrieval_source}, Confidence: {retrieval_result.retrieval_confidence:.2f}")

        retrieval_dict = {
            "retrieval_source": retrieval_result.retrieval_source,
            "retrieval_confidence": retrieval_result.retrieval_confidence,
            "final_context": retrieval_result.final_context,
            "retrieved_documents": retrieval_result.retrieved_documents
        }

        # STAGE 6: Context Builder
        self._log("CONTEXT", "Building structured LLM context")
        llm_context = build_llm_context(retrieval_dict, stm_context, ltm_context, active_issue)

        # STAGE 7: Gemini Response Generation
        self._log("LLM", "Invoking Gemini via orchestration prompt")
        llm_response = generate_llm_response(
            query,
            llm_context,
            conv_history_dicts,
            stm_context,
            ltm_context
        )

        # STAGE 8: Post-Loop Detection
        self._log("POST-LOOP", "Checking for response loops")
        assistant_history = [
            m.get("content") for m in self.conversation_state.get("stm", {}).get("recent_messages", [])
            if m.get("role") == "assistant"
        ]
        post_loop_result = self.loop_detector.detect_post_loop(llm_response["response"], assistant_history)
        if post_loop_result.get("post_loop_detected"):
            self._log("POST-LOOP", f"Post-loop risk: {post_loop_result.get('post_loop_score', 0):.2f}")
        else:
            self._log("POST-LOOP", "No post-loop detected")

        # STAGE 9: Validation (with enriched state context)
        self._log("VALIDATION", "Running state-aware validation pipeline")
        validation_result = validate_response(
        llm_response["response"],
        retrieval_dict,
        post_loop_result,
        assistant_history,
        system_state={
        "intent_data": intent_data,
        "active_issue": active_issue,
        "escalation_history": self.conversation_state.get("escalation_history", []),
        "retrieval_confidence": retrieval_result.retrieval_confidence,
        "loop_risk": pre_loop_result.get("loop_risk_score", 0.0)
        }
    )
        self._log("VALIDATION", f"Confidence: {validation_result.confidence_score:.2f}, Grounding: {validation_result.grounding['grounding_score']:.2f}")

        # Final confidence comes from validation — not hardcoded
        final_confidence = validation_result.confidence_score

        # STAGE 10: Post-Validation Escalation Check
        self._log("ESCALATION", "Post-validation escalation evaluation")
        escalation_result = orchestrate_escalation(
            query,
            retrieval_dict,
            asdict(validation_result),
            post_loop_result,
            {"unresolved_turn_count": active_issue.get("unresolved_turn_count", 0)} if active_issue else {},
            conv_history_dicts,
            pre_escalation_result
        )

        if escalation_result.escalation_required:
            self._log("ESCALATION", f"🚨 POST-VALIDATION ESCALATION: {escalation_result.reason}")
            return self._escalate_handoff(escalation_result, query, intent_data, conv_history_dicts)
        else:
            self._log("ESCALATION", "Post-validation escalation not required")

        # STAGE 11: Memory Finalization — update STM with assistant response
        self._log("MEMORY", "Finalising memory: storing assistant response in STM")
        self.conversation_state["stm"] = update_stm(
            self.conversation_state["stm"],
            {"role": "assistant", "content": llm_response["response"]},
            intent_data["intent"]
        )
        self.conversation_state = update_retrieval_history(self.conversation_state, retrieval_dict)
        self.conversation_state = update_validation_history(self.conversation_state, asdict(validation_result))

        self._log("SUCCESS", "✅ Orchestration complete — response ready")

        return OrchestrationResult(
            status="success",
            response=llm_response["response"],
            sources=retrieval_result.retrieved_documents[:3] if retrieval_result.retrieved_documents else [],
            confidence=final_confidence,
            requires_escalation=False,
            escalation_ticket=None,
            orchestration_log=self.log
        )

    def _escalate_handoff(
        self,
        escalation_data: Any,
        query: str,
        intent_data: Dict,
        conversation_history: List[Dict]
    ) -> OrchestrationResult:
        """Handle escalation handoff."""

        if not self.conversation_state.get("active_issue"):
            self.conversation_state["active_issue"] = init_active_issue(intent_data["intent"])

        self.conversation_state = update_escalation_history(
            self.conversation_state,
            {
                "reason": escalation_data.reason if hasattr(escalation_data, "reason") else escalation_data.get("reason"),
                "priority": escalation_data.priority if hasattr(escalation_data, "priority") else escalation_data.get("priority"),
                "escalation_stage": escalation_data.escalation_stage if hasattr(escalation_data, "escalation_stage") else escalation_data.get("escalation_stage")
            }
        )

        summary = summarize_conversation(conversation_history)

        handoff_ticket = {
            "ticket_id": f"TICK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "escalation_reason": escalation_data.reason if hasattr(escalation_data, "reason") else escalation_data.get("reason"),
            "priority": escalation_data.priority if hasattr(escalation_data, "priority") else escalation_data.get("priority"),
            "conversation_summary": summary,
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat()
        }

        return OrchestrationResult(
            status="escalation",
            response=f"Thank you for your patience. Your issue has been escalated to our support team.\n\nTicket ID: {handoff_ticket['ticket_id']}",
            sources=[],
            confidence=0.0,
            requires_escalation=True,
            escalation_ticket=handoff_ticket,
            orchestration_log=self.log
        )


# ──────────────────────────────────────────────────────────────
# RESPONSE DISPLAY
# ──────────────────────────────────────────────────────────────

def display_result(result: OrchestrationResult):
    """Print clean formatted orchestration output."""
    print(f"\n{'═'*54}")
    print(f"  FINAL RESPONSE")
    print(f"{'═'*54}")

    if result.requires_escalation:
        print(f"\n  🚨 HUMAN-IN-THE-LOOP ESCALATION\n")
    
    print(f"  Assistant:\n")
    for line in result.response.split("\n"):
        print(f"    {line}")

    print(f"\n  Confidence : {result.confidence:.2f}")

    if result.sources:
        print(f"  Sources    :")
        for src in result.sources:
            print(f"    • {src}")
    else:
        print(f"  Sources    : —")

    print(f"  Escalation : {'Yes 🚨' if result.requires_escalation else 'No'}")

    if result.requires_escalation and result.escalation_ticket:
        print(f"  Ticket ID  : {result.escalation_ticket['ticket_id']}")

    print(f"{'═'*54}\n")


# ──────────────────────────────────────────────────────────────
# LIVE INTERACTIVE RUNTIME
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "█"*54)
    print("█  NOVADESK AI — ORCHESTRATION RUNTIME")
    print("█"*54)

    # API key validation
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY missing — set it in your .env file")
        sys.exit(1)
    else:
        print("✅ Gemini API initialized")

    print("\n  Type your query to begin. Type 'exit' or 'quit' to stop.\n")

    system = NovaDesk(user_id="user_001")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Session terminated. Goodbye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("\n👋 Session terminated. Goodbye.")
            break

        result = system.process(query)
        display_result(result)