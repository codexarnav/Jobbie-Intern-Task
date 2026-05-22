import streamlit as st
import sys
import os
from dataclasses import asdict

# ── Path setup so backend modules resolve ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="NovaDesk AI Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",  # handles sidebar — no CSS needed
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS — Dark industrial AI ops aesthetic
# Fixes applied:
#   1. block-container padding restored (was 0, caused left-bleed)
#   2. Dead .nd-body/.nd-left/.nd-right rules removed
#   3. calc(100vh) on wrappers removed — handled by st.container(height=)
#   4. stSidebar CSS rule removed (redundant, can conflict)
#   5. stTextInput selector changed to class-based (stable across versions)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c10 !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,180,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,120,255,0.05) 0%, transparent 60%),
        #080c10 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }

/* FIX 1: Restore padding so columns don't bleed to viewport edge.
   Previously was padding:0 which caused the left-alignment bug. */
.block-container {
    padding: 0 1.5rem 1rem 1.5rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.nd-header {
    border-bottom: 1px solid rgba(0,255,180,0.15);
    padding: 20px 36px 16px;
    background: rgba(0,255,180,0.02);
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 1rem;
}
.nd-logo {
    width: 36px; height: 36px;
    border: 1.5px solid #00ffb4;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #00ffb4;
    font-family: 'Syne', sans-serif; font-weight: 800;
    flex-shrink: 0;
}
.nd-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px; font-weight: 800;
    color: #e8f4f0; letter-spacing: 0.05em;
}
.nd-subtitle {
    font-size: 10px; color: #4a7c6a;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.nd-status {
    margin-left: auto;
    display: flex; align-items: center; gap: 8px;
    font-size: 10px; color: #4a7c6a; letter-spacing: 0.1em;
}
.nd-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #00ffb4;
    box-shadow: 0 0 8px #00ffb4;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Message bubbles ── */
.msg-user {
    align-self: flex-end; max-width: 72%;
    background: rgba(0,255,180,0.07);
    border: 1px solid rgba(0,255,180,0.2);
    border-radius: 2px 12px 12px 12px;
    padding: 12px 16px;
    font-size: 13px; color: #d0ede5;
    margin-bottom: 12px;
}
.msg-user .msg-role { color: #00ffb4; font-size: 9px; letter-spacing: 0.15em; margin-bottom: 6px; }

.msg-assistant {
    align-self: flex-start; max-width: 80%;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    font-size: 13px; color: #c0cdd9; line-height: 1.7;
    margin-bottom: 12px;
}
.msg-assistant .msg-role { color: #4a90d9; font-size: 9px; letter-spacing: 0.15em; margin-bottom: 6px; }

.msg-escalation {
    align-self: flex-start; max-width: 80%;
    background: rgba(255,60,60,0.07);
    border: 1px solid rgba(255,60,60,0.3);
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    font-size: 13px; color: #ffaaaa; line-height: 1.6;
    margin-bottom: 12px;
}
.msg-escalation .msg-role { color: #ff4444; font-size: 9px; letter-spacing: 0.15em; margin-bottom: 6px; }

/* ── Orchestration panel ── */
.orch-panel { padding: 0 8px; }
.orch-title {
    font-family: 'Syne', sans-serif; font-size: 11px;
    font-weight: 700; color: #4a7c6a;
    letter-spacing: 0.18em; text-transform: uppercase;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 14px;
}

/* ── Stage cards ── */
.stage-card {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 10px;
    overflow: hidden;
}
.stage-header {
    padding: 10px 14px;
    font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stage-body { padding: 12px 14px; font-size: 11px; line-height: 1.9; }

.stage-ok .stage-header   { color: #00ffb4; background: rgba(0,255,180,0.04); }
.stage-warn .stage-header { color: #ffc84a; background: rgba(255,200,74,0.05); }
.stage-crit .stage-header { color: #ff4444; background: rgba(255,68,68,0.06); }
.stage-info .stage-header { color: #4a90d9; background: rgba(74,144,217,0.05); }
.stage-idle .stage-header { color: #445566; background: rgba(255,255,255,0.01); }

/* ── Metric rows ── */
.metric-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 2px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.metric-row:last-child { border-bottom: none; }
.metric-key { color: #445a6a; font-size: 10px; letter-spacing: 0.08em; }
.metric-val { color: #c8d6e5; font-size: 11px; font-weight: 600; }
.metric-val.green  { color: #00ffb4; }
.metric-val.yellow { color: #ffc84a; }
.metric-val.red    { color: #ff4444; }
.metric-val.blue   { color: #4a90d9; }

/* ── Progress bars ── */
.bar-wrap { margin: 8px 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 4px; color: #445a6a; }
.bar-track { height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.bar-green  { background: linear-gradient(90deg, #00ffb4, #00cc8f); }
.bar-yellow { background: linear-gradient(90deg, #ffc84a, #ff9900); }
.bar-red    { background: linear-gradient(90deg, #ff4444, #cc0000); }

/* ── Escalation alert ── */
.esc-alert {
    background: rgba(255,40,40,0.08);
    border: 1px solid rgba(255,40,40,0.4);
    border-radius: 6px; padding: 14px;
    margin-bottom: 10px;
}
.esc-alert-title {
    color: #ff4444; font-size: 11px;
    font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 10px;
}

/* ── Gemini warning ── */
.gemini-warn {
    background: rgba(255,200,74,0.07);
    border: 1px solid rgba(255,200,74,0.3);
    border-radius: 4px; padding: 10px 14px;
    font-size: 11px; color: #ffc84a;
    margin-bottom: 10px; letter-spacing: 0.05em;
}

/* ── Idle state ── */
.idle-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; gap: 12px;
    color: #2a3a4a; text-align: center;
    padding: 40px 0;
}
.idle-icon { font-size: 32px; opacity: 0.3; }
.idle-text { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; line-height: 2; }

/* FIX 2: Use stable class-based selector for text input.
   data-testid attributes shift between Streamlit versions. */
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,255,180,0.2) !important;
    border-radius: 4px !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput input:focus {
    border-color: rgba(0,255,180,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,255,180,0.08) !important;
    outline: none !important;
}
.stTextInput label { display: none !important; }

.stButton > button {
    background: rgba(0,255,180,0.1) !important;
    border: 1px solid rgba(0,255,180,0.35) !important;
    color: #00ffb4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    border-radius: 4px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(0,255,180,0.18) !important;
    border-color: #00ffb4 !important;
}

/* Clear button variant */
.clear-btn .stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #445a6a !important;
}
.clear-btn .stButton > button:hover {
    border-color: rgba(255,80,80,0.4) !important;
    color: #ff6666 !important;
}

/* ── st.container scroll area styling ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(0,255,180,0.08) !important;
    border-radius: 6px !important;
    background: rgba(0,0,0,0.15) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 6px !important;
    background: rgba(255,255,255,0.01) !important;
}
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #4a7c6a !important;
    letter-spacing: 0.1em !important;
}

/* ── Confidence bar below input ── */
.conf-bar-wrap { padding: 6px 0 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,255,180,0.15); border-radius: 2px; }

/* ── Column gap fix — prevent right panel overlap ── */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; align-items: flex-start !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════

def init_session():
    if "novadesk" not in st.session_state:
        st.session_state.novadesk = None
        st.session_state.init_error = None
        try:
            from main import NovaDesk
            st.session_state.novadesk = NovaDesk(user_id="ui_user_001")
        except Exception as e:
            st.session_state.init_error = str(e)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # {role, content, escalation}

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_log_raw" not in st.session_state:
        st.session_state.last_log_raw = {}

    if "gemini_failed" not in st.session_state:
        st.session_state.gemini_failed = False

init_session()


# ══════════════════════════════════════════════════════════════
# HELPER RENDERERS
# ══════════════════════════════════════════════════════════════

def color_for_score(score: float) -> str:
    if score >= 0.7: return "green"
    if score >= 0.4: return "yellow"
    return "red"

def bar_class(score: float) -> str:
    if score >= 0.7: return "bar-green"
    if score >= 0.4: return "bar-yellow"
    return "bar-red"

def risk_color(score: float) -> str:
    if score <= 0.3: return "green"
    if score <= 0.6: return "yellow"
    return "red"

def bar_class_risk(score: float) -> str:
    if score <= 0.3: return "bar-green"
    if score <= 0.6: return "bar-yellow"
    return "bar-red"

def stage_class(score: float, inverted=False) -> str:
    if inverted:
        if score <= 0.3: return "stage-ok"
        if score <= 0.6: return "stage-warn"
        return "stage-crit"
    if score >= 0.7: return "stage-ok"
    if score >= 0.4: return "stage-warn"
    return "stage-crit"

def metric_html(key: str, val, color: str = "") -> str:
    cls = f"metric-val {color}" if color else "metric-val"
    return f"""<div class="metric-row">
        <span class="metric-key">{key}</span>
        <span class="{cls}">{val}</span>
    </div>"""

def bar_html(label: str, score: float, inverted=False) -> str:
    pct = int(score * 100)
    bc  = bar_class_risk(score) if inverted else bar_class(score)
    lc  = risk_color(score)     if inverted else color_for_score(score)
    return f"""<div class="bar-wrap">
        <div class="bar-label"><span>{label}</span><span class="metric-val {lc}">{pct}%</span></div>
        <div class="bar-track"><div class="bar-fill {bc}" style="width:{pct}%"></div></div>
    </div>"""


# ══════════════════════════════════════════════════════════════
# ORCHESTRATION PANEL RENDERERS
# ══════════════════════════════════════════════════════════════

def render_idle_panel():
    st.markdown("""
    <div class="idle-state">
        <div class="idle-icon">◈</div>
        <div class="idle-text">ORCHESTRATION PIPELINE<br>awaiting first query</div>
    </div>
    """, unsafe_allow_html=True)


def render_loop_card(result):
    log = st.session_state.get("last_log_raw", {})
    loop_risk = log.get("pre_loop_risk", 0.0)
    sc = stage_class(loop_risk, inverted=True)
    st.markdown(f"""
    <div class="stage-card {sc}">
        <div class="stage-header">◉ Pre-Loop Detection</div>
        <div class="stage-body">
            {bar_html("Loop Risk Score", loop_risk, inverted=True)}
            {metric_html("Status", "LOOP DETECTED" if loop_risk > 0.6 else "Clear", risk_color(loop_risk))}
            {metric_html("Recovery Action", log.get("pre_loop_recovery", "continue_normal_flow"))}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_intent_card(result):
    log = st.session_state.get("last_log_raw", {})
    intent    = log.get("intent", "—")
    risk      = log.get("risk_level", "low")
    esc_prob  = log.get("escalation_probability", 0.0)
    risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "red"}
    rc = risk_colors.get(risk, "")
    st.markdown(f"""
    <div class="stage-card stage-info">
        <div class="stage-header">◎ Intent & Risk Classification</div>
        <div class="stage-body">
            {metric_html("Detected Intent", intent.replace("_", " ").title(), "blue")}
            {metric_html("Risk Level", risk.upper(), rc)}
            {bar_html("Escalation Probability", esc_prob, inverted=True)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_retrieval_card(result):
    log      = st.session_state.get("last_log_raw", {})
    ret_conf = log.get("retrieval_confidence", 0.0)
    ret_src  = log.get("retrieval_source", "—")
    doc_cnt  = log.get("doc_count", 0)
    sc = stage_class(ret_conf)
    st.markdown(f"""
    <div class="stage-card {sc}">
        <div class="stage-header">⬡ Retrieval Layer</div>
        <div class="stage-body">
            {bar_html("Retrieval Confidence", ret_conf)}
            {metric_html("Source", ret_src.upper(), color_for_score(ret_conf))}
            {metric_html("Documents Retrieved", doc_cnt)}
            {metric_html("FAQ Exact Match", "Yes" if ret_src == "faq_exact" else "No",
                         "green" if ret_src == "faq_exact" else "")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_memory_card(result):
    log  = st.session_state.get("last_log_raw", {})
    stm  = log.get("stm", {})
    ltm  = log.get("ltm", {})
    st.markdown(f"""
    <div class="stage-card stage-info">
        <div class="stage-header">⊞ Memory State</div>
        <div class="stage-body">
            {metric_html("STM Active Topic", stm.get("active_topic", "N/A"))}
            {metric_html("Recent Turns", stm.get("recent_turns", 0))}
            {metric_html("Clarification Attempts", stm.get("clarification_attempts", 0),
                         "yellow" if stm.get("clarification_attempts", 0) > 1 else "")}
            {metric_html("Unresolved Turns", log.get("unresolved_turns", 0),
                         "yellow" if log.get("unresolved_turns", 0) > 2 else "")}
            {metric_html("LTM Recurring Issues", len(ltm.get("recurring_issues", [])))}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_generation_card(result):
    log       = st.session_state.get("last_log_raw", {})
    gemini_ok = not st.session_state.gemini_failed
    ctx_len   = log.get("context_length", 0)
    stm_inj   = log.get("stm_injected", False)
    ltm_inj   = log.get("ltm_injected", False)
    sc = "stage-ok" if gemini_ok else "stage-warn"

    if not gemini_ok:
        st.markdown("""
        <div class="gemini-warn">
            ⚠ Gemini unavailable — deterministic fallback response active
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stage-card {sc}">
        <div class="stage-header">◈ Response Generation</div>
        <div class="stage-body">
            {metric_html("Model", "gemini-2.0-flash")}
            {metric_html("Temperature", "0.3")}
            {metric_html("Context Length", f"{ctx_len:,} chars")}
            {metric_html("Retrieval Grounded", "Active", "green")}
            {metric_html("STM Injected", "Yes" if stm_inj else "No", "green" if stm_inj else "")}
            {metric_html("LTM Injected", "Yes" if ltm_inj else "No", "green" if ltm_inj else "")}
            {metric_html("Status", "Online" if gemini_ok else "Fallback Mode",
                         "green" if gemini_ok else "yellow")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_validation_card(result):
    if not result:
        return
    log        = st.session_state.get("last_log_raw", {})
    conf       = result.confidence
    grounding  = log.get("grounding_score", 0.0)
    halluc     = log.get("hallucination_risk", 0.0)
    post_loop  = log.get("post_loop_risk", 0.0)
    esc        = result.requires_escalation
    sc = "stage-crit" if esc else stage_class(conf)
    st.markdown(f"""
    <div class="stage-card {sc}">
        <div class="stage-header">◇ Validation Layer</div>
        <div class="stage-body">
            {bar_html("Confidence Score", conf)}
            {bar_html("Grounding Score", grounding)}
            {bar_html("Hallucination Risk", halluc, inverted=True)}
            {bar_html("Post-Loop Risk", post_loop, inverted=True)}
            {metric_html("Escalation Triggered", "YES" if esc else "No", "red" if esc else "green")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_escalation_card(result):
    if not result or not result.requires_escalation:
        return
    ticket   = result.escalation_ticket or {}
    reason   = ticket.get("escalation_reason", "Unknown")
    priority = ticket.get("priority", "—").upper()
    tid      = ticket.get("ticket_id", "—")
    pc = "red" if priority in ("CRITICAL", "HIGH") else "yellow"
    st.markdown(f"""
    <div class="esc-alert">
        <div class="esc-alert-title">⚠ Human-in-the-Loop Escalation</div>
        {metric_html("Ticket ID", tid, "red")}
        {metric_html("Priority", priority, pc)}
        {metric_html("Reason", reason[:50] + ("…" if len(reason) > 50 else ""))}
    </div>
    """, unsafe_allow_html=True)


def render_orchestration_panel(result):
    st.markdown('<div class="orch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="orch-title">Orchestration Pipeline</div>', unsafe_allow_html=True)

    if result is None:
        render_idle_panel()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    render_escalation_card(result)
    render_loop_card(result)
    render_intent_card(result)
    render_retrieval_card(result)
    render_memory_card(result)
    render_generation_card(result)
    render_validation_card(result)

    with st.expander("ORCHESTRATION LOG", expanded=False):
        for entry in result.orchestration_log:
            color = (
                "#ff4444" if ("ESCALATION" in entry or "🚨" in entry) else
                "#ffc84a" if ("WARN" in entry or "LOOP" in entry) else
                "#00ffb4" if ("SUCCESS" in entry or "✅" in entry) else
                "#445a6a"
            )
            st.markdown(
                f'<div style="font-size:10px;color:{color};font-family:JetBrains Mono,monospace;'
                f'padding:2px 0;letter-spacing:0.05em">{entry}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PROCESS QUERY
# ══════════════════════════════════════════════════════════════

def process_query(query: str):
    system = st.session_state.novadesk
    if system is None:
        return None

    result = system.process(query)

    stm          = system.conversation_state.get("stm", {})
    ltm          = system.conversation_state.get("ltm", {})
    active_issue = system.conversation_state.get("active_issue", {})
    ret_hist     = system.conversation_state.get("retrieval_history", [{}])
    val_hist     = system.conversation_state.get("validation_history", [{}])
    last_ret     = ret_hist[-1] if ret_hist else {}
    last_val     = val_hist[-1] if val_hist else {}

    log = result.orchestration_log
    pre_loop_risk     = 0.0
    pre_loop_recovery = "continue_normal_flow"
    intent            = "general_inquiry"
    risk_level        = "low"
    esc_prob          = 0.0
    post_loop_risk    = 0.0

    for entry in log:
        if "Loop risk:" in entry:
            try: pre_loop_risk = float(entry.split("Loop risk:")[-1].strip())
            except: pass
        if "Intent:" in entry:
            parts = entry.split(",")
            try:
                intent     = parts[0].split("Intent:")[-1].strip()
                risk_level = parts[1].split("Risk:")[-1].strip()
                esc_prob   = float(parts[2].split("Esc. Prob:")[-1].strip())
            except: pass
        if "Post-loop risk:" in entry:
            try: post_loop_risk = float(entry.split("Post-loop risk:")[-1].strip())
            except: pass

    stm_injected = stm.get("turn_count", 0) > 1
    ltm_injected = bool(ltm.get("recurring_issues"))

    st.session_state.gemini_failed = "encountered an issue" in result.response.lower()

    st.session_state.last_log_raw = {
        "pre_loop_risk":          pre_loop_risk,
        "pre_loop_recovery":      pre_loop_recovery,
        "intent":                 intent,
        "risk_level":             risk_level,
        "escalation_probability": esc_prob,
        "retrieval_confidence":   last_ret.get("retrieval_confidence", result.confidence),
        "retrieval_source":       last_ret.get("retrieval_source", "multi"),
        "doc_count":              len(result.sources),
        "stm": {
            "active_topic":           stm.get("active_topic", intent),
            "recent_turns":           stm.get("turn_count", 0),
            "clarification_attempts": stm.get("clarification_count", 0),
        },
        "ltm":             ltm,
        "unresolved_turns": active_issue.get("unresolved_turn_count", 0) if active_issue else 0,
        "grounding_score":  last_val.get("grounding_score", result.confidence),
        "hallucination_risk": last_val.get("hallucination_risk", 0.0),
        "post_loop_risk":   post_loop_risk,
        "context_length":   0,
        "stm_injected":     stm_injected,
        "ltm_injected":     ltm_injected,
    }

    return result


# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="nd-header">
    <div class="nd-logo">N</div>
    <div>
        <div class="nd-title">NovaDesk AI Reliability Console</div>
        <div class="nd-subtitle">Bounded Conversational AI Orchestration System</div>
    </div>
    <div class="nd-status">
        <div class="nd-dot"></div>
        SYSTEM ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Init error guard ──────────────────────────────────────────
if st.session_state.init_error:
    st.markdown(f"""
    <div style="margin:24px;padding:16px;background:rgba(255,40,40,0.08);
         border:1px solid rgba(255,40,40,0.4);border-radius:6px;
         font-size:12px;color:#ff6666;font-family:JetBrains Mono,monospace">
        <strong>System Initialization Error</strong><br><br>
        {st.session_state.init_error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Two-column body ───────────────────────────────────────────
# FIX 3: gap="large" gives breathing room so right panel doesn't
#         crowd the chat column.
left_col, right_col = st.columns([2, 1], gap="large")


# ══════════════════════════════════════════════════════════════
# LEFT — CHAT INTERFACE
# ══════════════════════════════════════════════════════════════
with left_col:

    # FIX 4: st.container(height=520) creates a bounded scrollable
    #         area. Messages grow inside it; input stays pinned below.
    #         Previously there was no height bound, so messages pushed
    #         the input down and off-screen.
    with st.container(height=520):
        if not st.session_state.messages:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:80px 20px;gap:10px;text-align:center">
                <div style="font-size:28px;opacity:0.15">◈</div>
                <div style="font-size:11px;color:#2a3a4a;letter-spacing:0.12em;
                            text-transform:uppercase;line-height:2.2">
                    NovaDesk Orchestration Engine Ready<br>
                    Submit a query to begin pipeline execution
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                role   = msg["role"]
                content = msg["content"]
                is_esc = msg.get("escalation", False)

                if role == "user":
                    st.markdown(f"""
                    <div class="msg-user">
                        <div class="msg-role">USER QUERY</div>
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                elif is_esc:
                    st.markdown(f"""
                    <div class="msg-escalation">
                        <div class="msg-role">ESCALATION — HUMAN HANDOFF</div>
                        {content.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="msg-assistant">
                        <div class="msg-role">NOVADESK AGENT</div>
                        {content.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

    # ── Confidence bar (shown after first result) ─────────────
    if st.session_state.last_result:
        conf = st.session_state.last_result.confidence
        st.markdown(
            f'<div class="conf-bar-wrap">{bar_html("Response Confidence", conf)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Input area — always visible, pinned below chat container ─
    inp_col, btn_col, clr_col = st.columns([6, 1, 1], gap="small")

    with inp_col:
        query = st.text_input(
            "query",
            key="query_input",
            placeholder="Enter support query...",
            label_visibility="collapsed",
        )

    with btn_col:
        send = st.button("SEND", use_container_width=True)

    with clr_col:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        clear = st.button("CLEAR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RIGHT — ORCHESTRATION PANEL
# FIX 5: Wrap in st.container(height=620) so the right panel has
#         its own scroll and doesn't stretch the page vertically,
#         which was misaligning it against the left column.
# ══════════════════════════════════════════════════════════════
with right_col:
    with st.container(height=620):
        render_orchestration_panel(st.session_state.last_result)


# ══════════════════════════════════════════════════════════════
# EVENT HANDLERS
# ══════════════════════════════════════════════════════════════

if clear:
    st.session_state.messages     = []
    st.session_state.last_result  = None
    st.session_state.last_log_raw = {}
    st.session_state.gemini_failed = False
    try:
        from main import NovaDesk
        st.session_state.novadesk   = NovaDesk(user_id="ui_user_001")
        st.session_state.init_error = None
    except Exception as e:
        st.session_state.init_error = str(e)
    st.rerun()

if send and query and query.strip():
    st.session_state.messages.append({"role": "user", "content": query.strip()})

    with st.spinner(""):
        result = process_query(query.strip())

    if result:
        st.session_state.last_result = result
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    result.response,
            "escalation": result.requires_escalation,
        })
    else:
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    "System unavailable. Please check backend initialization.",
            "escalation": False,
        })

    st.rerun()