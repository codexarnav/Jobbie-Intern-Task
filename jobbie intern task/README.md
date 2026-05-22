# NovaDesk AI — Reliability-Centered Conversational Orchestration System

NovaDesk AI is a production-oriented conversational AI orchestration system designed to improve reliability, contextual continuity, hallucination resistance, escalation handling, and conversational stability in enterprise support environments.

Unlike traditional chatbot systems, NovaDesk focuses on bounded AI orchestration through:

* hybrid retrieval,
* conversational memory management,
* loop detection,
* validation pipelines,
* and Human-in-the-Loop (HIL) escalation.

The system was designed as a modular orchestration architecture where every component is state-aware and works together as part of a unified conversational pipeline.

---

# System Overview

The orchestration pipeline follows this flow:

```text
User Query
   ↓
Pre-Loop Detection
   ↓
Intent + Risk Classification
   ↓
Escalation Check
   ↓
Hybrid Retrieval Pipeline
   ↓
Context Builder
   ↓
LLM Response Generation (Gemini)
   ↓
Validation Layer
   ↓
Post-Validation Escalation
   ↓
Memory Update
   ↓
Final Response
```

The architecture was intentionally designed to:

* reduce inconsistent responses,
* prevent conversational looping,
* improve retrieval grounding,
* and enable safe escalation workflows.

---

# Core Features

## Hybrid Retrieval Layer

NovaDesk combines deterministic and probabilistic retrieval techniques:

* FAQ exact matching
* BM25 keyword retrieval
* semantic vector retrieval
* personalized memory retrieval

This improves grounding quality and retrieval reliability.

---

## Conversational Memory System

The system uses:

* STM (Short-Term Memory)
* LTM (Long-Term Memory)
* Active Issue Tracking

STM maintains recent conversational continuity.

LTM stores summarized historical memory and recurring patterns.

Active Issues maintain unresolved conversational workflows and escalation continuity.

---

## Loop Detection System

The orchestration system includes:

* pre-loop detection
* post-loop detection

The loop detection engine identifies:

* semantic repetition,
* unresolved conversational patterns,
* clarification failures,
* and response stagnation.

This helps reduce repetitive or stuck conversations.

---

## Validation Pipeline

The validation layer evaluates:

* hallucination risk,
* grounding confidence,
* response confidence,
* escalation triggers,
* and conversational safety.

This prevents unsupported responses and improves production reliability.

---

## Human-in-the-Loop Escalation

The escalation system proactively routes risky or unresolved interactions to support escalation workflows.

Escalation triggers include:

* high risk conversations,
* failed grounding validation,
* legal or compliance threats,
* repeated unresolved loops,
* and unsafe conversational states.

---

# Data Structure

The project uses structured demo enterprise support data for orchestration testing.

## Included Data

* FAQs
* refund policies
* pricing plans
* escalation rules
* edge case scenarios
* realistic conversation flows

These datasets are used to simulate:

* retrieval grounding,
* escalation workflows,
* validation behavior,
* and conversational continuity.

---

# Database Design

The schema is divided into three major memory components:

## 1. STM (Short-Term Memory)

Maintains:

* recent conversation summaries,
* active conversational context,
* loop risk scores,
* and temporary orchestration state.

---

## 2. LTM (Long-Term Memory)

Stores:

* summarized historical memory,
* recurring user patterns,
* escalation history,
* and semantic memory embeddings.

---

## 3. Active Issue Tracking

Tracks:

* unresolved issues,
* escalation stages,
* conversational continuity,
* and support workflow progression.

Additional tables include:

* users
* conversations
* messages
* retrieval_logs
* validation_logs
* escalations

---

# File Structure

```text
project-root/
│
├── backend/
│   ├── main.py
│   ├── retrieval.py
│   ├── memory.py
│   ├── loop_detection.py
│   ├── validation.py
│   ├── escalation.py
│
├── data/
│   ├── faqs.json
│   ├── pricing_plans.json
│   ├── refund_policies.json
│   ├── escalation_rules.json
│   ├── edge_cases.json
│   └── conversation_flows.json
│
├── vector_store/
│
├── docs/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# Logic Overview

## retrieval.py

Implements the hybrid retrieval system:

* FAQ exact matching
* BM25 retrieval
* semantic vector retrieval
* retrieval confidence scoring
* context merging and deduplication

This module powers the deterministic → probabilistic retrieval pipeline.

---

## memory.py

Handles:

* STM updates,
* LTM summarization,
* active issue tracking,
* memory retrieval,
* and conversational continuity.

This module maintains orchestration-aware conversational state.

---

## loop_detection.py

Implements:

* pre-loop detection,
* post-loop detection,
* semantic repetition scoring,
* clarification failure detection,
* and recovery recommendations.

This prevents repetitive or stuck conversational states.

---

## validation.py

Responsible for:

* hallucination detection,
* grounding validation,
* confidence scoring,
* and escalation-safe validation logic.

This layer ensures reliability before final responses are returned.

---

## escalation.py

Handles:

* escalation triggers,
* escalation routing,
* support handoff logic,
* and Human-in-the-Loop orchestration.

---

## main.py

Acts as the central orchestration runtime.

This file integrates:

* memory,
* retrieval,
* loop detection,
* validation,
* escalation,
* and Gemini-based response generation

into one unified conversational workflow.

---

# Gemini Integration

NovaDesk uses:

* Google Gemini API
* LangChain
* LangGraph

for LLM orchestration and response generation.

A Gemini API key is required.

---

# Setup Instructions

## 1. Clone Project

```bash
git clone <repository-url>
cd project-folder
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

# Running Backend Orchestration

Navigate to backend:

```bash
cd backend
```

Run orchestration system:

```bash
python main.py
```

This launches the conversational orchestration runtime locally.

---

# Running Streamlit Dashboard

Return to project root:

```bash
cd ..
```

Run dashboard:

```bash
py -m streamlit run app.py
```

OR:

```bash
python -m streamlit run app.py
```

The dashboard will launch locally at:

```text
http://localhost:8501
```

---

# Technology Stack

* Python
* Streamlit
* LangChain
* LangGraph
* Google Gemini API
* Sentence Transformers
* BM25 Retrieval
* Semantic Vector Retrieval
* PostgreSQL Schema Design
* DBDiagram
* Eraser.io

---

# Project Goal

The primary objective of NovaDesk AI was to demonstrate how enterprise conversational systems can move beyond simple chatbot architectures and evolve into:

* reliability-aware,
* retrieval-grounded,
* escalation-safe,
* memory-centric orchestration systems.

The project emphasizes orchestration visibility, bounded reasoning, and production-oriented conversational reliability.
