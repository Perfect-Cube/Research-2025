## Multi 3

```
                   Time ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝

Main Thread:
  ┌────────────────────────────┐
  │  🧠 Coordinator creates     │
  │  subtasks (1 list)         │
  └────────────────────────────┘
             │
             ▼
  ┌────────────────────────────┐
  │  🔁 Create ThreadPool      │
  │  for N worker agents       │
  └────────────────────────────┘
             │
             ▼

Threads:
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ...
 │ 🤖 Agent 1   │   │ 🤖 Agent 2   │   │ 🤖 Agent 3   │
 │ 🔎 Search    │   │ 🔎 Search    │   │ 🔎 Search    │  ← All start at same time
 │ ✉️ LLM Call  │   │ ✉️ LLM Call  │   │ ✉️ LLM Call  │
 └─────────────┘   └─────────────┘   └─────────────┘
        │                │                 │
        ▼                ▼                 ▼
  ✅ Finished      ✅ Finished       ✅ Finished

Main Thread:
  ┌────────────────────────────┐
  │  🧠 Synthesizer combines    │
  │  all results into 1 report │
  └────────────────────────────┘

```

## Async

```
                   Time ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝ ➝

Main (Event Loop):
  ┌────────────────────────────┐
  │  🧠 Orchestrator receives   │
  │  query and decomposes it   │
  └────────────────────────────┘
             │
             ▼
  ┌────────────────────────────┐
  │ 🔁 Sends async messages     │
  │ to N SearchWorker agents   │
  └────────────────────────────┘
             │
             ▼

Async Agents (handled by event loop concurrently):
 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ...
 │ 🤖 Agent 1   │   │ 🤖 Agent 2   │   │ 🤖 Agent 3   │
 │ 🔎 Search    │   │ 🔎 Search    │   │ 🔎 Search    │
 │ ✉️ LLM Call  │   │ ✉️ LLM Call  │   │ ✉️ LLM Call  │
 └─────────────┘   └─────────────┘   └─────────────┘
        │                │                 │
        ▼                ▼                 ▼
  ✅ Finished      ✅ Finished       ✅ Finished

Main:
  ┌────────────────────────────┐
  │  🧠 Orchestrator synthesizes│
  │  all results to final reply│
  └────────────────────────────┘

```
