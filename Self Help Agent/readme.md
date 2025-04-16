```
+---------------------+
|     User Goal       |
+---------------------+
           │
           ▼
+---------------------+
|     Planner LLM     |
+---------------------+
           │
           ▼
+---------------------+
|  Action Decision    |
+---------------------+
           │
    ┌──────┼─────────┬────────────┐
    │      │         │            │
    ▼      ▼         ▼            ▼
+-----------+  +---------------+  +------------+
| Need Tool |  | Sufficient    |  | Cannot     |
|           |  | Info          |  | Proceed    |
+-----------+  +---------------+  +------------+
    │                 │                │
    ▼                 ▼                ▼
+---------------------+        +-----------------+
|   Tool Registry     |        |   Fail State    |
+---------------------+        +-----------------+
    │                             │
    ▼                             ▼
+---------------------------+  +----------------+
| Format Tool Input         |  | Format Final   |
+---------------------------+  | Answer         |
    │                        +----------------+
    ▼                             │
+---------------------------+       ▼
| Execute Existing Tool     |  +----------------+
+---------------------------+  | Output to User |
    │                             +----------------+
    ▼
+---------------------------+
|   Observation (Result,    |
|      or Error)            |
+---------------------------+
    │
    ▼
   (Feedback to Planner LLM)


--- For New Tool Creation Branch ---

From [Action Decision]: If "Need New Tool" is decided:

    ▼
+---------------------------+
| Specify New Tool (LLM)    |
+---------------------------+
    │
    ▼
+---------------------------+
| Generate Tool Code (LLM)  |
+---------------------------+
    │
    ▼
+---------------------------+
| Validate & Sandbox Test   |
+---------------------------+
    │
┌───┴─────────┐
│             │
▼             ▼
[Validation   [Validation
Failed]       Passed]
   │             │
   ▼             ▼
+---------------------------+   (Feedback to Planner LLM)
| Observation (Error/Result)|
+---------------------------+
                    │
                    ▼
+---------------------------+
| Update Tool Registry      |
+---------------------------+
                    │
                    ▼
          (Now goes to "Tool Registry" for use)


--- Optional Path: Use Newly Created Tool ---

From [Action Decision]: If "Use Newly Created Tool" is chosen:

    ▼
+--------------------------------------+
| Format Tool Input (for new tool)     |
+--------------------------------------+
    │
    ▼
+--------------------------------------+
| Execute New Tool (via Sandbox)       |
+--------------------------------------+
    │
    ▼
+---------------------------+
|   Observation             |
+---------------------------+
    │
    ▼
   (Feedback to Planner LLM)
 
```
