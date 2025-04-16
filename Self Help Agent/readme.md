```
# Simple LLM Tool Creation & Execution Pipeline

+---------------------+
|     User Goal       |
+---------------------+
          │
          ▼
+---------------------+
|    Planner LLM      |
+---------------------+
          │
          ▼
+---------------------+
|  Action Decision    |
+---------------------+
          │
          ▼
┌────────────────────┬──────────────────────────────┬─────────────────┐
│                    │                              │                 │
▼                    ▼                              ▼                 ▼
Need Existing     Sufficient Info                Need New Tool    Cannot Proceed
    Tool              → → → Final Answer            (Create New)          │
                          (Output to User)                │               │
                                                         (Specify New Tool)
                                                                │
                                                                ▼
                                                      +---------------------+
                                                      | Generate Tool Code  |
                                                      +---------------------+
                                                                │
                                                                ▼
                                                     +----------------------+
                                                     | Validate & Sandbox   |
                                                     +----------------------+
                                                                │
                                                  ┌─────────────┴─────────────┐
                                                  │                           │
                                                Failed                      Passed
                                                  │                           │
                                                  ▼                           ▼
                                          +-------------+             +---------------------+
                                          | Observation |             | Update Tool Registry|
                                          | (Feedback)  |             +---------------------+
                                          +-------------+                     │
                                                                            └─────────────┐
                                                                                          ▼
                                                                              (Now tool exists)
                                                                                          │
                                                                                          ▼
                                                                      +-----------------------------+
                                                                      |  Format Tool Input          |
                                                                      +-----------------------------+
                                                                                          │
                                                                                          ▼
                                                                      +-----------------------------+
                                                                      |  Execute Tool (Sandbox)     |
                                                                      +-----------------------------+
                                                                                          │
                                                                                          ▼
                                                                      +-----------------------------+
                                                                      |    Observation (Result)     |
                                                                      +-----------------------------+
                                                                                          │
                                                                                          ▼
                                                                            (Feedback to Planner LLM)
 
```


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
