# Input & Analysis Pipeline Flowchart

### 1. Input Phase
------------------------------------------------
| **Start:** Receive Input Prompt              |
|    (Text, Image, etc.)                         |
------------------------------------------------
             │
             ▼
------------------------------------------------
| **Preprocessing:**                           |
| - Tokenization & Language Detection          |
| - Normalize Input (remove noise, standardize case) |
------------------------------------------------

### 2. Fuzzy Matching Module
             │
             ▼
------------------------------------------------
| **Fuzzy Matching Module:**                   |
| - Apply Levenshtein, Soundex, and n-Gram matching |
------------------------------------------------

### 3. LLM Contextual Check
             │
             ▼
------------------------------------------------
| **LLM Contextual Check:**                    |
| - One-Shot Evaluation to understand context  |
------------------------------------------------

### 4. Category Analysis
             │
             ▼
------------------------------------------------
| **Category Analysis:**                       |
| - Check for Bias (Religion/Political)        |
| - Detect Jailbreak Attempts                  |
| - NSFW Content Screening                     |
------------------------------------------------

### 5. Severity Scoring
             │
             ▼
------------------------------------------------
| **Severity Scoring:**                        |
| - Assign a score (0-10) for each category      |
| - Aggregate weighted scores                  |
------------------------------------------------

### 6. Multi-Modal Integration
             │
             ▼
------------------------------------------------
| **Multi-Modal Integration:**                 |
| - For images/audio: extract features & match |
------------------------------------------------

### 7. Output Phase
             │
             ▼
------------------------------------------------
| **Output:**                                  |
| - Final Severity Score (0-10)                |
------------------------------------------------
 
