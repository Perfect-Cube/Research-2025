```
     +--------------+
     |  User Input  |
     +--------------+
             |
             v
+--------------------------+
| Pre-Filter (Your Script) |
+--------------------------+
             |
         [IF BLOCKED?]
           /      \
       Yes         No
        |           |
  +-----------+     v
  |   Stop    | +----------------+
  +-----------+ |  Generation    |
                |      LLM       |
                +----------------+
                         |
                         v
                   +-------------+
                   | Generated   |
                   |   Output    |
                   +-------------+
                         |
                         v
         +--------------------------------+
         | Post-Filter Layer 1            |
         | (Keywords, PII, Toxicity)      |
         +--------------------------------+
                         |
                     [IF SEVERE?]
                       /      \
                   Yes         No
                    |           |
              +----------+     v
              |  Block   | +---------------------------+
              +----------+ | Post-Filter Layer 2       |
                           | (Grounding, Fact-Checking)|
                           +---------------------------+
                                      |
                             [IF Hallucination/
                               Unsafe?]
                             /             \
                         Yes                No
                          |                  |
                     +---------+       +----------------------------+
                     |  Block  |       | Post-Filter Layer 3        |
                     +---------+       | (Patterns, Format)         |
                                         +----------------------------+
                                                  |
                                          [IF Gibberish/Invalid?]
                                             /              \
                                          Yes                No
                                           |                  |
                                      +----------+      +-------------------------+
                                      |  Block   |      | Post-Filter Layer 4     |
                                      +----------+      | (Model-as-Judge)        |
                                                        +-------------------------+
                                                                 |
                                                        [IF Judge Flags?]
                                                          /             \
                                                      Yes                No
                                                       |                  |
                                                  +----------+       +--------------------+
                                                  |  Block   |       | Final Output to    |
                                                  +----------+       |       User         |
                                                                     +--------------------+
 
```
```
                              +--------------+
                              |  User Input  |
                              +--------------+
                                     │
                                     ▼
                              +-----------------------+
                              |  Pre-Filter Script    |  
                              |  - Sanity/Policy Check|
                              +-----------------------+
                                     │
                                     │ (if input is flagged, STOP)
                                     ▼
                              +-----------------------+
                              |  Generation LLM       |
                              +-----------------------+
                                     │
                                     ▼
                              +-----------------------+
                              |  Generated Output     |
                              +-----------------------+
                                     │
                                     ▼
                              +---------------------------------------------------+
                              |         Post-Filters (Layers 1-4)                 |
                              |   1. Keywords, PII, Toxicity Check                |
                              |      - Severe issues result in BLOCK              |
                              |   2. Grounding & Fact-Checking                    |
                              |      - Hallucination/unsafe content → BLOCK       |
                              |   3. Patterns & Format Verification              |
                              |      - Gibberish or format errors → BLOCK         |
                              |   4. Model-as-Judge (Optional)                   |
                              |      - If judge flags content → BLOCK             |
                              +---------------------------------------------------+
                                     │
                                     │ (if any layer flags, output is BLOCK)
                                     ▼
                              +---------------------------+
                              | Final Output to User      |
                              +---------------------------+
 
```
