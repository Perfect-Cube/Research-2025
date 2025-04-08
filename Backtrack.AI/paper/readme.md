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
