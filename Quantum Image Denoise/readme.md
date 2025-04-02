

<p float="left">
  <img src="https://github.com/user-attachments/assets/22dedeb9-2fbb-49bc-a5a8-a1857ad0637b" width="500" />
  <img src="https://github.com/user-attachments/assets/70b5959e-c67a-4938-a0f5-8b7b32fde6a0"  width="500"/> 
</p>



```
      +---------------------------------------+
      |   Start: Input Noisy Image            |
      +---------------------------------------+
                     |
                     v
      +---------------------------------------+
      |   Preprocessing (Classical)           |
      +---------------------------------------+
                     |
                     v
      +---------------------------------------+
      |   Classical Feature Analysis          |
      |                                       |
      +---------------------------------------+
                     |  Gradients, Edges
                     v
      +---------------------------------------+
      |   Quantum Analysis Module             |
      |         (Histogram )                  |
      +---------------------------------------+
                     ^         |
   Image Patches/    |         | Encoded Data [For demonstration only]
        Data         |         v
      +-------------------------------+
      | Quantum Subroutine Execution  |
      +-------------------------------+
                     |
                     v
      +-----------------------------------------------+
      |   Quantum Tasks:     (classical counterpart)  |
      |    1. Accelerated NLM Patch Search            |
      |       (e.g., Grover-like?)                    |
      |    2. Quantum Statistical Analysis            |
      |    3. QML Inference for Parameters/Masks      |
      +-----------------------------------------------+
                     |
                     v   (Measurement)
      +-----------------------------------------------+
      |   Quantum Results:                            |
      |   Patch Indices, Optimal Params,              |
      |   Region Masks                                |
      +-----------------------------------------------+
                     |
                     v   (Decoded Classical Info)
      +-----------------------------------------------+
      |   Classical Filtering Control Logic           |
      +-----------------------------------------------+
                     |
                     v
      +-----------------------------------------------+
      |         Classical Filtering Pipeline          |
      |                                               |
      |   +------------------+    +---------------+   |
      |   |  Guided Filter   |    | Non-Local     |   |
      |   |                  |    | Means (NLM)   |   |
      |   +------------------+    +---------------+   |
      |            \                /                 |
      |             \              /                  |
      |              \            /                   |
      |           +--------------------------+        |
      |           | Layer Combination /      |        |
      |           | Reconstruction (Classical)|       |
      |           +--------------------------+        |
      |                     /    |                    |
      |                    /     |                    |
      |   +-----------------+   +-----------------+   |
      |   |   Median Filter |   | (Other Variants)|   |
      |   +-----------------+   +-----------------+   |
      +-----------------------------------------------+
                     |
                     v
      +---------------------------------------+
      |   Postprocessing (Optional)           |
      +---------------------------------------+
                     |
                     v
      +---------------------------------------+
      |     End: Denoised Image               |
      +---------------------------------------+

```
   -----------------------------------------------------------------
   NOTES on Variations in the Filtering Pipeline:

   1. Variation 1: 
      - Decomposition Path:
         [Preprocessing] --Decompose--> [Guided Filter] --(Detail Layer)-->
         [Non-Local Means] --(Denoised Detail)--> [Layer Combination]
         (Also, [Guided Filter] provides a Base Layer to the combination)

   2. Variation 2: 
      - Parallel Filtering:
         [Preprocessing] --> [GF Variant], [NLM Variant], [Median Variant]
         These outputs are fused together in [Layer Combination]

   3. Variation 3:
      - Guided NLM/Median:
         The Classical Filtering Control Logic may supply Edge Masks (e.g., from HOG analysis)
         to the Non-Local Means and Median Filter for enhanced guidance.
   -----------------------------------------------------------------
