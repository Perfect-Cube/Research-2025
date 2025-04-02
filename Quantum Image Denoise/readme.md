<p float="left">
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/rain1.jpg" width="500" />
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/corrected/image.jpg"  width="500"/> 
</p>
<p float="left">
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/rain2.jpg" width="500" />
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/blur/rain_strong2_detail.png"  width="500"/> 
</p>
<p float="left">
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/rain3.jpg" width="500" />
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/blur/rain_strong3_detail.png"  width="500"/> 
</p>
<p float="left">
  <img src="https://github.com/user-attachments/assets/22dedeb9-2fbb-49bc-a5a8-a1857ad0637b" width="500" />
  <img src="https://github.com/user-attachments/assets/70b5959e-c67a-4938-a0f5-8b7b32fde6a0"  width="500"/> 
</p>
<p float="left">
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/rain5.jpg" width="500" />
  <img src="https://github.com/Perfect-Cube/Research-2025/blob/main/Quantum%20Image%20Denoise/image/blur/rain_strong5_detail.png"  width="500"/> 
</p>

## Rain mask

![rain3_mask](https://github.com/user-attachments/assets/d36493be-4ec8-4173-a30e-79e426c07d23)

# dataflow

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
