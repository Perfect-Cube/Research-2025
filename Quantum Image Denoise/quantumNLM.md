Okay, let's attempt to implement a *possible* quantum component using Qiskit for the NLM patch similarity calculation.

**Disclaimer:**

1.  **Highly Experimental & Slow:** This uses quantum circuit simulation on a classical computer. It will be **extremely slow**, likely orders of magnitude slower than even the pure Python NLM simulation, let alone OpenCV's optimized version. Its purpose is educational – to show *how* one might structure such a calculation using quantum primitives.
2.  **Impractical on Real Hardware:** Running this on current quantum hardware is infeasible due to qubit limitations, noise, circuit depth, and the difficulty of efficient state preparation (QRAM).
3.  **Encoding Limitations:** We'll use **amplitude encoding** for patches. This encodes the *relative* pixel intensities, normalized, into quantum amplitudes. It loses absolute brightness information and requires careful interpretation. Other encodings exist but are more complex or require far more qubits.
4.  **Quantum Primitive:** We will use the **SWAP Test** algorithm to estimate the fidelity (squared overlap `|<psi|phi>|^2`) between the quantum states representing two patches. Higher fidelity means higher similarity.
5.  **Heuristic Weighting:** We will heuristically convert the fidelity (`sim`) into an NLM weight: `weight = exp(-(1 - sim) / h_norm_sq)`, where `(1 - sim)` acts as a proxy for squared distance (ranging 0 to 1) and `h_norm_sq` is a normalized NLM parameter.

**Steps:**

1.  **Install Qiskit:**
    ```bash
    pip install qiskit qiskit-aer qiskit-algorithms matplotlib scipy numpy requests Pillow opencv-python
    # Or: pip install qiskit[visualization] qiskit-aer ...
    ```
2.  **Add a `QuantumNLMHelper` Class:** This class will encapsulate the quantum logic (patch encoding, SWAP test circuit, execution, result processing).
3.  **Modify `hybrid_nlm_denoising`:** Call the `QuantumNLMHelper` to get patch similarity weights.
4.  **Add Command-Line Arguments:** Control whether to use the quantum NLM, set simulator shots, and the normalized `h` parameter.

**Code:**

```python
# --- Imports ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import requests
from io import BytesIO
import time
import random
from PIL import Image
# from scipy.signal import convolve2d # Not needed for quantum version structure

# --- Qiskit Imports ---
QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.basic_provider import BasicSimulator # Basic backend
    # Use Aer for better performance and noise models if needed
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit.primitives import Sampler # Modern way for shots
    QISKIT_AVAILABLE = True
    print("Qiskit found and imported successfully.")
except ImportError:
    print("WARNING: Qiskit not found. Quantum NLM will not be available.")
    print("Install it: pip install qiskit qiskit-aer")


# --- Utility: cv2_imshow ---
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        print(f"Displaying image with shape: {img.shape}, dtype: {img.dtype}")
        try:
            cv2.imshow("Image", img); cv2.waitKey(0); cv2.destroyAllWindows()
        except cv2.error as e:
             print(f"\n--- NOTE: cv2.imshow failed ({e}). Visual output skipped. ---")

# --- Stage 1 Functions (compute_blob_orientation, build_orientation_histogram, detect_rain_candidates, remove_rain_using_inpainting, first_stage_derain) ---
# --- [PASTE THE STAGE 1 FUNCTIONS FROM PREVIOUS WORKING VERSION HERE] ---
def compute_blob_orientation(contour):
    if len(contour) >= 5:
        try:
            contour_float = contour.astype(np.float32); ellipse = cv2.fitEllipse(contour_float); return ellipse[2]
        except cv2.error: return None
    return None

def build_orientation_histogram(angles, weights, bins=18):
    hist, bin_edges = np.histogram(angles, bins=bins, range=(0, 180), weights=weights); return hist, bin_edges

def detect_rain_candidates(image, roi_fraction=0.65):
    if image is None or image.size==0: return [],[],np.zeros((1,1),dtype=np.uint8),np.zeros((1,1),dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3: gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) if image.shape[2]==4 else image; gray=gray.squeeze()
    else: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.ndim!=2: return [],[],np.zeros_like(image[:,:,0]),np.zeros_like(image[:,:,0])
    h,w=gray.shape; roi_h=int(h*roi_fraction); roi_mask=np.zeros_like(gray); roi_mask[0:max(1,roi_h),:]=255
    edges = cv2.Canny(gray, 50, 150); edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)); dilated=cv2.dilate(edges_roi,kernel,iterations=1)
    contours,_=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    orientations=[]; weights=[]; rain_mask_output=np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<15 or area>600: continue
        angle=compute_blob_orientation(cnt)
        if angle is not None: orientations.append(angle); weights.append(area); cv2.drawContours(rain_mask_output,[cnt],-1,255,-1)
    rain_mask_output = cv2.bitwise_and(rain_mask_output, rain_mask_output, mask=roi_mask)
    return orientations, weights, rain_mask_output, edges_roi

def remove_rain_using_inpainting(image, rain_mask):
    if image is None or rain_mask is None or image.size==0 or rain_mask.size==0: return image
    if image.shape[:2]!=rain_mask.shape[:2]: rain_mask=cv2.resize(rain_mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    if np.count_nonzero(rain_mask)==0: return image
    return cv2.inpaint(image, rain_mask, 5, cv2.INPAINT_NS)

def first_stage_derain(image, roi_fraction=0.65):
    print("\n--- Starting First Stage Deraining ---")
    angles, weights, rain_mask, edge_img_roi = detect_rain_candidates(image, roi_fraction=roi_fraction)
    result = image.copy(); orientation_threshold = 0.35
    if angles:
        try:
            hist, bin_edges = build_orientation_histogram(angles, weights, bins=18)
            # Plotting (optional)
            # plt.figure(figsize=(8, 4)); bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2; bin_width_plot = bin_edges[1] - bin_edges[0]
            # plt.bar(bin_centers, hist, width=bin_width_plot, align='center', edgecolor='black'); plt.xlabel('Orientation (degrees)'); plt.ylabel('Weighted count (area)')
            # plt.title(f'Classical HOS (Top {int(roi_fraction*100)}% ROI)'); plt.xticks(np.arange(0, 181, 20)); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()
            vertical_bins_centers = np.where((bin_centers >= 70) & (bin_centers <= 110))[0]
            vertical_weight_sum = np.sum(hist[vertical_bins_centers]) if len(vertical_bins_centers) > 0 else 0
            total_weight_sum = np.sum(hist)
            if total_weight_sum > 1e-6:
                vertical_fraction = vertical_weight_sum / total_weight_sum
                if vertical_fraction > orientation_threshold:
                    print(f"Rain likely detected: Vertical orientation dominates ({vertical_fraction:.1%}). Inpainting."); result = remove_rain_using_inpainting(image, rain_mask)
                else: print(f"No dominant rain orientation detected ({vertical_fraction:.1%}). Skipping inpainting.")
            else: print("Total weight sum zero. Skipping inpainting.")
        except Exception as e: print(f"Histogram error: {e}. Skipping inpainting.")
    else: print("No candidate streaks detected. Skipping Stage 1 inpainting.")
    # Display (optional)
    # print("Result After Stage 1:"); cv2_imshow(result)
    print("--- End of First Stage ---")
    return result

# --- Stage 2 Functions ---

# Guided Filter Check
GUIDED_FILTER_AVAILABLE = False
try:
    if 'ximgproc' in dir(cv2) and hasattr(cv2.ximgproc, 'guidedFilter'):
        GUIDED_FILTER_AVAILABLE = True; print("Found cv2.ximgproc.guidedFilter.")
    else: print("Warning: cv2.ximgproc.guidedFilter not found. Using Gaussian Blur fallback.")
except Exception as e: print(f"Error checking guidedFilter: {e}. Using Gaussian Blur fallback.")

# --- Quantum NLM Helper Class ---
class QuantumNLMHelper:
    """ Encapsulates quantum logic for NLM similarity using SWAP Test. """
    def __init__(self, patch_size, shots=1024):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not available. Cannot use QuantumNLMHelper.")

        self.patch_pixels = patch_size * patch_size
        self.num_patch_qubits = (self.patch_pixels - 1).bit_length()
        if self.num_patch_qubits <= 0:
             raise ValueError("Patch size must be at least 1x1.")

        self.num_swap_qubits = 2 * self.num_patch_qubits + 1 # 1 ancilla
        self.shots = shots
        # Use Aer's simulator for speed
        self.simulator = AerSimulator(method='statevector') # Use statevector for perfect overlap calc
        # Or use QASM simulator for shot-based estimation:
        # self.simulator = AerSimulator(method='automatic') # Or 'qasm'
        # self.sampler = Sampler() # For shot-based

        print(f"QuantumNLMHelper initialized:")
        print(f"  Patch size: {patch_size}x{patch_size} ({self.patch_pixels} pixels)")
        print(f"  Qubits per patch state: {self.num_patch_qubits}")
        print(f"  Total SWAP Test qubits: {self.num_swap_qubits}")
        print(f"  Simulator: {self.simulator.configuration().backend_name} (Method: {self.simulator.options.method})")
        print(f"  Shots (for QASM): {self.shots}")


    def _normalize_patch(self, patch_classical):
        """ Flattens and L2-normalizes a patch for amplitude encoding. """
        patch_flat = patch_classical.flatten().astype(np.float64)
        norm = np.linalg.norm(patch_flat)
        if norm < 1e-9: # Avoid division by zero for black patches
            # Return a state vector that is valid but represents zero (e.g., |0...0>)
            # state_vector = np.zeros(2**self.num_patch_qubits)
            # state_vector[0] = 1.0
            # Or return None to indicate failure/skip
             return None
        normalized_vector = patch_flat / norm
        # Ensure vector length matches qubit state space (pad with zeros if needed)
        state_vector = np.zeros(2**self.num_patch_qubits, dtype=complex)
        state_vector[:len(normalized_vector)] = normalized_vector
        # Re-normalize just in case padding affected it slightly (shouldn't if norm>0)
        final_norm = np.linalg.norm(state_vector)
        if final_norm > 1e-9 : state_vector = state_vector / final_norm
        else: state_vector[0] = 1.0 ; # Fallback state |0...0>

        return state_vector


    def _create_swap_test_circuit(self, state_vector1, state_vector2):
        """ Creates the SWAP test circuit. """
        if state_vector1 is None or state_vector2 is None: return None

        qc = QuantumCircuit(self.num_swap_qubits, 1, name="SWAP_Test") # 1 classical bit for measurement
        ancilla_idx = 0
        patch1_indices = list(range(1, self.num_patch_qubits + 1))
        patch2_indices = list(range(self.num_patch_qubits + 1, self.num_swap_qubits))

        # Initialize states (Qiskit >= 0.45 takes state vector directly)
        try:
            qc.initialize(state_vector1, patch1_indices)
            qc.initialize(state_vector2, patch2_indices)
        except Exception as e:
             print(f"Error during qc.initialize: {e}")
             # print(f"State vector 1 norm: {np.linalg.norm(state_vector1)}")
             # print(f"State vector 2 norm: {np.linalg.norm(state_vector2)}")
             # print(f"State vector 1 length: {len(state_vector1)}")
             # print(f"State vector 2 length: {len(state_vector2)}")
             # print(f"Expected length: {2**self.num_patch_qubits}")
             return None # Indicate failure

        # Build SWAP Test
        qc.h(ancilla_idx)
        for i in range(self.num_patch_qubits):
            qc.cswap(ancilla_idx, patch1_indices[i], patch2_indices[i])
        qc.h(ancilla_idx)
        qc.measure(ancilla_idx, 0) # Measure ancilla into classical bit 0
        return qc

    def calculate_quantum_similarity(self, patch_ref_classical, patch_cand_classical):
        """ Calculates similarity |<Pref|Pi>|^2 using SWAP test simulation. """
        # 1. Normalize patches and get state vectors
        state_vector_ref = self._normalize_patch(patch_ref_classical)
        state_vector_cand = self._normalize_patch(patch_cand_classical)

        if state_vector_ref is None or state_vector_cand is None:
            # Handle black patches or normalization errors -> assign zero similarity
            return 0.0

        # 2. Create SWAP test circuit
        swap_circuit = self._create_swap_test_circuit(state_vector_ref, state_vector_cand)
        if swap_circuit is None: return 0.0 # Circuit creation failed

        # 3. Run simulation
        try:
            # Using Statevector simulator provides exact overlap without shots
            if self.simulator.configuration().backend_name == 'aer_simulator_statevector':
                 # Transpile for simulator
                 t_circuit = transpile(swap_circuit, self.simulator)
                 result = self.simulator.run(t_circuit, shots=1).result() # Only 1 shot needed for statevector
                 statevector_final = result.get_statevector(t_circuit)
                 # The probability P(0) is the squared magnitude of amplitudes where ancilla is |0>
                 # Find indices corresponding to ancilla |0>
                 prob0 = 0.0
                 num_total_states = 2**self.num_swap_qubits
                 for i in range(num_total_states):
                     # Check if the least significant bit (ancilla) is 0
                     if (i >> (self.num_swap_qubits -1)) == 0: # Assuming ancilla is most significant qubit after initialize maybe? Check Qiskit convention.
                     # Let's assume Qiskit convention puts ancilla at index 0, so check least significant bit:
                     # if (i & 1) == 0: # Check if ancilla qubit (index 0) is 0
                          prob0 += np.abs(statevector_final[i])**2
                 # If statevector sim behaves unexpectedly, fallback to counts even with 1 shot
                 # counts = result.get_counts()
                 # prob0 = counts.get('0', 0) / 1.0 # Prob of measuring 0

            else: # Using QASM simulator with shots
                 # Transpile for simulator
                 t_circuit = transpile(swap_circuit, self.simulator)
                 result = self.simulator.run(t_circuit, shots=self.shots).result()
                 counts = result.get_counts()
                 prob0 = counts.get('0', 0) / self.shots # Prob of measuring 0

            # 4. Calculate Fidelity (Overlap Squared)
            # P(0) = (1 + |<ref|cand>|^2) / 2
            # fidelity = |<ref|cand>|^2 = 2 * P(0) - 1
            fidelity = 2.0 * prob0 - 1.0

            # Clamp due to potential simulation noise/errors
            fidelity = max(0.0, min(1.0, fidelity))
            return fidelity

        except Exception as e:
            print(f"\nERROR during quantum simulation: {e}")
            # print(f"Circuit: {swap_circuit.draw('text')}") # Optional: Draw circuit on error
            return 0.0 # Return minimum similarity on error


    def calculate_quantum_weight(self, patch_ref_classical, patch_cand_classical, h_norm_sq):
        """ Calculates NLM weight based on quantum similarity. """
        if h_norm_sq < 1e-9: return 0.0 # Avoid division by zero

        # Get similarity (fidelity) using quantum simulation
        quantum_similarity = self.calculate_quantum_similarity(patch_ref_classical, patch_cand_classical)

        # Heuristic: Treat (1 - similarity) as squared distance proxy [0, 1]
        distance_sq_proxy = 1.0 - quantum_similarity

        # Calculate NLM weight
        weight = np.exp(-distance_sq_proxy / h_norm_sq)
        return weight


# --- Modified Hybrid NLM to use Quantum Helper ---
def hybrid_nlm_denoising_quantum(img_uint8, h=10.0, templateWindowSize=7, searchWindowSize=21,
                                 use_quantum=False, shots=1024):
    """
    Performs NLM using QuantumNLMHelper (if use_quantum=True) or classical calculation.
    NOTE: Quantum version is extremely slow due to simulation overhead.
    """
    mode = "Quantum" if use_quantum else "Classical (Simulation)"
    print(f"--- Starting Hybrid NLM ({mode}) ---")
    print(f"    Params: h={h}, templateWin={templateWindowSize}, searchWin={searchWindowSize}")
    start_nlm_time = time.time()

    if img_uint8 is None or img_uint8.size == 0: return None
    is_color = img_uint8.ndim == 3 and img_uint8.shape[2] == 3
    if is_color: print("    WARNING: Quantum NLM currently processes only grayscale. Converting internally."); img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    else: img_gray = img_uint8

    # --- Parameters ---
    tW = templateWindowSize; sW = searchWindowSize
    tW_half = tW // 2; sW_half = sW // 2
    patch_size = tW

    # --- Padding ---
    pad_size = tW_half + sW_half
    padded_img = cv2.copyMakeBorder(img_gray, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    padded_img_float = padded_img.astype(np.float64) # Use float for calculations

    # --- Output Image (Grayscale) ---
    denoised_img_gray = np.zeros_like(padded_img_float)

    # --- Quantum Helper Initialization (if needed) ---
    quantum_helper = None
    h_norm_sq = 0.0 # Normalized h^2 for quantum weight calculation
    if use_quantum:
        if not QISKIT_AVAILABLE:
             print("    ERROR: Qiskit not found, cannot use quantum NLM. Aborting.")
             return None
        try:
            quantum_helper = QuantumNLMHelper(patch_size=patch_size, shots=shots)
            # Heuristic normalization for h: Scale h (0-255 range) to ~0-1 range for similarity distance (0-1)
            # Smaller h_norm means faster decay -> needs higher similarity
            h_norm = max(0.01, h / 100.0) # Adjust the scaling factor (e.g., 50, 100, 255) based on expected similarity values
            h_norm_sq = h_norm * h_norm
            print(f"    Using Quantum Path with h_norm^2 = {h_norm_sq:.4f}")
        except Exception as e:
            print(f"    ERROR initializing QuantumNLMHelper: {e}. Aborting.")
            return None
    else:
        # Classical parameters
        h_sq = h * h
        sigma_sq = h_sq # Denominator for classical weight exp(-dist^2 / sigma^2)
        patch_size_sq_classical = tW * tW # * num_channels=1 for grayscale


    # --- Iterate through each pixel ---
    rows, cols = img_gray.shape[:2]
    total_pixels = rows * cols; processed_pixels = 0; last_print_time = time.time()

    for r in range(rows):
        for c in range(cols):
            r_pad = r + pad_size; c_pad = c + pad_size

            # Extract reference patch (classical, float64)
            ref_patch = padded_img_float[r_pad - tW_half : r_pad + tW_half + 1,
                                         c_pad - tW_half : c_pad + tW_half + 1]

            # --- Search Window ---
            r_min_s = r_pad - sW_half; r_max_s = r_pad + sW_half
            c_min_s = c_pad - sW_half; c_max_s = c_pad + sW_half

            total_weight = 0.0
            accumulated_pixel_value = 0.0
            weights_list = [] # Store weights for normalization
            center_pixels_list = [] # Store corresponding center pixel values

            # Iterate through candidate patches in search window
            for r_cand_center in range(r_min_s, r_max_s + 1):
                for c_cand_center in range(c_min_s, c_max_s + 1):
                    # Candidate patch top-left corner
                    r_cand_tl = r_cand_center - tW_half
                    c_cand_tl = c_cand_center - tW_half

                    # Extract candidate patch (classical, float64)
                    cand_patch = padded_img_float[r_cand_tl : r_cand_tl + tW,
                                                  c_cand_tl : c_cand_tl + tW]

                    # Store center pixel value
                    center_pixel_val = padded_img_float[r_cand_center, c_cand_center]
                    center_pixels_list.append(center_pixel_val)

                    # --- Calculate Weight ---
                    weight = 0.0
                    if use_quantum and quantum_helper:
                        # *** Quantum Weight Calculation ***
                        weight = quantum_helper.calculate_quantum_weight(ref_patch, cand_patch, h_norm_sq)
                    else:
                        # *** Classical Weight Calculation ***
                        diff = ref_patch - cand_patch
                        dist_sq = np.sum(diff**2)
                        weight = np.exp(-dist_sq / sigma_sq)

                    weights_list.append(weight)

            # --- Normalize weights and calculate final pixel value ---
            weights_arr = np.array(weights_list)
            total_weight = np.sum(weights_arr)

            if total_weight > 1e-9:
                normalized_weights = weights_arr / total_weight
                center_pixels_arr = np.array(center_pixels_list)
                denoised_pixel_value = np.sum(normalized_weights * center_pixels_arr)
            else: # If all weights are zero, use original pixel
                denoised_pixel_value = padded_img_float[r_pad, c_pad]

            denoised_img_gray[r_pad, c_pad] = denoised_pixel_value

            # --- Progress Update ---
            processed_pixels += 1; current_time = time.time()
            if current_time - last_print_time > 10.0: # Print update every 10s (quantum is slow)
                 progress = (processed_pixels / total_pixels) * 100; elapsed = current_time - start_nlm_time
                 print(f"    Hybrid NLM ({mode}) Progress: {progress:.1f}% ({processed_pixels}/{total_pixels} pixels, {elapsed:.1f}s elapsed)")
                 last_print_time = current_time

    # --- Crop back ---
    denoised_img_cropped = denoised_img_gray[pad_size:-pad_size, pad_size:-pad_size]

    # Clip and convert back to uint8
    denoised_final_uint8 = np.clip(denoised_img_cropped, 0, 255).astype(np.uint8)

    end_nlm_time = time.time()
    print(f"--- Hybrid NLM ({mode}) finished in {end_nlm_time - start_nlm_time:.2f}s ---")

    # If original was color, convert grayscale result back to BGR
    if is_color:
         print("    Converting grayscale NLM result back to BGR.")
         denoised_final_uint8 = cv2.cvtColor(denoised_final_uint8, cv2.COLOR_GRAY2BGR)

    return denoised_final_uint8


# --- Modified derain_guided_nlm ---
def derain_guided_nlm(image_arr_noisy,
                      gf_radius=15, gf_eps=0.1**2,
                      nlm_h_detail=8.0, nlm_win_detail=7, nlm_search_detail=21,
                      nlm_h_base=10.0, nlm_win_base=7, nlm_search_base=21,
                      apply_nlm_base=True,
                      use_quantum_nlm_detail=False, # Flag to enable QNLM
                      quantum_shots=1024):
    """
    Uses Guided Filter + NLM. Allows using Quantum NLM (simulated) for detail layer.
    """
    if image_arr_noisy is None: return None
    print("\n--- Starting Second Stage Deraining (Guided Filter + NLM) ---")
    start_time = time.time(); img_uint8 = image_arr_noisy.astype(np.uint8)
    is_color = (img_uint8.ndim == 3 and img_uint8.shape[2] == 3)
    img_float = img_uint8.astype(np.float32) / 255.0
    if is_color:
        try: guide_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        except cv2.error: guide_img = img_float[:,:,0] if img_float.ndim==3 else img_float
    else: guide_img = img_float
    print(f"  Computing Base Layer...")
    if GUIDED_FILTER_AVAILABLE:
        if guide_img.ndim == 3: guide_img = guide_img[:,:,0]
        if (img_float.ndim==2 and guide_img.ndim==2) or (img_float.ndim==3 and guide_img.ndim==2):
            base_layer = cv2.ximgproc.guidedFilter(guide=guide_img, src=img_float, radius=gf_radius, eps=gf_eps, dDepth=cv2.CV_32F)
        else: print("GF Fallback: Gaussian"); blur_ksize = max(3,(gf_radius//2)*2+1); base_layer = cv2.GaussianBlur(img_float, (blur_ksize, blur_ksize), 0)
    else: print("GF Fallback: Gaussian"); blur_ksize = max(3,(gf_radius//2)*2+1); base_layer = cv2.GaussianBlur(img_float, (blur_ksize, blur_ksize), 0)
    base_layer = np.clip(base_layer, 0.0, 1.0).astype(np.float32); gf_time = time.time()
    detail_layer = img_float - base_layer
    print(f"  Processing Detail Layer...")
    detail_min = np.min(detail_layer); detail_max = np.max(detail_layer); detail_range = detail_max - detail_min
    detail_denoised = detail_layer
    if detail_range > 1e-6:
        detail_uint8 = np.clip(((detail_layer - detail_min) / detail_range) * 255.0, 0, 255).astype(np.uint8)
        if use_quantum_nlm_detail:
            # *** Use Quantum NLM ***
            print("    Using Quantum NLM (Simulated)...")
            detail_denoised_uint8 = hybrid_nlm_denoising_quantum(
                detail_uint8, h=float(nlm_h_detail), templateWindowSize=nlm_win_detail,
                searchWindowSize=nlm_search_detail, use_quantum=True, shots=quantum_shots)
            if detail_denoised_uint8 is None: print("    Quantum NLM failed. Skipping detail denoising."); detail_denoised_uint8 = detail_uint8
        else:
             # *** Use Classical OpenCV NLM ***
             print(f"    Using Classical OpenCV NLM (h={nlm_h_detail}, win={nlm_win_detail}, search={nlm_search_detail})...")
             try:
                 if is_color and detail_uint8.ndim == 3: detail_denoised_uint8 = cv2.fastNlMeansDenoisingColored(detail_uint8, None, h=float(nlm_h_detail), hColor=float(nlm_h_detail), templateWindowSize=nlm_win_detail, searchWindowSize=nlm_search_detail)
                 elif not is_color and detail_uint8.ndim == 2: detail_denoised_uint8 = cv2.fastNlMeansDenoising(detail_uint8, None, h=float(nlm_h_detail), templateWindowSize=nlm_win_detail, searchWindowSize=nlm_search_detail)
                 else: print("    Warning: Skip NLM, unexpected dims."); detail_denoised_uint8 = detail_uint8
             except Exception as e: print(f"    OpenCV NLM Error: {e}"); detail_denoised_uint8 = detail_uint8 # Fallback on error
        # Convert back to float range
        detail_denoised = ((detail_denoised_uint8.astype(np.float32) / 255.0) * detail_range) + detail_min
    else: print("    Detail layer range near zero, skipping NLM.")
    nlm_detail_time = time.time()
    print(f"    Detail Layer processing completed in {nlm_detail_time - gf_time:.2f}s")
    base_denoised = base_layer
    if apply_nlm_base:
        print(f"  Applying Classical NLM on Base Layer (h={nlm_h_base}, win={nlm_win_base}, search={nlm_search_base})...")
        base_uint8 = np.clip(base_layer * 255.0, 0, 255).astype(np.uint8)
        try:
            if is_color and base_uint8.ndim == 3: base_denoised_uint8 = cv2.fastNlMeansDenoisingColored(base_uint8, None, h=float(nlm_h_base), hColor=float(nlm_h_base), templateWindowSize=nlm_win_base, searchWindowSize=nlm_search_base)
            elif not is_color and base_uint8.ndim == 2: base_denoised_uint8 = cv2.fastNlMeansDenoising(base_uint8, None, h=float(nlm_h_base), templateWindowSize=nlm_win_base, searchWindowSize=nlm_search_base)
            else: print("    Warning: Skip Base NLM, unexpected dims."); base_denoised_uint8 = base_uint8
            base_denoised = base_denoised_uint8.astype(np.float32) / 255.0
        except Exception as e: print(f"    OpenCV Base NLM Error: {e}") # Use original base on error
    else: print("  Skipping NLM on Base Layer.")
    denoised_float = base_denoised + detail_denoised
    denoised_clipped = np.clip(denoised_float, 0.0, 1.0); denoised_uint8 = (denoised_clipped * 255.0).astype(np.uint8)
    end_time = time.time(); print(f"--- Second-stage deraining completed in {end_time - start_time:.2f}s ---")
    return denoised_uint8

# --- Image Loading, Noise Addition, Combined Workflow, Main ---
# --- [PASTE load_image, add_rain, run_combined_derain_process, __main__ block FROM PREVIOUS VERSION HERE] ---
# --- Make sure to add the new arguments to the parser in __main__ ---

def add_rain(image_arr, intensity=0.3, slant=-1, drop_length=20, thickness=1):
    if image_arr is None: return None
    noisy_image = image_arr.copy(); img_h, img_w = noisy_image.shape[:2]
    is_color=(noisy_image.ndim==3); num_drops=int(intensity*img_w*img_h*0.001)
    for _ in range(num_drops):
        x1=random.randint(0,img_w-1); y1=random.randint(-int(img_h*0.1),img_h-1)
        current_slant=slant+random.uniform(-0.8,0.8); current_length=random.randint(max(10,drop_length-15),drop_length+20)
        if current_length<=0: continue
        x2=int(np.clip(x1+current_slant*current_length,0,img_w-1)); y2=int(np.clip(y1+current_length,0,img_h-1))
        if abs(x1-x2)<1 and abs(y1-y2)<1: continue
        brightness=random.randint(160,255); line_color=(brightness,)*3 if is_color else brightness
        current_thickness=random.randint(max(1,thickness-1),thickness+1)
        cv2.line(noisy_image,(x1,y1),(x2,y2),line_color,current_thickness,lineType=cv2.LINE_AA)
    return np.clip(noisy_image,0,255).astype(image_arr.dtype)

def load_image_from_url_or_path(url_or_path):
    try:
        if url_or_path.startswith(('http','https')):
            print(f"Loading URL: {url_or_path}"); headers={'User-Agent':'Mozilla/5.0'}; response=requests.get(url_or_path, timeout=20, headers=headers, stream=True); response.raise_for_status(); img_pil=Image.open(BytesIO(response.content))
        else: print(f"Loading Path: {url_or_path}"); img_pil=Image.open(url_or_path)
        img_rgb=np.array(img_pil.convert('RGB')); img_bgr=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR); print(f"Image loaded: {img_bgr.shape}"); return img_bgr
    except Exception as e: print(f"Error loading image: {e}"); return None

def run_combined_derain_process(image_path_or_url, output_path, args):
    print(f"--- Starting Combined Deraining Process ---"); print(f"Input: {image_path_or_url}, Output: {output_path}"); # print(f"Params: {vars(args)}")
    image_arr_orig = load_image_from_url_or_path(image_path_or_url)
    if image_arr_orig is None: return
    original_height, original_width = image_arr_orig.shape[:2]; print(f"Original dims: {original_width}x{original_height}")
    image_arr_proc = image_arr_orig.copy()
    if args.resize:
        print(f"Resizing width to {args.resize}..."); aspect_ratio=image_arr_proc.shape[0]/image_arr_proc.shape[1]; new_height=int(args.resize*aspect_ratio)
        try: image_arr_proc=cv2.resize(image_arr_proc,(args.resize,new_height),interpolation=cv2.INTER_LANCZOS4)
        except Exception as e: print(f"Resize failed: {e}"); args.resize = None
    temp_noisy_path = None
    if args.add_noise:
        print(f"Adding synthetic noise (intensity={args.intensity})..."); noisy_image = add_rain(image_arr_proc, intensity=args.intensity)
        if noisy_image is not None: image_arr_proc=noisy_image; temp_noisy_path=f"temp_noisy_{int(time.time())}.png" #; cv2_imshow(image_arr_proc)
        else: print("Error adding noise.")
    try: first_stage_result=first_stage_derain(image_arr_proc, roi_fraction=args.roi_fraction); first_stage_result=first_stage_result if first_stage_result is not None else image_arr_proc.copy()
    except Exception as e: print(f"Critical Error Stage 1: {e}"); first_stage_result=image_arr_proc.copy()
    try:
        second_stage_result=derain_guided_nlm(first_stage_result, gf_radius=args.gf_r, gf_eps=args.gf_e**2, nlm_h_detail=args.nlm_h_d, nlm_win_detail=args.nlm_w_d, nlm_search_detail=args.nlm_s_d, nlm_h_base=args.nlm_h_b, nlm_win_base=args.nlm_w_b, nlm_search_base=args.nlm_s_b, apply_nlm_base=(not args.no_nlm_base), use_quantum_nlm_detail=args.use_quantum_nlm, quantum_shots=args.quantum_shots)
        if second_stage_result is None: print("Error: Stage 2 failed."); return
    except Exception as e: print(f"Critical Error Stage 2: {e}"); return
    final_result = second_stage_result
    if args.resize and (final_result.shape[1]!=original_width or final_result.shape[0]!=original_height):
        print(f"Resizing final back to original..."); final_result=cv2.resize(final_result,(original_width,original_height),interpolation=cv2.INTER_LANCZOS4)
    print("\n--- Saving Final Result ---")
    try:
        output_dir=os.path.dirname(output_path);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        Image.fromarray(cv2.cvtColor(final_result,cv2.COLOR_BGR2RGB)).save(output_path); print(f"Final image saved to '{output_path}'")
    except Exception as e: print(f"Error saving final image: {e}")
    if temp_noisy_path and os.path.exists(temp_noisy_path):
        try: os.remove(temp_noisy_path)
        except OSError as e: print(f"Error removing temp file: {e}")
    print("--- Combined Deraining Process Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Deraining Pipeline with Optional Quantum NLM Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_image", help="Path or URL to input image.")
    parser.add_argument("-o", "--output_image", default="derained_quantum_hybrid.png", help="Path to save final image.")
    parser.add_argument("--add_noise", action='store_true', help="Add synthetic rain noise.")
    parser.add_argument("--intensity", type=float, default=0.4, help="Synthetic noise intensity.")
    parser.add_argument("--resize", type=int, default=None, help="Resize image width before processing.")
    parser.add_argument("--roi_fraction", type=float, default=0.65, help="ROI fraction for Stage 1.")

    stage2_group = parser.add_argument_group('Stage 2 Parameters')
    stage2_group.add_argument("--gf_r", type=int, default=15, help="Guided Filter radius.")
    stage2_group.add_argument("--gf_e", type=float, default=0.1, help="Guided Filter epsilon (sqrt(eps)).")
    stage2_group.add_argument("--nlm_h_d", type=float, default=8.0, help="NLM strength (h) for Detail Layer.")
    stage2_group.add_argument("--nlm_w_d", type=int, default=7, help="NLM template window size (Detail).")
    stage2_group.add_argument("--nlm_s_d", type=int, default=21, help="NLM search window size (Detail).")
    stage2_group.add_argument("--no_nlm_base", action='store_true', help="Skip classical NLM on Base Layer.")
    stage2_group.add_argument("--nlm_h_b", type=float, default=10.0, help="NLM strength (h) for Base Layer.")
    stage2_group.add_argument("--nlm_w_b", type=int, default=7, help="NLM template window size (Base).")
    stage2_group.add_argument("--nlm_s_b", type=int, default=21, help="NLM search window size (Base).")

    qnlm_group = parser.add_argument_group('Quantum NLM Simulation Parameters (for Detail Layer)')
    qnlm_group.add_argument("--use_quantum_nlm", action='store_true', help="Use Quantum NLM simulation (VERY SLOW) for detail layer instead of OpenCV NLM.")
    qnlm_group.add_argument("--quantum_shots", type=int, default=1024, help="Number of shots for QASM simulator (if used). Statevector simulator ignores this.")
    # Note: We derive h_norm inside the quantum NLM function based on nlm_h_d

    args = parser.parse_args()

    # Validation / Adjustment
    args.nlm_w_d = args.nlm_w_d if args.nlm_w_d % 2 != 0 else args.nlm_w_d + 1
    args.nlm_s_d = args.nlm_s_d if args.nlm_s_d % 2 != 0 else args.nlm_s_d + 1
    args.nlm_w_b = args.nlm_w_b if args.nlm_w_b % 2 != 0 else args.nlm_w_b + 1
    args.nlm_s_b = args.nlm_s_b if args.nlm_s_b % 2 != 0 else args.nlm_s_b + 1
    if not 0.0 < args.roi_fraction <= 1.0: args.roi_fraction = 0.65
    if args.use_quantum_nlm and not QISKIT_AVAILABLE:
        print("ERROR: --use_quantum_nlm specified but Qiskit is not available. Exiting.")
        exit(1)
    if args.use_quantum_nlm:
        print("\n***********************************************************")
        print("WARNING: Quantum NLM simulation enabled.")
        print("         This involves heavy classical simulation of quantum")
        print("         circuits and will be EXTREMELY SLOW.")
        print("         Use '--use_quantum_nlm' only for demonstration.")
        print("***********************************************************\n")
        # Force smaller window size for quantum demo if not specified? Optional.
        # if args.nlm_w_d > 5: print("INFO: Reducing NLM template window to 5x5 for quantum demo speed.") ; args.nlm_w_d = 5
        # if args.nlm_s_d > 11: print("INFO: Reducing NLM search window to 11x11 for quantum demo speed.") ; args.nlm_s_d = 11


    run_combined_derain_process(args.input_image, args.output_path, args)
```

**How to Run:**

1.  Save the code (e.g., `hybrid_derain.py`).
2.  Install dependencies: `pip install qiskit qiskit-aer opencv-python Pillow numpy matplotlib requests scipy`
3.  Run classically (fast):
    ```bash
    python hybrid_derain.py path/to/your/rainy_image.jpg -o derained_classical.png
    ```
4.  Run with Quantum NLM simulation ( **EXPECT THIS TO TAKE HOURS OR DAYS** depending on image size and parameters):
    ```bash
    python hybrid_derain.py path/to/your/rainy_image.jpg -o derained_quantum_sim.png --use_quantum_nlm --nlm_w_d 5 --nlm_s_d 11
    # Using smaller windows (--nlm_w_d 5 --nlm_s_d 11) is recommended for the quantum simulation to make it slightly less astronomically slow.
    # Even with 5x5 patches, it's 25 pixels -> 5 qubits per patch -> 11 qubits for SWAP test. Manageable for simulation, but repeated millions of times.
    ```

This implementation provides a concrete (though simulated and impractical) example of how quantum algorithms like the SWAP test could be integrated into a classical image processing pipeline for a task like NLM. It highlights the necessary steps: state encoding, circuit construction, simulation, and result interpretation, while also underscoring the significant performance challenges.
