import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from PIL import Image
import argparse
import os
import requests
from io import BytesIO
import math
import time
import random  # Needed for add_rain

print(f"OpenCV version: {cv2.__version__}")

# Check for Guided Filter
GUIDED_FILTER_AVAILABLE = False
try:
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        GUIDED_FILTER_AVAILABLE = True
        print("Found cv2.ximgproc.guidedFilter.")
    else:
        print("Warning: cv2.ximgproc.guidedFilter not found.")
        print("         Will use Gaussian Blur fallback (less effective).")
except Exception as e:
    print(f"Warning: Error checking for OpenCV contrib module ({e}).")

# --- First Stage: Rain Removal via Edge Detection and Inpainting ---

def compute_blob_orientation(contour):
    """
    Compute the orientation of a contour using ellipse fitting.
    Returns angle in degrees (0-180) if possible.
    """
    if len(contour) >= 5:  # cv2.fitEllipse requires at least 5 points
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]  # angle of the ellipse in degrees
        return angle
    return None

def build_orientation_histogram(angles, weights, bins=18):
    """
    Build a weighted histogram of angles.
    bins: number of bins (default 18 gives bins of 10° each over 0°-180°).
    """
    hist, bin_edges = np.histogram(angles, bins=bins, range=(0, 180), weights=weights)
    return hist, bin_edges

def detect_rain_candidates(image):
    """
    Detect candidate rain streak regions in a single image.
    Returns:
      - a list of blob orientations (in degrees)
      - corresponding weights (using area of blob)
      - a binary rain mask
      - the edge image (for visualization)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detector to find high-frequency changes (potential streaks)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Dilate edges slightly to merge nearby edge pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours from the dilated edge image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    orientations = []
    weights = []
    rain_mask = np.zeros_like(gray, dtype=np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out noise: ignore very small or very large blobs
        if area > 500:
            continue
        angle = compute_blob_orientation(cnt)
        if angle is not None:
            orientations.append(angle)
            weights.append(area)
            # Draw the blob on the rain mask
            cv2.drawContours(rain_mask, [cnt], -1, 255, -1)
    
    return orientations, weights, rain_mask, edges

def remove_rain_using_inpainting(image, rain_mask):
    """
    Removes rain streaks by inpainting the regions marked in the rain_mask.
    """
    # Inpaint the image using the Telea algorithm (radius can be tuned)
    inpainted = cv2.inpaint(image, rain_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

def first_stage_derain(image):
    """
    Run the first-stage deraining algorithm (edge-based detection and inpainting).
    Returns the derained image.
    """
    angles, weights, rain_mask, edge_img = detect_rain_candidates(image)
    
    if angles:
        hist, bin_edges = build_orientation_histogram(angles, weights, bins=18)
        plt.figure(figsize=(8, 4))
        plt.bar(bin_edges[:-1], hist, width=10, align='edge', edgecolor='black')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Weighted count (area)')
        plt.title('Histogram of Orientation of Detected Streaks (HOS)')
        plt.show()
        
        vertical_candidates = [angle for angle in angles if 80 <= angle <= 100]
        if len(vertical_candidates) > 0.5 * len(angles):
            print("Rain detected based on orientation histogram (first stage).")
        else:
            print("No dominant rain orientation detected (first stage).")
    else:
        print("No candidate rain streaks detected (first stage).")
    
    # Remove rain by inpainting using the rain mask
    result = remove_rain_using_inpainting(image, rain_mask)
    
    # Optionally, display intermediate images (can be commented out if not needed)
    print("First Stage - Original Image:")
    cv2_imshow(image)
    print("First Stage - Edge Image:")
    cv2_imshow(edge_img)
    print("First Stage - Rain Mask:")
    cv2_imshow(rain_mask)
    print("First Stage - Derained Result:")
    cv2_imshow(result)
    
    return result

# --- Second Stage: Deraining via Guided Filter + Non-Local Means (NLM) ---

def add_rain(image_arr, intensity=0.3, slant=-1, drop_length=20, thickness=1):
    """
    Add synthetic rain streak noise to the image.
    """
    img_h, img_w = image_arr.shape[:2]
    is_color = image_arr.ndim == 3
    noisy_image = image_arr.copy()
    num_drops = int(intensity * img_w * img_h / (drop_length + 5))
    for _ in range(num_drops):
        x1 = random.randint(0, img_w - 1)
        y1 = random.randint(0, img_h - 1)
        current_slant = slant + random.uniform(-0.5, 0.5)
        current_length = random.randint(max(5, drop_length - 10), drop_length + 10)
        x2 = int(np.clip(x1 + current_slant * current_length, 0, img_w - 1))
        y2 = int(np.clip(y1 + current_length, 0, img_h - 1))
        color_val = random.randint(180, 255)
        line_color = (color_val, color_val, color_val) if is_color else color_val
        current_thickness = random.randint(max(1, thickness-1), thickness+1)
        cv2.line(noisy_image, (x1, y1), (x2, y2), line_color, current_thickness)
    noisy_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
    return np.clip(noisy_image, 0, 255).astype(image_arr.dtype)

def load_image_from_url_or_path(url_or_path):
    """
    Load an image from a URL or local path and convert it to BGR format.
    """
    try:
        if url_or_path.startswith(('http://', 'https://')):
            response = requests.get(url_or_path, timeout=15)
            response.raise_for_status()
            img_pil = Image.open(BytesIO(response.content))
            print("Loaded image from URL.")
        else:
            if not os.path.exists(url_or_path):
                raise FileNotFoundError(f"Image file not found at: {url_or_path}")
            img_pil = Image.open(url_or_path)
            print(f"Loaded image from path: {url_or_path}")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        if img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2BGR)
        return img_bgr
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def derain_guided_nlm(image_arr_noisy,
                      gf_radius=15, gf_eps=0.1**2,
                      nlm_h_detail=8.0, nlm_win_detail=7, nlm_search_detail=21,
                      nlm_h_base=10.0, nlm_win_base=7, nlm_search_base=21,
                      apply_nlm_base=True):
    """
    Removes rain using Guided Filter decomposition followed by Non-Local Means
    applied separately to the base and detail layers.
    Input should be uint8 BGR or Grayscale. Output is uint8.
    All NLM window sizes must be odd.
    """
    print("Applying classical deraining using Guided Filter + NLM on layers...")
    start_time = time.time()
    img_uint8 = image_arr_noisy.astype(np.uint8)
    is_color = img_uint8.ndim == 3

    # --- Work in float32 [0,1] range for intermediate steps ---
    img_float = img_uint8.astype(np.float32) / 255.0

    # --- Create Guide Image (Grayscale) ---
    if is_color:
        guide_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        guide_img = img_float

    # --- Step 1 & 2: Calculate Base Layer (B) using Guided Filter ---
    print(f"  Step 1/2: Guided Filter (Radius={gf_radius}, Eps={gf_eps:.4f})...")
    if GUIDED_FILTER_AVAILABLE:
        base_layer = cv2.ximgproc.guidedFilter(guide=guide_img, src=img_float,
                                                 radius=gf_radius, eps=gf_eps,
                                                 dDepth=cv2.CV_32F)
    else:
        print("  Using Gaussian Blur fallback for base layer.")
        blur_ksize = max(3, gf_radius // 2 * 2 + 1)
        base_layer = cv2.GaussianBlur(img_float, (blur_ksize, blur_ksize), 0)
    base_layer = np.clip(base_layer, 0.0, 1.0).astype(np.float32)
    gf_time = time.time()
    print(f"    Guided Filter took {gf_time - start_time:.2f}s")

    # --- Step 3: Calculate Detail Layer (D) ---
    detail_layer = img_float - base_layer

    # --- Step 4a: Apply NLM to Detail Layer (D_denoised) ---
    print(f"  Step 4a: NLM on Detail Layer (h={nlm_h_detail}, win={nlm_win_detail}, search={nlm_search_detail})...")
    detail_min = np.min(detail_layer)
    detail_max = np.max(detail_layer)
    detail_range = detail_max - detail_min
    if detail_range > 1e-6:
        detail_uint8 = np.clip(((detail_layer - detail_min) / detail_range) * 255.0, 0, 255).astype(np.uint8)
        if is_color:
            detail_denoised_uint8 = cv2.fastNlMeansDenoisingColored(
                detail_uint8,
                None,
                h=float(nlm_h_detail),
                hColor=float(nlm_h_detail),
                templateWindowSize=nlm_win_detail,
                searchWindowSize=nlm_search_detail
            )
        else:
            detail_denoised_uint8 = cv2.fastNlMeansDenoising(
                detail_uint8,
                None,
                h=float(nlm_h_detail),
                templateWindowSize=nlm_win_detail,
                searchWindowSize=nlm_search_detail
            )
        detail_denoised = ((detail_denoised_uint8.astype(np.float32) / 255.0) * detail_range) + detail_min
    else:
        print("    Detail layer range is near zero, skipping NLM.")
        detail_denoised = detail_layer
    nlm_detail_time = time.time()
    print(f"    NLM on Detail took {nlm_detail_time - gf_time:.2f}s")

    # --- Step 4b: Apply NLM to Base Layer (B_denoised) ---
    base_denoised = base_layer
    if apply_nlm_base:
        print(f"  Step 4b: NLM on Base Layer (h={nlm_h_base}, win={nlm_win_base}, search={nlm_search_base})...")
        base_uint8 = (base_layer * 255.0).astype(np.uint8)
        if is_color:
            base_denoised_uint8 = cv2.fastNlMeansDenoisingColored(
                base_uint8,
                None,
                h=float(nlm_h_base),
                hColor=float(nlm_h_base),
                templateWindowSize=nlm_win_base,
                searchWindowSize=nlm_search_base
            )
        else:
            base_denoised_uint8 = cv2.fastNlMeansDenoising(
                base_uint8,
                None,
                h=float(nlm_h_base),
                templateWindowSize=nlm_win_base,
                searchWindowSize=nlm_search_base
            )
        base_denoised = base_denoised_uint8.astype(np.float32) / 255.0
        nlm_base_time = time.time()
        print(f"    NLM on Base took {nlm_base_time - nlm_detail_time:.2f}s")
    else:
        print("  Step 4b: Skipping NLM on Base Layer.")

    # --- Step 5: Reconstruct Image ---
    denoised_float = base_denoised + detail_denoised
    denoised_clipped = np.clip(denoised_float, 0.0, 1.0)
    denoised_uint8 = (denoised_clipped * 255.0).astype(np.uint8)
    end_time = time.time()
    print(f"Combined deraining process completed in {end_time - start_time:.2f}s.")
    return denoised_uint8

# --- Combined Workflow ---

def run_combined_derain_process(image_path_or_url, output_path, args):
    """
    Loads the image, applies first-stage deraining (edge/inpainting) and then
    second-stage deraining (guided filter + NLM) on the output.
    Saves the final result.
    """
    # Load image
    image_arr = load_image_from_url_or_path(image_path_or_url)
    if image_arr is None:
        return
    original_height, original_width = image_arr.shape[:2]
    
    # Optional resizing
    if args.resize:
        aspect_ratio = original_height / original_width
        new_height = int(args.resize * aspect_ratio)
        image_arr = cv2.resize(image_arr, (args.resize, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Optional: Add synthetic rain noise
    temp_noisy_path = None
    if args.add_noise:
        print(f"Adding synthetic 'rain' noise with intensity {args.intensity}...")
        try:
            image_arr = add_rain(image_arr.astype(float), intensity=args.intensity).astype(np.uint8)
            temp_noisy_path = "temp_noisy_rain_combined.png"
            Image.fromarray(cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)).save(temp_noisy_path)
            print(f"Temporary noisy image saved to '{temp_noisy_path}'")
        except Exception as e:
            print(f"Error adding noise: {e}")
    
    # --- First Stage Deraining ---
    print("Starting First Stage Deraining (Edge Detection + Inpainting)...")
    first_stage_result = first_stage_derain(image_arr)
    
    # --- Second Stage Deraining ---
    print("Starting Second Stage Deraining (Guided Filter + NLM)...")
    try:
        second_stage_result = derain_guided_nlm(
            first_stage_result,
            gf_radius=args.gf_r,
            gf_eps=args.gf_e**2,
            nlm_h_detail=args.nlm_h_d,
            nlm_win_detail=args.nlm_w_d,
            nlm_search_detail=args.nlm_s_d,
            nlm_h_base=args.nlm_h_b,
            nlm_win_base=args.nlm_w_b,
            nlm_search_base=args.nlm_s_b,
            apply_nlm_base=(not args.no_nlm_base)
        )
    except Exception as e:
        print(f"Error during second stage deraining process: {e}")
        import traceback
        traceback.print_exc()
        return

    # Optional: Resize final image back to original dimensions
    if args.resize and (second_stage_result.shape[1] != original_width or second_stage_result.shape[0] != original_height):
        second_stage_result = cv2.resize(second_stage_result, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)

    # Save final output
    try:
        denoised_img_rgb = cv2.cvtColor(second_stage_result, cv2.COLOR_BGR2RGB)
        denoised_img = Image.fromarray(denoised_img_rgb)
        denoised_img.save(output_path)
        print(f"Successfully processed image saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving processed image: {e}")

    # Clean up temporary files if any
    if temp_noisy_path and os.path.exists(temp_noisy_path):
        try:
            os.remove(temp_noisy_path)
        except OSError as e:
            print(f"Error removing temp file: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Deraining Pipeline: Edge-based Inpainting followed by Guided Filter + NLM")
    # Input image path or URL
    parser.add_argument("input_image", help="Path or URL to the input image file.")
    parser.add_argument("-o", "--output_image", default="derained_combined.png", help="Path to save the processed image.")
    parser.add_argument("--add_noise", action='store_true', help="Add synthetic rain noise before processing.")
    parser.add_argument("--intensity", type=float, default=0.4, help="Intensity for added synthetic rain noise (0 to 1).")
    parser.add_argument("--resize", type=int, default=None, help="Optional width to resize image to before processing.")

    # Guided Filter Parameters
    parser.add_argument("--gf_r", type=int, default=15, help="Guided Filter radius.")
    parser.add_argument("--gf_e", type=float, default=0.1, help="Guided Filter epsilon (sqrt(eps) value).")

    # NLM Detail Parameters
    parser.add_argument("--nlm_h_d", type=float, default=8.0, help="NLM strength (h) for Detail Layer.")
    parser.add_argument("--nlm_w_d", type=int, default=7, help="NLM template window size for Detail Layer (odd).")
    parser.add_argument("--nlm_s_d", type=int, default=21, help="NLM search window size for Detail Layer (odd).")

    # NLM Base Parameters
    parser.add_argument("--no_nlm_base", action='store_true', help="Skip applying NLM to the Base Layer.")
    parser.add_argument("--nlm_h_b", type=float, default=10.0, help="NLM strength (h) for Base Layer.")
    parser.add_argument("--nlm_w_b", type=int, default=7, help="NLM template window size for Base Layer (odd).")
    parser.add_argument("--nlm_s_b", type=int, default=21, help="NLM search window size for Base Layer (odd).")

    args = parser.parse_args()

    # Ensure NLM window sizes are odd
    args.nlm_w_d = args.nlm_w_d if args.nlm_w_d % 2 != 0 else args.nlm_w_d + 1
    args.nlm_s_d = args.nlm_s_d if args.nlm_s_d % 2 != 0 else args.nlm_s_d + 1
    args.nlm_w_b = args.nlm_w_b if args.nlm_w_b % 2 != 0 else args.nlm_w_b + 1
    args.nlm_s_b = args.nlm_s_b if args.nlm_s_b % 2 != 0 else args.nlm_s_b + 1

    run_combined_derain_process(args.input_image, args.output_image, args)
