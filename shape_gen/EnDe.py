import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt  # used internally; output is via Streamlit
from scipy.spatial import Delaunay
import random
from io import BytesIO
import math

# ----- Provided Functions with Modifications -----

def generate_max_random_circles(image_size=(512, 512), min_radius=50, max_radius=100, 
                                max_attempts=50000, max_fail_attempts=10000, max_circles_limit=10):
    img = np.zeros(image_size, dtype=np.uint8)
    circles = []

    def is_too_close(x, y, radius):
        for (cx, cy, cr) in circles:
            distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if distance < (cr + radius):
                return True
        return False

    attempts = 0
    failed_attempts = 0

    while (attempts < max_attempts and failed_attempts < max_fail_attempts and 
           len(circles) < max_circles_limit):
        used_space = sum([np.pi * cr**2 for (_, _, cr) in circles])
        total_space = image_size[0] * image_size[1]
        remaining_space = total_space - used_space

        remaining_capacity = remaining_space / total_space
        min_dynamic_radius = int(min_radius + (remaining_capacity * (max_radius - min_radius)))
        max_dynamic_radius = int(min_dynamic_radius * 1.5)
        max_dynamic_radius = min(max_dynamic_radius, max_radius)

        radius = random.randint(min_dynamic_radius, max_dynamic_radius)
        center_x = random.randint(radius, image_size[1] - radius)
        center_y = random.randint(radius, image_size[0] - radius)

        if not is_too_close(center_x, center_y, radius):
            cv2.circle(img, (center_x, center_y), radius, 255, 1)
            circles.append((center_x, center_y, radius))
            failed_attempts = 0
        else:
            failed_attempts += 1
        attempts += 1

    inverted_img = 255 - img
    return inverted_img, len(circles), circles

def resize_image_to_shape(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def compute_average_under_circles(input_image, circle_image, circles):
    output_image = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)
    resized_circle_image = cv2.resize(circle_image, (input_image.shape[1], input_image.shape[0]))
    for (cx, cy, radius) in circles:
        y, x = np.ogrid[:resized_circle_image.shape[0], :resized_circle_image.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        circle_pixels = input_image[mask]
        if len(circle_pixels) > 0:
            average_value = np.mean(circle_pixels, axis=0)
        else:
            average_value = [0, 0, 0]
        output_image[mask] = np.append(average_value, [255])
    return output_image

def overlay_mask_on_image(input_image, mask_image):
    output_image = input_image.copy()
    mask_alpha = mask_image[:, :, 3] / 255.0
    for c in range(3):
        output_image[:, :, c] = (1 - mask_alpha) * input_image[:, :, c] + mask_alpha * mask_image[:, :, c]
    return output_image

# ---------------
# NEW HELPER FUNCTION FROM tri.py
# ---------------
def generate_random_triangle(image_shape, min_size, max_size):
    """
    Generates a triangle with a random size between min_size and max_size,
    a random rotation, and a slight vertex distortion.
    The triangle is generated within image_shape dimensions.
    """
    h, w = image_shape
    # Pick a random size between min_size and max_size
    size = random.randint(min_size, max_size)
    # Use a margin to ensure the triangle is fully inside the padded image
    margin = max_size
    center_x = random.randint(margin, w - margin)
    center_y = random.randint(margin, h - margin)
    center = np.array([center_x, center_y])
    
    # Create a base equilateral triangle (centered at (0,0))
    base_triangle = np.array([
        [0, -size / math.sqrt(3)],
        [-size / 2, size / (2 * math.sqrt(3))],
        [size / 2, size / (2 * math.sqrt(3))]
    ])
    
    # Apply a slight random distortion to each vertex to avoid perfect symmetry
    distortion = np.random.uniform(0.9, 1.1, base_triangle.shape)
    base_triangle = base_triangle * distortion
    
    # Rotate the triangle by a random angle between 0 and 2pi
    angle = random.uniform(0, 2 * math.pi)
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])
    rotated_triangle = base_triangle.dot(rotation_matrix.T)
    
    # Shift the triangle to the chosen center
    triangle = rotated_triangle + center
    return triangle.astype(np.int32)

# ---------------
# Modified encode function (unchanged from your current version)
# ---------------
def encode(input_image, shape_type, output_path, **kwargs):
    shape_type = shape_type.lower()
    if shape_type in ['triangle', 'triangles']:
        image_resized = cv2.resize(input_image, (500, 500))
        num_triangles = kwargs.get('num_triangles', kwargs.get('num_shapes', 50))
        max_triangle_size = kwargs.get('max_size', 100)
        min_triangle_size = kwargs.get('min_size', max_triangle_size // 2)
        
        padding = max_triangle_size
        image_padded = cv2.copyMakeBorder(image_resized, padding, padding, padding, padding, cv2.BORDER_REFLECT)
        
        global_mask = np.zeros(image_padded.shape[:2], dtype=np.uint8)
        triangles = []
        attempts = 0
        max_attempts = num_triangles * 100

        # First pass: fixed max-sized triangles
        while len(triangles) < num_triangles and attempts < max_attempts:
            candidate = generate_random_triangle(image_padded.shape[:2], max_triangle_size, max_triangle_size)
            candidate_mask = np.zeros(image_padded.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(candidate_mask, candidate, 255)
            overlap = cv2.bitwise_and(global_mask, candidate_mask)
            if cv2.countNonZero(overlap) == 0:
                triangles.append(candidate)
                global_mask = cv2.bitwise_or(global_mask, candidate_mask)
            attempts += 1

        # Second pass: fill remaining spaces with random-sized triangles between min and max
        attempts = 0
        while len(triangles) < num_triangles and attempts < max_attempts:
            candidate = generate_random_triangle(image_padded.shape[:2], min_triangle_size, max_triangle_size)
            candidate_mask = np.zeros(image_padded.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(candidate_mask, candidate, 255)
            overlap = cv2.bitwise_and(global_mask, candidate_mask)
            if cv2.countNonZero(overlap) == 0:
                triangles.append(candidate)
                global_mask = cv2.bitwise_or(global_mask, candidate_mask)
            attempts += 1

        if len(triangles) < num_triangles:
            st.warning(f"Only generated {len(triangles)} non-overlapping triangles out of {num_triangles} requested.")

        overlay_padded = image_padded.copy()
        for tri in triangles:
            mask = np.zeros(image_padded.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, tri, 255)
            avg_color = cv2.mean(image_padded, mask=mask)[:3]
            avg_color = tuple(map(int, avg_color))
            cv2.fillConvexPoly(overlay_padded, tri, avg_color)

        overlay_cropped = overlay_padded[padding:padding+image_resized.shape[0], padding:padding+image_resized.shape[1]]
        original_resized = image_resized.copy()
        
        triangles_cropped = []
        for tri in triangles:
            tri_cropped = tri - np.array([padding, padding])
            tri_cropped[:, 0] = np.clip(tri_cropped[:, 0], 0, image_resized.shape[1]-1)
            tri_cropped[:, 1] = np.clip(tri_cropped[:, 1], 0, image_resized.shape[0]-1)
            triangles_cropped.append(tri_cropped)
        
        boundaries = triangles_cropped
        encoded_image = overlay_cropped.copy()
    else:
        st.error("Only the triangle shape is supported in this demo.")
        return None, None

    # -----------------------------
    # Encode the boundaries into the blue channel's LSB
    # -----------------------------
    encode_mask = np.zeros(original_resized.shape[:2], dtype=np.uint8)
    for tri in boundaries:
        cv2.polylines(encode_mask, [tri], isClosed=True, color=255, thickness=1)
    
    # Combine the overlay with the original image based on the boundary mask.
    final_encoded = encoded_image.copy()
    for i in range(final_encoded.shape[0]):
        for j in range(final_encoded.shape[1]):
            if encode_mask[i, j] == 255:
                final_encoded[i, j] = original_resized[i, j]
                final_encoded[i, j, 0] = (final_encoded[i, j, 0] & 254) | 1
            else:
                final_encoded[i, j, 0] = final_encoded[i, j, 0] & 254

    # -----------------------------
    # Add corner markers for validation (3x3 blocks in each corner)
    # -----------------------------
    corner_size = 3
    h_img, w_img, _ = final_encoded.shape
    corner_positions = {
        "top_left": (0, 0),
        "top_right": (0, w_img - corner_size),
        "bottom_left": (h_img - corner_size, 0),
        "bottom_right": (h_img - corner_size, w_img - corner_size)
    }
    expected_patterns = {
        "top_left": (1, 1, 1),
        "top_right": (0, 0, 1),
        "bottom_left": (0, 1, 0),
        "bottom_right": (1, 0, 0)
    }
    for corner, (y, x) in corner_positions.items():
        exp_b, exp_g, exp_r = expected_patterns[corner]
        for i in range(y, y + corner_size):
            for j in range(x, x + corner_size):
                final_encoded[i, j, 0] = (final_encoded[i, j, 0] & 254) | exp_b
                final_encoded[i, j, 1] = (final_encoded[i, j, 1] & 254) | exp_g
                final_encoded[i, j, 2] = (final_encoded[i, j, 2] & 254) | exp_r

    return final_encoded, boundaries

# -----------------------------
# Modified decode function with updated triangle detection and size filtering
# -----------------------------
def decode(encoded_image, shape_type, boundaries=None, **kwargs):
    shape_type = shape_type.lower()
    if encoded_image is None:
        st.error("Error: Encoded image is None.")
        return None, None, None
    h, w, _ = encoded_image.shape
    blue_lsb = encoded_image[:, :, 0] & 1
    corner_size = 3
    corner_positions = {
        "top_left": (0, 0),
        "top_right": (0, w - corner_size),
        "bottom_left": (h - corner_size, 0),
        "bottom_right": (h - corner_size, w - corner_size)
    }
    expected_patterns = {
        "top_left": (1, 1, 1),
        "top_right": (0, 0, 1),
        "bottom_left": (0, 1, 0),
        "bottom_right": (1, 0, 0)
    }
    threshold = 6
    valid = True
    for corner, (y, x) in corner_positions.items():
        exp_b, exp_g, exp_r = expected_patterns[corner]
        count_b = np.sum(blue_lsb[y:y+corner_size, x:x+corner_size] == exp_b)
        if count_b < threshold:
            valid = False
            st.warning(f"Corner '{corner}' failed validation.")
            break
    if valid:
        st.info("Valid encoding detected. Decoding boundaries.")
        binary_image = (blue_lsb * 255).astype(np.uint8)
    else:
        st.warning("No valid encoding found. Returning black binary image.")
        binary_image = np.zeros_like(blue_lsb, dtype=np.uint8)
    if cv2.countNonZero(binary_image) < 50:
        st.info("Binary image nearly empty; applying dilation.")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    rgb_values = []
    annotated = encoded_image.copy()
    if shape_type in ['triangle', 'triangles']:
        # If no boundaries provided, detect triangles using thresholded contours.
        if boundaries is None:
            ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundaries = []
            min_size = kwargs.get('min_size', None)
            max_size = kwargs.get('max_size', None)
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # Adjusted factor from 0.02 to 0.04
                if len(approx) == 3:
                    tri = approx.reshape(-1, 2)
                    xs = tri[:, 0]
                    ys = tri[:, 1]
                    width = xs.max() - xs.min()
                    height = ys.max() - ys.min()
                    if min_size is not None and (width < min_size or height < min_size):
                        continue
                    if max_size is not None and (width > max_size or height > max_size):
                        continue
                    boundaries.append(tri)
        for tri in boundaries:
            pts = np.int32(tri)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            center = np.mean(tri, axis=0)
            center_x = int(np.clip(center[0], 0, w - 1))
            center_y = int(np.clip(center[1], 0, h - 1))
            b, g, r = encoded_image[center_y, center_x]
            rgb_values.append([r, g, b])
    elif shape_type in ['rectangle', 'rectangles']:
        ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_size = kwargs.get('min_size', None)
        max_size = kwargs.get('max_size', None)
        for cnt in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            if w_rect > 1 and h_rect > 1:
                if min_size is not None and (w_rect < min_size or h_rect < min_size):
                    continue
                if max_size is not None and (w_rect > max_size or h_rect > max_size):
                    continue
                cv2.rectangle(annotated, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 1)
                center_x = x + w_rect // 2
                center_y = y + h_rect // 2
                b, g, r = encoded_image[center_y, center_x]
                rgb_values.append([r, g, b])
    elif shape_type in ['circle', 'circles']:
        ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_size = kwargs.get('min_size', None)
        max_size = kwargs.get('max_size', None)
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if min_size is not None and (radius < min_size):
                continue
            if max_size is not None and (radius > max_size):
                continue
            if radius > 3 and radius < 250:
                cv2.circle(annotated, center, radius, (0, 255, 0), 1)
                b, g, r = encoded_image[center[1], center[0]]
                rgb_values.append([r, g, b])
    else:
        st.error("Unsupported shape type for decoding.")
    
    return binary_image, annotated, rgb_values
