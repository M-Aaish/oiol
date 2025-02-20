import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from EnDe import decode, encode
from painterfun import oil_main  # Importing the oil_main function
import mixbox
import itertools
import math
import os

# Set page config for the merged app.
st.set_page_config(page_title="Merged App", layout="wide")

# --------------------------------------------------------------------
# --- Functions from the original str_main.py (Image Generator, Shape Detector,
#     Oil Painting Generator, Colour Merger)
# --------------------------------------------------------------------

# Function to calculate the Euclidean distance between two RGB colors
def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

# Function to group similar colors and count them
def group_similar_colors(rgb_vals, threshold=10):
    grouped_colors = []  # List to store final groups of similar colors
    counts = []  # List to store counts of similar colors

    for color in rgb_vals:
        found_group = False
        for i, group in enumerate(grouped_colors):
            if color_distance(color, group[0]) < threshold:
                grouped_colors[i].append(color)
                counts[i] += 1
                found_group = True
                break
        if not found_group:
            grouped_colors.append([color])
            counts.append(1)
    return [(group[0], count) for group, count in zip(grouped_colors, counts)]

def oil_painting_page():
    st.title("Oil Painting Image Generator")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    intensity = st.number_input("Enter the intensity (integer):", min_value=1, max_value=100, value=10)
    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.write("Upload an image to see it here")
    if st.button("Generate"):
        if uploaded_file is not None:
            input_image_cv = np.array(input_image)
            if len(input_image_cv.shape) == 2:
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_GRAY2RGB)
            elif input_image_cv.shape[2] == 4:
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_RGBA2RGB)
            output_image_cv = oil_main(input_image_cv, intensity)
            output_image_cv = (output_image_cv * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image_cv)
            with col2:
                st.image(output_image, caption="Processed Image", use_column_width=True)
            img_byte_arr = BytesIO()
            output_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            st.download_button(
                label="Download Processed Image",
                data=img_byte_arr,
                file_name="processed_image.png",
                mime="image/png"
            )

def color_mixing_app():
    st.title("RGB Color Mixing")
    if 'colors' not in st.session_state:
        st.session_state.colors = [
            {"rgb": [255, 0, 0], "weight": 0.3},  # Default color 1 (Red)
            {"rgb": [0, 255, 0], "weight": 0.6}   # Default color 2 (Green)
        ]

    def rgb_to_latent(rgb):
        return mixbox.rgb_to_latent(rgb)

    def latent_to_rgb(latent):
        return mixbox.latent_to_rgb(latent)

    def get_mixed_rgb(colors):
        z_mix = [0] * mixbox.LATENT_SIZE
        total_weight = sum(c["weight"] for c in colors)
        for i in range(len(z_mix)):
            z_mix[i] = sum(c["weight"] * rgb_to_latent(c["rgb"])[i] for c in colors) / total_weight
        return latent_to_rgb(z_mix)

    def add_new_color():
        st.session_state.colors.append({"rgb": [255, 255, 255], "weight": 0.1})

    def delete_color(index):
        st.session_state.colors.pop(index)

    for idx, color in enumerate(st.session_state.colors):
        with st.expander(f"Color {idx + 1}"):
            r = st.number_input(f"Red value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][0])
            g = st.number_input(f"Green value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][1])
            b = st.number_input(f"Blue value for Color {idx + 1}", min_value=0, max_value=255, value=color['rgb'][2])
            col_weight = st.slider(f"Weight for Color {idx + 1}", 0.0, 1.0, value=color["weight"], step=0.05)
            color["rgb"] = [r, g, b]
            color["weight"] = col_weight
            st.markdown(f"<div style='width: 100px; height: 100px; background-color: rgb({r}, {g}, {b});'></div>", unsafe_allow_html=True)
            if len(st.session_state.colors) > 2:
                delete_button = st.button(f"Delete Color {idx + 1}", key=f"delete_button_{idx}")
                if delete_button:
                    delete_color(idx)
                    st.session_state.colors = st.session_state.colors
    if st.button("Add Color"):
        add_new_color()
        st.session_state.colors = st.session_state.colors

    mixed_rgb = get_mixed_rgb(st.session_state.colors)
    st.subheader("Mixed Color")
    st.write(f"RGB: {mixed_rgb}")
    st.markdown(f"<div style='width: 200px; height: 200px; background-color: rgb{mixed_rgb};'></div>", unsafe_allow_html=True)

def image_generator_app():
    st.header("Image Generator")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])
    num_shapes = st.number_input("Enter the number of shapes to encode:", min_value=1, value=10)
    shape_size = st.number_input("Enter the size of the shape:", min_value=1, value=10)
    col1, col2 = st.columns([1, 1])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error reading the image. Please try another file.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with col1:
                st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate"):
        if uploaded_file is not None:
            shape = shape_option
            encoded_image, boundaries = encode(img, shape, output_path="",
                                               num_shapes=num_shapes, shape_size=shape_size)
            encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(encoded_image_rgb, caption=f"Encoded {shape_option} Image", use_container_width=True)
            is_success, buffer = cv2.imencode(".png", encoded_image)
            if is_success:
                st.download_button(
                    label="Download Encoded Image",
                    data=buffer.tobytes(),
                    file_name="encoded_image.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload an image first.")

def shape_detector_app():
    st.header("Shape Detector")
    uploaded_file = st.file_uploader("Upload an Encoded Image", type=["jpg", "jpeg", "png"])
    shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])
    col1, col2 = st.columns([1, 1])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        encoded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if encoded_image is None:
            st.error("Error reading the image. Please try another file.")
        else:
            encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
            with col1:
                st.image(encoded_image_rgb, caption="Uploaded Encoded Image", use_container_width=True)
    if st.button("Decode"):
        if uploaded_file is not None:
            shape = shape_option
            binary_img, annotated_img, rgb_vals = decode(encoded_image, shape, boundaries=None)
            grouped_colors = group_similar_colors(rgb_vals, threshold=10)
            grouped_colors = sorted(grouped_colors, key=lambda x: x[1], reverse=True)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            with col2:
                st.image(annotated_img_rgb, caption=f"Decoded Annotated {shape_option} Image", use_container_width=True)
            st.subheader("Grouped Colors (Ranked by Count)")
            col1c, col2c, col3c = st.columns(3)
            for idx, (color, count) in enumerate(grouped_colors):
                rgb_str = f"RGB: {color} - Count: {count}"
                color_box = f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); height: 30px; width: 30px; margin-right: 10px; display: inline-block;"
                if idx % 3 == 0:
                    with col1c:
                        st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                elif idx % 3 == 1:
                    with col2c:
                        st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                else:
                    with col3c:
                        st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
            is_success, buffer = cv2.imencode(".png", annotated_img)
            if is_success:
                st.download_button(
                    label="Download Decoded Image",
                    data=buffer.tobytes(),
                    file_name="decoded_image.png",
                    mime="image/png"
                )
        else:
            st.warning("Please upload an image first.")

# --------------------------------------------------------------------
# --- Functions from painter2.py (Painter App - Recipe Generator and Colors DataBase)
# --------------------------------------------------------------------

# File name for our color database.
COLOR_DB_FILE = "shape_gen/color.txt"

@st.cache_data
def read_color_file(filename=COLOR_DB_FILE):
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        st.error("Error reading color.txt: " + str(e))
        return ""

def parse_color_db(txt):
    databases = {}
    current_db = None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            current_db = line
            databases[current_db] = []
        else:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            rgb_str = tokens[-2]
            color_name = " ".join(tokens[1:-2])
            try:
                r, g, b = [int(x) for x in rgb_str.split(",")]
            except Exception:
                continue
            databases[current_db].append((color_name, (r, g, b)))
    return databases

color_txt = read_color_file()
databases = parse_color_db(color_txt)

def convert_db_list_to_dict(color_list):
    d = {}
    for name, rgb in color_list:
        d[name] = {"rgb": list(rgb)}
    return d

def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

def mix_colors(recipe):
    total, r_total, g_total, b_total = 0, 0, 0, 0
    for color, perc in recipe:
        r, g, b = color
        r_total += r * perc
        g_total += g * perc
        b_total += b * perc
        total += perc
    if total == 0:
        return (0, 0, 0)
    return (round(r_total / total), round(g_total / total), round(b_total / total))

def color_error(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def generate_recipes(target, base_colors_dict, step=10.0):
    candidates = []
    base_list = [(name, info["rgb"]) for name, info in base_colors_dict.items()]
    for name, rgb in base_list:
        err = color_error(tuple(rgb), target)
        if err < 5:
            recipe = [(name, 100.0)]
            candidates.append((recipe, tuple(rgb), err))
    for (name1, rgb1), (name2, rgb2), (name3, rgb3) in itertools.combinations(base_list, 3):
        for p1 in np.arange(0, 100 + step, step):
            for p2 in np.arange(0, 100 - p1 + step, step):
                p3 = 100 - p1 - p2
                if p3 < 0:
                    continue
                recipe = [(name1, p1), (name2, p2), (name3, p3)]
                mix_recipe = [(rgb1, p1), (rgb2, p2), (rgb3, p3)]
                mixed = mix_colors(mix_recipe)
                err = color_error(mixed, target)
                candidates.append((recipe, mixed, err))
    candidates.sort(key=lambda x: x[2])
    top = []
    seen = set()
    for rec, mixed, err in candidates:
        key = tuple(sorted((name, perc) for name, perc in rec if perc > 0))
        if key not in seen:
            seen.add(key)
            top.append((rec, mixed, err))
        if len(top) >= 3:
            break
    return top

def display_color_block(color, label=""):
    hex_color = rgb_to_hex(*color)
    st.markdown(
        f"<div style='background-color: {hex_color}; width:100px; height:100px; border:1px solid #000; text-align: center; line-height: 100px;'>{label}</div>",
        unsafe_allow_html=True,
    )

def display_thin_color_block(color):
    hex_color = rgb_to_hex(*color)
    st.markdown(
        f"<div style='background-color: {hex_color}; width:50px; height:20px; border:1px solid #000; display:inline-block; margin-right:10px;'></div>",
        unsafe_allow_html=True,
    )

def add_color_to_db(selected_db, color_name, r, g, b):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for update: " + str(e))
        return False

    new_lines = []
    in_section = False
    inserted = False
    last_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if in_section and not inserted:
                new_lines.append(f"{last_index+1} {color_name} {r},{g},{b} 0\n")
                inserted = True
            new_lines.append(line)
            if stripped == selected_db:
                in_section = True
            else:
                in_section = False
            continue
        if in_section:
            tokens = stripped.split()
            if tokens[0].isdigit():
                try:
                    idx = int(tokens[0])
                    last_index = max(last_index, idx)
                except:
                    pass
        new_lines.append(line)
    if in_section and not inserted:
        new_lines.append(f"{last_index+1} {color_name} {r},{g},{b} 0\n")
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def remove_color_from_db(selected_db, color_name):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for removal: " + str(e))
        return False

    new_lines = []
    in_section = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if stripped == selected_db:
                in_section = True
            else:
                in_section = False
            new_lines.append(line)
            continue
        if in_section and not removed:
            tokens = stripped.split()
            current_name = " ".join(tokens[1:-2]).strip()
            if current_name.lower() == color_name.lower():
                removed = True
                continue
        new_lines.append(line)
    if not removed:
        st.warning("Color not found in the selected database.")
        return False
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def create_custom_database(new_db_name):
    line = f"\n{new_db_name}\n"
    try:
        with open(COLOR_DB_FILE, "a") as f:
            f.write(line)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def remove_database(db_name):
    try:
        with open(COLOR_DB_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        st.error("Error reading file for removal: " + str(e))
        return False

    new_lines = []
    in_target = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if not stripped[0].isdigit():
            if stripped == db_name:
                in_target = True
                removed = True
                continue
            else:
                in_target = False
                new_lines.append(line)
        else:
            if in_target:
                continue
            else:
                new_lines.append(line)
    if not removed:
        st.warning("Database not found.")
        return False
    try:
        with open(COLOR_DB_FILE, "w") as f:
            f.writelines(new_lines)
        read_color_file.clear()
        return True
    except Exception as e:
        st.error("Error writing to file: " + str(e))
        return False

def show_databases_page():
    st.title("Color Database - Data Bases")
    selected_db = st.selectbox("Select a color database:", list(databases.keys()))
    st.write(f"### Colors in database: {selected_db}")
    for name, rgb in databases[selected_db]:
        st.write(f"**{name}**: {rgb_to_hex(*rgb)} ({rgb[0]},{rgb[1]},{rgb[2]})")
        display_thin_color_block(rgb)

def show_add_colors_page():
    global databases
    st.title("Colors DataBase - Add Colors")
    selected_db = st.selectbox("Select database to add a new color:", list(databases.keys()))
    with st.form("add_color_form"):
        new_color_name = st.text_input("New Color Name")
        r = st.number_input("Red", min_value=0, max_value=255, value=255)
        g = st.number_input("Green", min_value=0, max_value=255, value=255)
        b = st.number_input("Blue", min_value=0, max_value=255, value=255)
        submitted = st.form_submit_button("Add Color")
        if submitted:
            if new_color_name:
                success = add_color_to_db(selected_db, new_color_name, int(r), int(g), int(b))
                if success:
                    st.success(f"Color '{new_color_name}' added to {selected_db}!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to add color.")
            else:
                st.error("Please enter a color name.")

def show_remove_colors_page():
    global databases
    st.title("Colors DataBase - Remove Colors")
    selected_db = st.selectbox("Select database to remove a color from:", list(databases.keys()))
    with st.form("remove_color_form"):
        color_name = st.text_input("Color Name to Remove")
        submitted = st.form_submit_button("Remove Color")
        if submitted:
            if color_name:
                success = remove_color_from_db(selected_db, color_name)
                if success:
                    st.success(f"Color '{color_name}' removed from {selected_db}!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to remove color or color not found.")
            else:
                st.error("Please enter a color name.")

def show_create_custom_db_page():
    global databases
    st.title("Colors DataBase - Create Custom Data Base")
    with st.form("create_db_form"):
        new_db_name = st.text_input("Enter new database name:")
        submitted = st.form_submit_button("Create Database")
        if submitted:
            if new_db_name:
                success = create_custom_database(new_db_name)
                if success:
                    st.success(f"Database '{new_db_name}' created!")
                    color_txt = read_color_file(COLOR_DB_FILE)
                    databases = parse_color_db(color_txt)
                else:
                    st.error("Failed to create database.")
            else:
                st.error("Please enter a database name.")

def show_remove_database_page():
    global databases
    st.title("Colors DataBase - Remove Database")
    with st.form("remove_db_form"):
        db_name = st.text_input("Enter the name of the database to remove:")
        submitted = st.form_submit_button("Remove Database")
        if submitted:
            if db_name:
                confirm = st.checkbox("I confirm that I want to permanently delete this database.")
                if confirm:
                    success = remove_database(db_name)
                    if success:
                        st.success(f"Database '{db_name}' removed!")
                        color_txt = read_color_file(COLOR_DB_FILE)
                        databases = parse_color_db(color_txt)
                    else:
                        st.error("Failed to remove database.")
                else:
                    st.warning("Please check the confirmation box to proceed.")
            else:
                st.error("Please enter a database name.")

def painter_recipe_generator():
    st.title("Painter App - Recipe Generator")
    st.write("Enter your desired paint color to generate paint recipes using base colors.")
    db_choice = st.selectbox("Select a color database:", list(databases.keys()))
    selected_db_dict = convert_db_list_to_dict(databases[db_choice])
    method = st.radio("Select input method:", ["Color Picker", "RGB Sliders"])
    if method == "Color Picker":
        desired_hex = st.color_picker("Pick a color", "#ffffff")
        desired_rgb = tuple(int(desired_hex[i:i+2], 16) for i in (1, 3, 5))
    else:
        st.write("Select RGB values manually:")
        r = st.slider("Red", 0, 255, 255)
        g = st.slider("Green", 0, 255, 255)
        b = st.slider("Blue", 0, 255, 255)
        desired_rgb = (r, g, b)
        desired_hex = rgb_to_hex(r, g, b)
    st.write("**Desired Color:**", desired_hex)
    display_color_block(desired_rgb, label="Desired")
    step = st.slider("Select percentage step for recipe generation:", 4.0, 10.0, 10.0, step=0.5)
    if st.button("Generate Recipes"):
        recipes = generate_recipes(desired_rgb, selected_db_dict, step=step)
        if recipes:
            st.write("### Top 3 Paint Recipes")
            for idx, (recipe, mixed, err) in enumerate(recipes):
                st.write(f"**Recipe {idx+1}:** (Error = {err:.2f})")
                cols = st.columns(4)
                with cols[0]:
                    st.write("Desired:")
                    display_color_block(desired_rgb, label="Desired")
                with cols[1]:
                    st.write("Result:")
                    display_color_block(mixed, label="Mixed")
                with cols[2]:
                    st.write("Composition:")
                    for name, perc in recipe:
                        if perc > 0:
                            base_rgb = tuple(selected_db_dict[name]["rgb"])
                            st.write(f"- **{name}**: {perc:.1f}%")
                            display_color_block(base_rgb, label=name)
                with cols[3]:
                    st.write("Difference:")
                    st.write(f"RGB Distance: {err:.2f}")
        else:
            st.error("No recipes found.")

def painter_colors_database():
    st.title("Colors DataBase")
    st.write("Select an action:")
    if "subpage" not in st.session_state:
        st.session_state.subpage = None
    row1_cols = st.columns(3)
    with row1_cols[0]:
        if st.button("Data Bases"):
            st.session_state.subpage = "databases"
    with row1_cols[1]:
        if st.button("Add Colors"):
            st.session_state.subpage = "add"
    with row1_cols[2]:
        if st.button("Remove Colors"):
            st.session_state.subpage = "remove_colors"
    row2_cols = st.columns(3)
    with row2_cols[0]:
        if st.button("Create Custom Data Base"):
            st.session_state.subpage = "custom"
    with row2_cols[1]:
        if st.button("Remove Database"):
            st.session_state.subpage = "remove_database"
    with row2_cols[2]:
        st.write("")
    if st.session_state.subpage == "databases":
        show_databases_page()
    elif st.session_state.subpage == "add":
        show_add_colors_page()
    elif st.session_state.subpage == "remove_colors":
        show_remove_colors_page()
    elif st.session_state.subpage == "custom":
        show_create_custom_db_page()
    elif st.session_state.subpage == "remove_database":
        show_remove_database_page()

# --------------------------------------------------------------------
# --- Main Navigation (6 radio buttons)
# --------------------------------------------------------------------
def main():
    st.sidebar.title("Options")
    app_mode = st.sidebar.radio("Select Mode", [
        "Image Generator", 
        "Shape Detector", 
        "Oil Painting Generator", 
        "Colour Merger", 
        "Recipe Generator", 
        "Colors DataBase"
    ])
    if app_mode == "Image Generator":
        image_generator_app()
    elif app_mode == "Shape Detector":
        shape_detector_app()
    elif app_mode == "Oil Painting Generator":
        oil_painting_page()
    elif app_mode == "Colour Merger":
        color_mixing_app()
    elif app_mode == "Recipe Generator":
        painter_recipe_generator()
    elif app_mode == "Colors DataBase":
        painter_colors_database()

if __name__ == "__main__":
    main()
