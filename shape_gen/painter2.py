import streamlit as st
import itertools
import math
import numpy as np
import os

# Set page config only when this file is run directly.
if __name__ == '__main__':
        st.set_page_config(page_title="Painter App", layout="wide")
    st.sidebar.title("Options")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Recipe Generator", "Color DataBase"]
    )
# -----------------------------
# File name for our color database.
# -----------------------------
COLOR_DB_FILE = "color.txt"

# -----------------------------
# Read the color database from the text file.
# -----------------------------
@st.cache_data
def read_color_file(filename=COLOR_DB_FILE):
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        st.error("Error reading color.txt: " + str(e))
        return ""

# -----------------------------
# Parsing function: Reads the text file and creates a dictionary of databases.
# -----------------------------
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
            index = tokens[0]
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

def main():
    st.set_page_config(page_title="Image Generator, Shape Detector, & Oil Painting", layout="wide")
    st.sidebar.title("Options")
    app_mode = st.sidebar.radio(
        "Select Mode",
        [
            "Image Generator",
            "Shape Detector",
            "Oil Painting Generator",
            "Colour Merger",
            "Recipe Generator",
            "Color DataBase"
        ]
    )

    if app_mode == "Image Generator":
        st.header("Image Generator")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])
        # --- New options for number and size of shapes ---
        num_shapes = st.number_input("Enter the number of shapes to encode:", min_value=1, value=10)
        shape_size = st.number_input("Enter the size of the shape:", min_value=1, value=10)
        # -----------------------------------------------------
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
                encoded_image, boundaries = encode(img, shape, output_path="", num_shapes=num_shapes, shape_size=shape_size)
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

    elif app_mode == "Shape Detector":
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
                col1, col2, col3 = st.columns(3)
                for idx, (color, count) in enumerate(grouped_colors):
                    rgb_str = f"RGB: {color} - Count: {count}"
                    color_box = f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); height: 30px; width: 30px; margin-right: 10px; display: inline-block;"
                    if idx % 3 == 0:
                        with col1:
                            st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                    elif idx % 3 == 1:
                        with col2:
                            st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                    else:
                        with col3:
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

    elif app_mode == "Oil Painting Generator":
        oil_painting_page()

    elif app_mode == "Colour Merger":
        color_mixing_app()

    # -----------------------------
    # New Modes: Recipe Generator and Color DataBase
    # -----------------------------
    elif app_mode == "Recipe Generator":
        import painter2
        painter2.main()
    elif app_mode == "Color DataBase":
        import painter2
        painter2.main()

if __name__ == "__main__":
    main()
