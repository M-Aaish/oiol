import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from EnDe import decode, encode
from painterfun import oil_main  # Importing the oil_main function

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
            # If the color is close enough to an existing group, add it to the group
            if color_distance(color, group[0]) < threshold:
                grouped_colors[i].append(color)
                counts[i] += 1
                found_group = True
                break
        if not found_group:
            # If no close group is found, start a new group
            grouped_colors.append([color])
            counts.append(1)

    # Return a list of colors representing the group (typically the first color) and their counts
    return [(group[0], count) for group, count in zip(grouped_colors, counts)]

def oil_painting_page():
    st.title("Oil Painting Image Generator")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Input field to accept an integer argument for the oil_main function
    intensity = st.number_input("Enter the intensity (integer):", min_value=1, max_value=100, value=10)

    # Create two columns, one for the uploaded image and one for the processed image
    col1, col2 = st.columns(2)

    # Show the uploaded image in the first column
    with col1:
        if uploaded_file is not None:
            # Load the uploaded image using PIL
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.write("Upload an image to see it here")

    # Show an empty placeholder for the output image in the second column


    # A button to trigger the oil painting generation
    if st.button("Generate"):
        if uploaded_file is not None:
            # Convert the uploaded PIL image to a numpy array (OpenCV format)
            input_image_cv = np.array(input_image)

            # Ensure that the image is in the correct format for OpenCV (BGR)
            if len(input_image_cv.shape) == 2:  # If grayscale, convert to RGB
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_GRAY2RGB)
            elif input_image_cv.shape[2] == 4:  # If image has alpha channel, remove it
                input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_RGBA2RGB)

            # Process the image with the oil_main function, passing the intensity argument
            output_image_cv = oil_main(input_image_cv, intensity)  # Pass OpenCV image and intensity

            # Convert processed image back to RGB (OpenCV uses BGR by default)
            # output_image_cv = cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB)

            # Convert to uint8 before passing to PIL.Image
            output_image_cv = (output_image_cv * 255).astype(np.uint8)

            # Convert the OpenCV image back to PIL for Streamlit compatibility
            output_image = Image.fromarray(output_image_cv)

            # Show the processed image in the second column
            with col2:
                st.image(output_image, caption="Processed Image", use_column_width=True)

            # Convert the processed image to bytes for downloading
            img_byte_arr = BytesIO()
            output_image.save(img_byte_arr, format="PNG")  # Save in PNG format
            img_byte_arr.seek(0)

            # Create a download button for the processed image
            st.download_button(
                label="Download Processed Image",
                data=img_byte_arr,
                file_name="processed_image.png",
                mime="image/png"
            )

def main():
    # Streamlit app layout
    st.set_page_config(page_title="Image Generator, Shape Detector, & Oil Painting", layout="wide")

    # Sidebar with page selection
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Select Mode", ["Image Generator", "Shape Detector", "Oil Painting Generator"])

    if app_mode == "Image Generator":
        st.header("Image Generator")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        shape_option = st.selectbox("Select Shape", ["Triangle", "Rectangle", "Circle"])

        # Create two columns
        col1, col2 = st.columns([1, 1])  # This creates two equal-width columns

        # Initially, we show the uploaded image in col1 if it's available
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Error reading the image. Please try another file.")
            else:
                # Convert uploaded image to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Display uploaded image in the left column
                with col1:
                    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        # When the Generate button is clicked
        if st.button("Generate"):
            if uploaded_file is not None:
                shape = shape_option  # encode handles shape conversion internally
                encoded_image, boundaries = encode(img, shape, output_path="")  # encoded image and boundaries
                encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)

                # Display the encoded image in the second column
                with col2:
                    st.image(encoded_image_rgb, caption=f"Encoded {shape_option} Image", use_container_width=True)

                # Provide the download button for encoded image
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

        # Create two columns
        col1, col2 = st.columns([1, 1])  # This creates two equal-width columns

        # Initially, we show the uploaded encoded image in col1 if it's available
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            encoded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if encoded_image is None:
                st.error("Error reading the image. Please try another file.")
            else:
                # Convert uploaded encoded image to RGB
                encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)

                # Display uploaded encoded image in the left column
                with col1:
                    st.image(encoded_image_rgb, caption="Uploaded Encoded Image", use_container_width=True)

        # When the Decode button is clicked
        if st.button("Decode"):
            if uploaded_file is not None:
                shape = shape_option  # decode handles shape conversion internally
                binary_img, annotated_img, rgb_vals = decode(encoded_image, shape, boundaries=None)  # Correct unpacking here

                # Apply the color grouping function
                grouped_colors = group_similar_colors(rgb_vals, threshold=10)

                # Sort colors by count in descending order
                grouped_colors = sorted(grouped_colors, key=lambda x: x[1], reverse=True)

                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # Display the decoded annotated image in the second column
                with col2:
                    st.image(annotated_img_rgb, caption=f"Decoded Annotated {shape_option} Image", use_container_width=True)

                # Show grouped colors with RGB and count in 3 columns
                st.subheader("Grouped Colors (Ranked by Count)")

                # Create 3 columns to display the colors in a row-wise fashion
                col1, col2, col3 = st.columns(3)

                # Loop through the colors and display them across the columns
                for idx, (color, count) in enumerate(grouped_colors):
                    rgb_str = f"RGB: {color} - Count: {count}"

                    # Choose the column based on the index
                    color_box = f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); height: 30px; width: 30px; margin-right: 10px; display: inline-block;"

                    # Determine which column to display the color in
                    if idx % 3 == 0:
                        with col1:
                            st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                    elif idx % 3 == 1:
                        with col2:
                            st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)
                    else:
                        with col3:
                            st.markdown(f"<div style='{color_box}'></div> {rgb_str}", unsafe_allow_html=True)

                # Provide the download button for decoded image
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

if __name__ == "__main__":
    main()
