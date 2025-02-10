import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
from painterfun import oil_main  # Importing the oil_main function

def main():
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
            #output_image_cv = cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB)

            # Convert to uint8 before passing to PIL.Image
            output_image_cv = (output_image_cv * 255).astype(np.uint8)

            # Convert the OpenCV image back to PIL for Streamlit compatibility
            output_image = Image.fromarray(output_image_cv)

            # Show the processed image in the second column
            with col2:
                st.image(output_image, caption="Processed Image", use_column_width=True)

            # Convert the processed image to bytes for downloading
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format="PNG")  # Save in PNG format
            img_byte_arr.seek(0)

            # Create a download button for the processed image
            st.download_button(
                label="Download Processed Image",
                data=img_byte_arr,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
