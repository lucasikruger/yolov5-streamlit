import streamlit as st
import torch
from PIL import Image
import cv2
from functions import *


# Main function to run the Streamlit app
def main():
    # Logo
    img = Image.open('logo.png')
    # Display the image as a logo at the top of the page
    st.image(img, use_column_width=True)
    # Set the title of the app
    st.title("YOLOv5 Object Detection")
    # Create a dropdown select box to choose the model
    model_source = st.selectbox('Select your model', ['yolov5s', 'yolov8n-pose'])

    if model_source == 'yolov5s':
        # Create a multi-select box to select the classes
        selected_classes = select_classes(model_source)

        # Create a slider to select the minimum confidence score threshold
        selected_confidence = st.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)
    else: 
        selected_classes = None
        selected_confidence = None
    # Create a file uploader to upload the image or video file
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    # Check if a file is uploaded
    if uploaded_file is not None:

        # Show the details of the uploaded file
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # If an image file is uploaded
        if uploaded_file.type.startswith('image/'):
            # Load the image
            orig_image = load_image(uploaded_file)
            
            # Display the uploaded image
            st.image(orig_image, caption='Uploaded Image.', use_column_width=True)
            
            # Create a button to start the inference
            if st.button('Infer'):
                # Load the selected model
                model = load_model(model_source)

                with st.spinner('Detecting...'):
                    run_folder = create_run_folder('images')
                    save_image(orig_image, f'{run_folder}/{uploaded_file.name}')
                    image = inference(orig_image, model, selected_classes, selected_confidence, model_source == 'yolov8n-pose')
                    save_image(image, f'{run_folder}/output.jpg')
                    st.image(image, caption='Detected Objects.', use_column_width=True)

        # If a video file is uploaded
        elif uploaded_file.type.startswith('video/'):
            run_folder = create_run_folder('videos')

            if st.button('Infer video'):
                # Load the selected model
                model = load_model(model_source)

                input_path =f'{run_folder}/{uploaded_file.name}'
                output_path = f'{run_folder}/output.mp4'
                with open(input_path, 'wb') as out_file:
                    out_file.write(uploaded_file.getbuffer())
                with st.spinner('Processing video...'):
                    convert_video(input_path)
                st.write("Original:")
                st.video(input_path)
                with st.spinner('Detecting...'):
                    process_video(input_path, output_path, model, selected_classes, selected_confidence, model_source == 'yolov8n-pose')
                    st.write("Output:")
                    st.video(output_path)

if __name__ == "__main__":
   main()
