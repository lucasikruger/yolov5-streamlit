import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import subprocess
import shutil
import datetime
from pathlib import Path

cocoClassesLst = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat", \
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",\
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",\
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",\
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
    
customClassesLst = ["person","bicycle","car"]
    

# Function to load the model, using @st.cache_resource for optimized loading

def load_model(model_name):
    # Check which model was selected and load corresponding model path
    if model_name == 'yolov5s':
        model_path = 'yolov5s.pt'
    # Set the device to CUDA if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    # Load the model using the Ultralytics YOLOv5 PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    # Send the model to the device
    model = model.to(device)
    return model

# Function to save an image to a specified path
def save_image(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Function to perform prediction on an image using a specified model
def predict(image, model):
    img = np.array(image)
    results = model(img)
    return results

# Function to convert a video file to the MP4 format using FFMPEG
def convert_video(path):
    temp_path = path + '.temp.mp4'
    subprocess.call(['ffmpeg', '-y', '-i', path, '-c:v', 'libx264', temp_path])
    shutil.move(temp_path, path)

# Function to perform inference on a frame using a specified model and selected classes
def inference(frame, model, wanted_classes, selected_confidence):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    # Filter out unwanted detections
    selected_results = []
    for *box, conf, cls in results.xyxy[0]:
        if ((wanted_classes is None) or (int(cls) in wanted_classes)) and conf > selected_confidence:
            selected_results.append((*box, conf, cls))
    results.xyxy[0] = torch.tensor(selected_results) if selected_results else torch.zeros((0,6))
    rendered = results.render()
    frame_bgr = cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR) if len(selected_results) > 0 else frame
    return frame_bgr

# Function to select classes from the model
def select_classes(model_source):
    # Assuming cocoClassesLst is defined
    if model_source == 'yolov5s':
        model_classes = cocoClassesLst
    else:
        model_classes = customClassesLst
    selected_classes = st.multiselect("Select Classes", range(len(model_classes)), format_func = lambda x: model_classes[x])
    # When no classes are selected, select all classes
    if len(selected_classes) == 0:
        selected_classes = range(len(model_classes))
    return selected_classes

# Function to process a video
def process_video(input_path, output_path, model, wanted_classes, selected_confidence):
    print('Processing video')
    # Load the video
    cap = cv2.VideoCapture(input_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Initialize the counter for processed frames
    processed_frames = 0

    # Loop through all frames in the video
    while cap.isOpened():
        print(f'Processing frame: {processed_frames}/{total_frames}')
        ret, frame = cap.read()
        if not ret:
            break
        frame_bgr = inference(frame, model, wanted_classes, selected_confidence)
        out.write(frame_bgr)

        # Update the progress bar
        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)

    # Release video resources when done
    cap.release()
    out.release()
    with st.spinner('Converting:'):
        convert_video(output_path)


# Function to create a new directory for each run
def create_run_folder(folder_name):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f'{folder_name}/run_{timestamp}'
    Path(run_folder).mkdir(parents=True, exist_ok=True)
    return run_folder
