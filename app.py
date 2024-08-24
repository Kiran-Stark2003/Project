import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8l.pt')

def detect_cell_phone(image):
    # Convert image from PIL to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(image)

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates, class, and confidence
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            class_id = int(box.cls)
            confidence = box.conf[0]

            if class_id == 67:  # class_id 67 corresponds to 'cell phone' in COCO dataset
                # Draw rectangle around detected object
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Add label with confidence score
                label = f"cell phone: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y_min = max(y_min, label_size[1] + 10)
                cv2.rectangle(image, (x_min, label_y_min - label_size[1] - 10), 
                              (x_min + label_size[0], label_y_min + 5), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, label, (x_min, label_y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 0), 1)

                # Add a thicker boundary on the bottom of the rectangle
                cv2.line(image, (x_min, y_max), (x_max, y_max), (0, 255, 0), 4)

    # Convert image back to PIL format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

# Set page config
st.set_page_config(
    page_title="Mobile Phone Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📱 Mobile Phone Detection - Live Streams")

# Sidebar content
st.sidebar.header("About")
st.sidebar.write("""
This app uses a YOLOv8 model to detect mobile phones in live video streams. You can choose to view one or two camera feeds.
""")

# Toggle for selecting number of cameras
camera_mode = st.sidebar.radio(
    "Select Camera Mode",
    ("Single Camera", "Dual Cameras")
)

# Dropdowns to select camera indices
camera_index1 = st.sidebar.selectbox(
    'Select First Camera',
    ('0', '1', '2', '3'),
    index=0,
    format_func=lambda x: f"Camera {x}"
)

camera_index2 = st.sidebar.selectbox(
    'Select Second Camera',
    ('0', '1', '2', '3'),
    index=1,
    format_func=lambda x: f"Camera {x}"
)

# Function to process the video stream from a single camera
def process_video_stream(camera_index, col_name):
    cap = cv2.VideoCapture(int(camera_index))
    st_frame = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write(f"Failed to capture image from {col_name}")
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # Detect cell phone in the frame
        result_img = detect_cell_phone(img_pil)
        
        # Display the image in the appropriate column
        with st_frame.container():
            st.image(result_img, caption=f"Live Feed from {col_name}", use_column_width=True)

    cap.release()

# Display based on selected mode
if camera_mode == "Single Camera":
    # Display single camera feed
    process_video_stream(camera_index1, "Selected Camera")

elif camera_mode == "Dual Cameras":
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Run the video streams in separate columns
    with col1:
        process_video_stream(camera_index1, "First Camera")

    with col2:
        process_video_stream(camera_index2, "Second Camera")
