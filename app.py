import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import math

# Load YOLO model
model = YOLO("best.pt")
# Define class names
class_names = ['Helmet', 'No Helmet']


# Streamlit app
def main():
    demo_video = 'bikes.mp4'
    st.title('Custom Object Detection YOLOV8')
    st.sidebar.title('Object Detection')

    # Create a radio button for selecting the input option
    option = st.sidebar.radio('Select Input Option', ['Video', 'Webcam'])

    if option == 'Video':
        # Display file uploader only when Video option is selected
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])

    # Button to stop the process
    stop_button = st.sidebar.button('Stop Process')

    # Confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.3)

    if option == 'Webcam':
        use_webcam = st.sidebar.button('Start Webcam')
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(demo_video)
    elif option == 'Video':
        if not video_file_buffer:
            vid = cv2.VideoCapture(demo_video)
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tffile:
                tffile.write(video_file_buffer.read())
                vid = cv2.VideoCapture(tffile.name)

    stframe = st.empty()

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Perform object detection using YOLOv8 model
        results = model(frame, stream=True)
        # Draw detection results on frame
        for result in results:
            boxes = result.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                currentclass = class_names[cls]
                if conf > confidence:
                    if currentclass == 'No Helmet':
                        mycolor = (0, 0, 255)
                    elif currentclass == 'Helmet':
                        mycolor = (0, 255, 0)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), mycolor, 3)
                    cv2.putText(frame, f'{currentclass} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mycolor, 2)

        stframe.image(frame, channels="BGR", use_column_width=True)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Check if the stop button is clicked
        if stop_button:
            break

    vid.release()


if __name__ == '__main__':
    main()
