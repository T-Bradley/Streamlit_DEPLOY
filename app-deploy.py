import streamlit as st
import PIL

import numpy
import utils
import io
from PIL import Image
import tempfile
import moviepy.editor as mpy
from camera_input_live import camera_input_live
import cv2


def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    fps = camera.get(cv2.CAP_PROP_FPS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_row=[]

    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            try: 
                visualized_image = utils.predict_image(frame, conf_threshold)

            except: 
                visualized_image = image
            st_frame.image(visualized_image, channels = "BGR")
            video_row.append(cv2.cvtColor(visualized_image,cv2.COLOR_BGR2RGB))

            frame_count +=1
            progress_bar.progress(frame_count/total_frames, text = None)

        else:
            camera.release()
            st_frame.empty()
            progress_bar.empty()
            break
    clip = mpy.ImageSequenceClip(video_row,fps=fps)
    clip.write_videofile(temp_file.name)

    st.video(temp_file.name)

st.set_page_config(
    page_title = "AI Fire Safety Project", 
    page_icon = ":fire:",
    layout = "centered", 
    initial_sidebar_state = "expanded")

st.title("AI Smoke and Fire Detection :fire:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the confidence threshold", 0, 100, 20))/100

input = None 
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)
  
        st.image(visualized_image, channels = "BGR")

    else: 
        st.image("assets/sample_image.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )

temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))

    if input is not None:

        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4" 

        with open(temporary_location, "wb") as out: 
            out.write(g.read())

        out.close() 

    if temporary_location is not None:

        play_video(temporary_location)

    else:
        st.video("assets/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an video." )


if source_radio == "WEBCAM":
    image = camera_input_live()

    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)

    st.image(visualized_image, channels = "BGR")























