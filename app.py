import streamlit as st
import numpy as np
import ultralytics
from ultralytics import YOLO
from IPython import display
from IPython.display import display, Image
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import openai

api_key = "API_KEY"
openai.api_key = api_key

routine = False
class_counts = [0] * 6
class_labels = ['blackheads', 'dark spot', 'nodules', 'papules', 'pustules', 'whiteheads']

model = YOLO('model/best.pt')

def analys(image):
  analys_text = ""
  level = "not defined"
  numpy_image = np.array(image)
  results = model(source = image)
  names = model.names
  new_image = np.copy(numpy_image)
  for result in results:                        
    boxes = result.boxes.cpu().numpy()   


    for i in range(len(boxes)):                                          
        r = boxes[i].xyxy[0].astype(int)
        cls = names[int(result.boxes.cls[i])]

        cv2.rectangle(new_image, (r[0], r[1]), (r[2], r[3]), (255, 255, 255), 2)
        cv2.putText(new_image, cls, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  
  for r in results:
    detections = r.boxes.cls
    for detection in detections:
        class_id = int(detection)
        class_counts[class_id] += 1
  
  for i, count in enumerate(class_counts):
    analys_text+= f'{class_labels[i]}-{count}; '

  
  for i in range(5):
     if class_counts[0] == 0 and class_counts[1] == 0 and class_counts[2] == 0  and class_counts[3] == 0  and class_counts[4] == 0 and class_counts[5] == 0:
        level = "Grade Zero"
     elif class_counts[0] >= 0 and class_counts[2] == 0 and class_counts[3] < 5 and class_counts[4] == 0  and class_counts[5] >= 0:
        level = "Grade 1"
     elif class_counts[0] >= 0 and class_counts[2] == 0 and class_counts[3] >= 0 and class_counts[4] < 5:
        level = "Grade 2"
     elif class_counts[0] >= 0 and class_counts[2] < 10 and class_counts[3] >= 0 and class_counts[4] < 10:
        level = "Grade 3"
     elif class_counts[2] >= 10 and class_counts[4] >= 10:
        level = "Grade 4"
    
  prompt = f'Hello. I have a {analys_text}. Can u write a morning and night routine for me. Dont write anything not related to routine'
  response = openai.chat.completions.create(
     model="gpt-3.5-turbo-1106",
     messages=[
    {"role": "system", "content": prompt},
  ],
    )
  routine_text = response.choices[0].message.content

  return analys_text, level, new_image, routine_text
  

st.set_page_config(
    page_title="AcneAI",
    layout="wide"
)

tab1, tab2, tab3 = st.tabs(["Home", "Analysis", "FAQ"])


with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Step 1: Upload a photo")
        st.image("https://cdn-icons-png.flaticon.com/512/2818/2818147.png", width=400)
        st.write("Start by snapping a clear picture of your face in good lighting. Don't be shy! The more your skin is visible, the better the AI can work its magic. Upload this photo through our user-friendly platform and let the journey to clearer skin begin!")

    with col2:
        st.header("Step 2: AI-Powered Analysis")
        st.image("https://cdn-icons-png.flaticon.com/512/826/826118.png", width=400)
        st.write("Hang tight while our advanced neural network gets to work. It meticulously scans your photo, identifying different types of acne and skin conditions with precision. This high-tech analysis goes way beyond the surface to understand what your skin truly needs.")

    with col3:
        st.header("Step 3: Discover a complete routine")
        st.image("https://cdn-icons-png.flaticon.com/512/6819/6819619.png", width=400)
        st.write("After the AI does its thing, you'll receive a complete, step-by-step skincare routine that's crafted just for you. We're talking about a selection of products and lifestyle tips that suit your unique skin type and acne condition. It’s your personal roadmap to clearer, healthier skin, powered by the smarts of AI and the care of skincare experts.")

    uploaded_photo = st.file_uploader("Upload a photo", accept_multiple_files=False, type = ["png", "jpg"])
    if uploaded_photo is not None:
        photo = Image.open(io.BytesIO(uploaded_photo.read()))
        analys_text, grade_level, analys_image, routine_text = analys(photo)
        analys_text = "You have: " + analys_text
        routine = True




with tab2:
    if routine == False:
       st.header("Looks like you have not made analysis yet!")
    elif routine == True:
       st.header("Here is your result")
       col4, col5 = st.columns(2)
       with col4:
        st.header("Photo Result")
        st.image(analys_image, width = 550)

       with col5:
        st.header(grade_level)
        st.write(analys_text)
        st.write(routine_text)
        st.markdown("*Please note that our web-application does not replace a professional cosmetologist.*")



with tab3:
   st.header("Which architecture of CNN is used?")
   st.write("Our web-application uses model on a YOLOv8 architecture, trained using 1255 images with labels of 6 different types of acne.")
   st.header("What is used for routine generation?")
   st.write("For now, we are using gpt-3.5-turbo-1106 throught api.")
