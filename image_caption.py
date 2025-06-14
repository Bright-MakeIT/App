
import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO

import google.generativeai as genai #<-
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) #<-
model = genai.GenerativeModel('gemini-2.0-flash-exp') #<-


# 이미지를 JPEG 형식으로 바이트 배열에 저장하고, Base64로 인코딩해 문자열로 반환합니다.
def encode_image_to_base64(image):
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode('utf-8')

st.title("이미지 설명 생성기")
uploaded_image = st.file_uploader("이미지를 업로드하세요: ", type=["jpg", "jpeg", "png"])

# 업로드된 이미지가 있으면, 이미지의 EXIF 데이터에서 회전 정보를 확인해서 화면에 올바르게 표시되게 조정합니다.
# 그리고 이미지 설명을 생성하는 버튼을 표시합니다. 사용자가 버튼을 클릭하면 이미지 설명을 생성합니다.
if uploaded_image is not None:
  image = Image.open(uploaded_image)

  # EXIF 데이터에서 회전 정보 확인 및 조정
  exif = image._getexif()
  if exif:
    orientation = exif.get(0x0112)
    if orientation == 3:
      image = image.rotate(180, expand=True)
    elif orientation == 6:
      image = image.rotate(270, expand=True)
    elif orientation == 8:
      image = image.rotate(90, expand=True)
  
  # st.image(image, caption="업로드된 이미지", use_column_width=True) #<- 경고
  st.image(image, caption='업로드된 이미지', use_container_width=True)
  if st.button("이미지 설명 생성"):
    base64_image = encode_image_to_base64(image)
    
    prompt = [{"text": "이 이미지에 무엇이 있나요?"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}] #<-
    response = model.generate_content(prompt) #<-
    try: #<-
      description = response.text #<-
      st.success("이미지 설명 생성 완료!")
      st.write(description)
    except Exception as e: #<-
      st.error("이미지를 분석할 수 없습니다.")
