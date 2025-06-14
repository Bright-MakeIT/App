
import streamlit as st
import google.generativeai as genai
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) #<-

st.title("Gemini-Bot")

# 모델 객체를 가져올 때 리소스 캐싱 기능을 사용합니다.
@st.cache_resource
def load_model():
  model = genai.GenerativeModel('gemini-2.0-flash-exp') 
  print("model loaded...")
  return model

model = load_model()


# 세셕별 이력 관리
if "chat_session" not in st.session_state:
  st.session_state["chat_session"] = model.start_chat(history=[]) # ChatSession 반환


# 대화 이력 출력
for content in st.session_state.chat_session.history:
  with st.chat_message("ai" if content.role == "model" else "user"):
    st.markdown(content.parts[0].text)


# 메시지 출력
if prompt := st.chat_input("메시지를 입력하세요."):
  with st.chat_message("user"):
    st.markdown(prompt)
  with st.chat_message("ai"):
    # 플레이스 홀더 적용
    message_placeholder = st.empty()
    full_response = ""
    with st.spinner("메시지 처리 중입니다..."):
      response = st.session_state.chat_session.send_message(prompt, stream=True) # ✅
      for chunk in response:
        full_response += chunk.text
        message_placeholder.markdown(full_response)
