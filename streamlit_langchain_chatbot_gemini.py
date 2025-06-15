
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# GPT 대신 Gemini 사용
from langchain_google_genai import ChatGoogleGenerativeAI #<-

import streamlit as st
st.title("스트림릿 랭체인 제미나이 챗봇 예제")


# 대화 기록을 저장할 메모리 설정
if "chat_history" not in st.session_state: #<- StreamlitChatMessageHistory는 stateful하게 동작하도록 설계
  st.session_state.chat_history = StreamlitChatMessageHistory() #<-

msgs = st.session_state.chat_history #<- StreamlitChatMessageHistory는 stateful하게 동작하도록 설계

# 대화 기록이 비어있다면 인사말을 추가합니다.
if len(msgs.messages) == 0:
  msgs.add_ai_message("안녕하세요?")

# 프롬프트 정의 : 시스템 메시지, 대화 기록, 사용자 질문으로 구성합니다.
# MessagesPlaceholder를 사용해 대화 기록을 프롬프트에 동적으로 삽입할 수 있습니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# ChatOpenAI 사용한 언어 모델 초기화 대신 ChatGoogleGenerativeAI 사용
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
) #<-


# 언어 모델을 프롬프트와 연결해 체인을 생성
chain_gemini = prompt | llm_gemini #<-

# 체인과 대화 기록을 RunnableWithMessageHistory 사용해 연결합니다. 
# 이렇게 하면 체인이 실행될 때마다 대화 기록이 자동으로 업데이트 됩니다.
chain_with_history_gemini = RunnableWithMessageHistory(
  chain_gemini, #<-
  lambda session_id: msgs,
  input_messages_key="question",
  history_messages_key="history",
)

# 저장된 이전 대화 내용을 화면에 표시
# msgs.message를 반복하면서 각 메시지를 적절한 유형(사용자 또는 챗봇)으로 표시합니다.
for msg in msgs.messages:
  st.chat_message(msg.type).write(msg.content)

# 사용자가 새로운 메시지를 입력하면, chain_with_history를 실행해 챗봇의 응답을 생성합니다.
# 새로 생성된 메시지는 자동으로 대화 기록에 추가되고, 화면에 표시됩니다.
if prompt := st.chat_input():
  st.chat_message("human").write(prompt)
  config = {"configurable": {"session_id": "any"}}
  response = chain_with_history_gemini.invoke({"question": prompt}, config)
  st.chat_message("ai").write(response.content)
