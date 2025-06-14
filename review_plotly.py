
import streamlit as st
import google.generativeai as genai #<-
from pydantic import BaseModel
import numpy as np
import plotly.graph_objects as go

# 문자열을 딕셔너리로 변환
import ast #<- 

st.title("상품 리뷰 분석 및 시각화")

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) #<-

# 리뷰 입력 받기
reviews_input = st.text_area("상품 리뷰를 입력하세요 (리뷰 긴 줄바꿈으로 구분):", height=200)

if st.button("리뷰 분석 및 시각화"):
  # 입력된 리뷰를 줄바꿈 기준으로 분리
  reviews = reviews_input.split("\n") #<-⚠️

  # 리뷰 분석을 위한 프롬프트
  analysis_prompt = """
  상품에 대한 고객의 점수와 리뷰입니다. 각 상품의 고객의 점수와 리뷰를 분석하여 카테고리별로 1부터 5까지의 점수를 부여하세요.
  이때 고객의 점수에 입각하여 해당 카테고리별 점수에 반영되도록 하세요.
  그리고 해당 없는 카테고리별 리뷰일지라도 반드시 점수는 0으로 부여해야 합니다.
  """
  model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=analysis_prompt) #<-

  # Pydantic 모델 정의
  class ReviewAnalysis(BaseModel):
    품질: int
    배송: int
    가격: int
    고객서비스: int

  # 카테고리별 점수 수집 초기화
  category_scores = {"품질": [], "배송": [], "가격": [], "고객서비스": []}

  # 리뷰 분석
  analyzed_reviews = []
  for review in reviews:
    # (OpenAI API 대신)Google API를 이용해 리뷰 분석
    try:
      completion = model.generate_content(review,
                                          generation_config=genai.GenerationConfig(response_mime_type="application/json",
                                                                                   response_schema=ReviewAnalysis),) #<-
      candidate = completion.candidates[0] #<-
      if candidate.finish_reason == 1: #<-
        message = completion.text #<-
        message = ast.literal_eval(message)  #<- 문자열을 딕셔너리로 변환
        analyzed_reviews.append(message) #<-

        # 카테고리별 점수 수집
        for category in category_scores.keys():
          category_scores[category].append(message[category]) #<-
      elif candidate.finish_reason == 3: #<-
        st.warning(f"리뷰 분석에 거부했습니다: {review}") #<-
      else:
        st.warning(f"리뷰 분석에 실패했습니다: {review}")
    except Exception as e:
      st.warning(f"리뷰 분석 중 오류가 발생했습니다: {review}\n오류 메시지: {e}") #<-⚠️

  # 분석된 리뷰 출력
  st.subheader("분석된 리뷰 점수")
  for idx, analyzed_review in enumerate(analyzed_reviews):
    st.write(f"리뷰 {idx+1}: {analyzed_review}") #<-

  # 카테고리별 평균 점수 계산
  category_avg_scores = {category: np.mean(scores) if scores else 0 for category, scores in category_scores.items()}

  # Plotly 극좌표계 차트 그리기
  fig = go.Figure()
  categories = list(category_avg_scores.keys())
  scores = list(category_avg_scores.values())
  categories.append(categories[0]) # 처음 카테고리를 리스트 끝에 추가해 차트를 완성합니다.
  scores.append(scores[0]) # 처음 점수를 리스트 끝에 추가합니다.
  fig.add_trace(go.Scatterpolar(r=scores, theta=categories, fill='toself'))
  fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)

  # Streamlit에 Plotly 차트 표시
  st.plotly_chart(fig, use_container_width=True)
