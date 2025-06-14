
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# Streamlit UI 구성
st.title("Stable Diffusion Image Creation with Options")

image_type = st.selectbox(
    "Image type:",
    [
        "general image",
        "icon",
        "logo",
        "tatoo design",
        "die-cut sticker",
        "minecraft skin",
        "custom emoji",
        "personalized bitmoji-style avatar",
        "personalized greeting card",
        "a poster",
        "a flyer",
    ]
)

user_prompt = st.text_input("What do you want to see?", "a cute cat")
style_option = st.selectbox("Image style:", ["natural", "vivid"], index=1)
quality_option = st.radio("Quality:", ["standard", "hd"], index=1)
aspect_ratio = st.selectbox("Aspect ratio:", ["512x512", "768x512", "512x768"])


# Stable Diffusion은 직접 style, quality 옵션이 없으니 prompt에 반영하거나 무시 가능
# def modify_prompt_based_on_selection(prompt, image_type):
#    return f"{image_type} of {prompt}"
def modify_prompt_based_on_selection(prompt, image_type, style, quality):
    style_text = f"in a {style} style" if style else ""
    quality_text = "high definition" if quality == "hd" else "standard quality"
    return f"{image_type} of {prompt}, {style_text}, {quality_text}"


# GPU 사용 여부 체크 및 파이프라인 초기화 (딱 1회만 실행)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

pipe = load_model()


# 스트림릿 이미지 생성 및 표시
if st.button("Create Image"):
    # modified_prompt = modify_prompt_based_on_selection(user_prompt, image_type)
    modified_prompt = modify_prompt_based_on_selection(user_prompt, image_type, style_option, quality_option)

    
    # 해상도 파싱 (기본 512x512 권장)
    width, height = map(int, aspect_ratio.split("x"))
    
    # 이미지 생성
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(modified_prompt, width=width, height=height).images[0]

    # 생성된 이미지 표시
    st.image(image, caption=f"Created Image: {modified_prompt}", use_container_width=True) #<-
    
    # 수정된 프롬프트 표시
    st.write(f"Prompt used: {modified_prompt}")
