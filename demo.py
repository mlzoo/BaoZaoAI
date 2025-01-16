from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
import streamlit as st

# 初始化模型和tokenizer
@st.cache_resource
def load_model():
    model_name = "MLZoo/Qwen2.5-1.5B-Instruct-DPO-bad-boy"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后一个生成的token是否是停止token
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_response_streaming(chat_history):
    # 构建对话历史
    messages = [{"role": "system", "content": "你是一个没素质的人，请用暴躁的语气回复我，尽量多说一些。"}]
    # 添加历史对话
    for msg in chat_history:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 使用 streamer 进行生成
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # 设置生成参数
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "max_length": 2048,  # 增加最大长度以支持更长的对话
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])],
    }
    
    # 在单独的线程中进行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 实时输出生成的文本
    token_counts = 0
    
    # 创建一个空的占位符
    message_placeholder = st.empty()
    full_response = ""
    
    for new_text in streamer:
        if token_counts < 4:
            token_counts += 1
            continue
        full_response += new_text
        # 更新显示的文本
        message_placeholder.markdown(full_response + "▌")
    
    # 显示完整的回复
    message_placeholder.markdown(full_response)
    return full_response

# Streamlit界面设置
st.title("暴躁AI哥 🤖")
st.write("我是DPO train出来的暴躁AI哥，有什么问题尽管问我！")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("在这里输入你的问题..."):
    # 添加用户消息到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成助手回复
    with st.chat_message("assistant"):
        response = generate_response_streaming(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})