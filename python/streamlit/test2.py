import streamlit as st

st.set_page_config(page_title="Baichuan-13B-Chat")
st.title("Baichuan-13B-Chat")

with st.chat_message("assistant", avatar='🤖'):
    st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
    with st.chat_message("user", avatar='🧑‍💻'):
        st.markdown(prompt)