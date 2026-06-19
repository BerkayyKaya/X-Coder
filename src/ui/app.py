import streamlit as st
import time
from PIL import Image
import requests
import json

def response_generator(response):
    if response is None:
        return
    for word in response.split():
        for char in word:
            yield char
            time.sleep(0.01)
        yield " "
        time.sleep(0.05)

icon_image = Image.open("./src/ui/logo/x-coder_logo_2_copy.png")
BACKEND_URL = "http://127.0.0.1:8000/api/chat"

st.set_page_config(
    page_title = "X-Coder",
    page_icon = icon_image,
    layout = "centered"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bir şeyler yazın!"):

    with st.spinner("İş akışı çalıştırılıyor... Lütfen bekleyiniz!"):
        try:
            payload = {"query": prompt}
            st.chat_message("user").markdown(prompt)
            response = requests.post(BACKEND_URL, json = payload, stream = True)

            st.session_state.messages.append({
                "role" : "user",
                "content" : prompt
            })

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")

                        if decoded_line.startswith("data: "):
                            json_str = decoded_line.replace("data: ", "", 1)

                            try:
                                result = json.loads(json_str)

                                status = result.get("status")
                                status_message = result.get("status_message")

                                st.info(f"Durum: {status_message}")

                                if status == "done":
                                    st.success("İşlem tamamlandı!")
                                    bot_response = result.get("message", "Sonuç alınamadı") # eğer değer yoksa sonuç alınamadı olarak atayacak
                            except json.JSONDecodeError:
                                st.warning("Gelen paket JSON'a çevirilemedi!")
            else:
                st.error("Backend hatası!")



            with st.chat_message("assistant"):
                st.write_stream(response_generator(bot_response))
            
            st.session_state.messages.append({
                "role" : "assistant",
                "content": bot_response
            })
        
        except requests.exceptions.ConnectionError:
            st.error("FastAPI sunucusuna bağlanılamadı!")