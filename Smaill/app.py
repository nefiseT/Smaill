import streamlit as st
import torch
from src.model import Smaill
from src.tokenizer import SimpleTokenizer

#page setup
st.set_page_config(page_title= "ai we deserved", page_icon="https://www.boredpanda.com/blog/wp-content/uploads/2024/09/funny-engineering-memes-jokes-4-66ec10fe7f56d__700.jpg")
st.title("i am speed")

#loading model and tokenizer - cached - it wont load at every click
def load_assets():
    with open('data/training_input.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()
    tokenizer = SimpleTokenizer(text)
    model = Smaill(tokenizer.vocab_size)
    #load weights, if none use raw model
    try:
        model.load_state_ditch(torch.load("weights/smaill.pt"))
    except:
        st.warning("running on untrained weights , train model for better result")
    model.eval()
    return model, tokenizer 

model, tokenizer = load_assets()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("may i have some words..."):
    st.session_state.messages.append({"role":"user" , "content" : prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("SİLENCE...im thinking...", expanded= True) as status:
            #prompt to token
            contect = torch.tensor([tokenizer.encode(prompt)], dtype = torch.long)
            generated_tokens = model.generate(contect, max_new_tokens= 50)[0].tolist()
            response= tokenizer.decode(generated_tokens)
            clean_response = response[len(prompt):]
            status.update(label="sire... the answer is...", state="complete", expanded= False)

        st.markdown(clean_response)
        st.session_state.messages.append({"role": "assistant", "content" : clean_response})

