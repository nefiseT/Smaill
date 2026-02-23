import streamlit as st
import torch
import os
from src.model import Smaill
from src.tokenizer import SimpleTokenizer

st.set_page_config(page_title="ai we deserved", page_icon="https://www.boredpanda.com/blog/wp-content/uploads/2024/09/funny-engineering-memes-jokes-4-66ec10fe7f56d__700.jpg")
st.title("i am speed")

weights_path = "weights/smaill.pt"
if os.path.exists(weights_path):
    st.caption(f"✓ Model loaded from: {weights_path}")
else:
    st.caption("⚠ No trained weights found - using random initialization")

@st.cache_resource
def load_assets():
    with open('data/training_input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = SimpleTokenizer(text)
    
    model = Smaill(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_embd=256,
        n_heads=4
    )
    
    try:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        st.caption("✓ Weights loaded successfully!")
    except Exception as e:
        st.warning(f"⚠ Running on untrained weights, train model for better result: {e}")
    
    model.eval()
    return model, tokenizer 

model, tokenizer = load_assets()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("may i have some words..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("SİLENCE...im thinking...", expanded=True) as status:
            encoded = tokenizer.encode(prompt)
            if not encoded:
                encoded = [0]
            
            input_length = len(encoded)
            context = torch.tensor([encoded], dtype=torch.long)
            
            full_output = model.generate(context, max_new_tokens=100)[0].tolist()

            generated_tokens = full_output[input_length:]
            response = tokenizer.decode(generated_tokens)
            
            status.update(label="sire... the answer is...", state="complete", expanded=False)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
