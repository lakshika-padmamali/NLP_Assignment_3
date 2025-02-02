try:
    import streamlit as st
    import base64
except ModuleNotFoundError:
    print("🚨 Error: The 'streamlit' module is not installed. Please install it using 'pip install streamlit' and run the script again.")
    exit()

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- Define Attention Mechanism -------------------
class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == "multiplicative":
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "additive":
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        if self.method == "general":
            energy = self.attn(encoder_outputs)
            attention_weights = torch.bmm(hidden, energy.permute(0, 2, 1)).squeeze(1)
        elif self.method == "multiplicative":
            energy = torch.matmul(hidden, self.attn(encoder_outputs).permute(0, 2, 1))
            attention_weights = energy.squeeze(1)
        elif self.method == "additive":
            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            attention_weights = torch.sum(self.v * energy, dim=2)

        return torch.softmax(attention_weights, dim=1)

# ------------------- Define Transformer-based Seq2Seq Model -------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=128, n_layers=2, n_heads=4, pf_dim=256, dropout=0.1, attn_type="general"):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, hid_dim)
        self.decoder = nn.Embedding(output_dim, hid_dim)
        self.transformer = nn.Transformer(
            d_model=hid_dim, nhead=n_heads, num_encoder_layers=n_layers,
            num_decoder_layers=n_layers, dim_feedforward=pf_dim, dropout=dropout, batch_first=True
        )
        self.attention = Attention(attn_type, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, trg):
        enc_src = self.encoder(src)
        dec_trg = self.decoder(trg)
        transformer_output = self.transformer(enc_src, dec_trg)
        attn_weights = self.attention(enc_src[:, -1, :], enc_src)  # Apply attention to encoder output
        return self.fc_out(transformer_output), attn_weights

# ------------------- Load the Pre-trained Model -------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        print("🚨 Model file not found. Check the path.")
        return None
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            print("⚠️ 'state_dict' not found. Loading as raw state dictionary...")
            state_dict = checkpoint
        
        input_dim = state_dict.get("encoder.weight", torch.zeros(1, 1)).shape[0]
        output_dim = state_dict.get("decoder.weight", torch.zeros(1, 1)).shape[0]
        model = Seq2SeqTransformer(input_dim, output_dim)
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        return None

# ------------------- Load Tokenizer -------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ------------------- Ensure Model is Loaded -------------------
model_path = "best_comic_translator.pth"
model = load_model(model_path)

# ------------------- Translate Text Function -------------------
def translate_text(input_text):
    if model is None:
        return "🚨 Model not loaded properly. Check model file.", None
    
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output, attn_weights = model(tokens, tokens)
    translated_text = tokenizer.decode(output.argmax(dim=-1)[0], skip_special_tokens=True)
    
    return translated_text, attn_weights

# ------------------- Streamlit UI -------------------
def main():
    st.set_page_config(page_title="Text Simplifier", layout="wide")
    
    # Load background image
    image_path = "background_translator.webp"
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        page_bg_css = f'''
        <style>
        .stApp {{
            background: url("data:image/webp;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
            color: #000000;
            font-weight: bold;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: #000000 !important;
            font-weight: bold !important;
        }}
        </style>
        '''
        st.markdown(page_bg_css, unsafe_allow_html=True)
    
    st.title("Simplified English Translator")
    st.write("Enter an English sentence and get its simplified version!")

    user_input = st.text_area("Enter your text:", "")

    if st.button("Translate"):
        if user_input.strip():
            translated_text, attn_weights = translate_text(user_input)
            st.success(f"**Simplified English Translation:** {translated_text}")
        else:
            st.warning("Please enter a sentence to translate.")

    st.write("---")
    st.subheader("How It Works")
    st.write("This web app uses a Transformer-based model with General Attention to simplify complex English sentences into easier-to-understand versions.")

if __name__ == "__main__":
    main()
