import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import base64
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Set Streamlit page config (must be first command)
st.set_page_config(page_title="English ↔ Sinhala Translator", layout="wide")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and process dictionary dataset
def load_dictionary(file_path):
    if not os.path.exists(file_path):
        st.error(f"🚨 Dictionary file not found: {file_path}")
        return pd.DataFrame(columns=["English", "Sinhala"])
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                english, sinhala = parts
                data.append((english, sinhala))
    return pd.DataFrame(data, columns=["English", "Sinhala"])

# Define paths
file_path = "En-Si-dict-FastText-V2.txt"
df = load_dictionary(file_path)

# Define source and target languages
SRC_LANGUAGE = 'English'
TRG_LANGUAGE = 'Sinhala'

# Tokenization
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer("basic_english")
token_transform[TRG_LANGUAGE] = get_tokenizer("basic_english")

# Tokenize dataset
df["English Tokens"] = df[SRC_LANGUAGE].apply(token_transform[SRC_LANGUAGE])
df["Sinhala Tokens"] = df[TRG_LANGUAGE].apply(token_transform[TRG_LANGUAGE])

# Build Vocabulary
def yield_tokens(data_column):
    for sentence in data_column:
        yield sentence

vocab_transform = {}
vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(df["English Tokens"]), min_freq=2, specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_transform[TRG_LANGUAGE] = build_vocab_from_iterator(yield_tokens(df["Sinhala Tokens"]), min_freq=2, specials=["<unk>", "<pad>", "<sos>", "<eos>"])

for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    vocab_transform[ln].set_default_index(vocab_transform[ln]["<unk>"])

# Define Transformer-based Seq2Seq Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, n_layers=1, n_heads=2, pf_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, hid_dim)
        self.decoder = nn.Embedding(output_dim, hid_dim)
        self.transformer = nn.Transformer(d_model=hid_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, trg):
        enc_src = self.encoder(src)
        dec_trg = self.decoder(trg)
        transformer_output = self.transformer(enc_src, dec_trg)
        return self.fc_out(transformer_output)

# Load Model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"🚨 Model file not found: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        input_dim = state_dict["encoder.weight"].shape[0]
        output_dim = state_dict["decoder.weight"].shape[0]
        
        model = Seq2SeqTransformer(input_dim, output_dim)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

model_path = "best_translator.pth"
model = load_model(model_path)

def translate_text(input_text):
    if df.empty:
        return "🚨 Dictionary data is unavailable. Please upload a valid dictionary file."
    return df.set_index("English").to_dict()["Sinhala"].get(input_text, "Translation not found")

# Streamlit UI with Sri Lankan background
def main():
    # Load background image
    image_path = "sl_background.webp"
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        page_bg_css = f'''
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                        url("data:image/webp;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: #222222 !important;
            font-weight: bold !important;
        }}
        </style>
        '''
        st.markdown(page_bg_css, unsafe_allow_html=True)
    
    st.title("🌍 English ↔ Sinhala Translator")
    st.write("Enter an English or Sinhala word to translate.")
    
    user_input = st.text_area("Enter your text:", "")
    if st.button("Translate"):
        if user_input.strip():
            translated_text = translate_text(user_input)
            st.success(f"**Translation:** {translated_text}")
        else:
            st.warning("Please enter a word to translate.")

    st.write("---")
    st.subheader("How It Works")
    st.write("This web app uses a Transformer-based model to translate between English and Sinhala.")

if __name__ == "__main__":
    main()
