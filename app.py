import streamlit as st
import torch
import re
import html
from transformers import BertTokenizer, BertForSequenceClassification

#  Configuration de la page
st.set_page_config(
    page_title="Analyse de sentiments",
    page_icon="🎬",
    layout="centered"
)

#  Chargement du modèle
@st.cache_resource
def load_model():
    save_dir = "./bert_imdb_model"
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model = BertForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    return tokenizer, model

#  Nettoyage du texte 
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<.*?>", "", text)
    text = html.unescape(text)
    text = text.replace("\\'", "'")
    text = text.replace("\\/", "/")
    text = text.replace("\\", "")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prédiction 
def predict_sentiment(text, tokenizer, model):
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoding)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    label = torch.argmax(probs).item()
    confidence = probs[label].item()
    return label, confidence

#  Interface
st.title(" Analyse de sentiments — IMDb")
st.markdown(
    "Entrez un avis sur un film et obtenez une prédiction de sentiment "
    "(**positif** ou **négatif**) grâce à un modèle BERT fine-tuné sur le dataset IMDb."
)

st.divider()

try:
    tokenizer, model = load_model()

    user_input = st.text_area(
        "Votre avis :",
        placeholder="Ex: This movie was absolutely fantastic, the acting was superb...",
        height=150
    )

    if st.button("Analyser", type="primary"):
        if not user_input.strip():
            st.warning("Veuillez entrer un texte avant d'analyser.")
        else:
            with st.spinner("Analyse en cours..."):
                label, confidence = predict_sentiment(user_input, tokenizer, model)

            if label == 1:
                st.success(f"😊 **Sentiment positif** — confiance : {confidence:.1%}")
            else:
                st.error(f"😞 **Sentiment négatif** — confiance : {confidence:.1%}")

            with st.expander("Détails de la prédiction"):
                st.write(f"**Texte nettoyé :** {clean_text(user_input)}")
                st.write(f"**Label prédit :** {'Positif (1)' if label == 1 else 'Négatif (0)'}")
                st.write(f"**Score de confiance :** {confidence:.4f}")

except Exception:
    st.error(
        "Modèle introuvable. Assurez-vous que le dossier `bert_imdb_model/` "
        "est présent dans le même répertoire que `app.py`."
    )
    st.info("Pour générer le modèle, exécutez d'abord le notebook `Projet_Analyse_Sentiments_LLM.ipynb`.")

#  Foote
st.divider()
st.caption("Projet NLP | EFREI Paris · Promo Ingé3 BDML1 2025/2026")
