import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import glob
import random
import traceback
import json
import requests
import sys
import platform
import psutil

from models.model_loader import load_efficientnet_transformer_model, load_categories, get_model_paths, get_hugging_face_token, HF_MODEL_URL
from models.inference import predict_image
from utils.visualization import plot_prediction_bars
from utils.preprocessing import resize_and_pad_image, apply_data_augmentation

st.set_page_config(
    page_title="Classification d'Images - ConvNeXtTiny",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

def test_hugging_face_connection():
    """
    Teste la connexion à Hugging Face en vérifiant l'accès au modèle.
    """
    try:
        # Obtenir le token
        hf_token = get_hugging_face_token()
        
        # Préparer les headers
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            st.info("🔑 Token d'authentification trouvé")
        else:
            st.warning("⚠️ Aucun token d'authentification trouvé")
        
        # Faire la requête
        with st.spinner("Test en cours..."):
            response = requests.head(HF_MODEL_URL, headers=headers, timeout=10)
            
            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                st.success(f"✅ Connexion réussie! \nTaille du modèle: {size_mb:.2f} MB")
                return True
            else:
                st.error(f"❌ Erreur HTTP {response.status_code}")
                
                # Afficher des informations de débogage
                with st.expander("Détails de l'erreur"):
                    st.write({
                        "Status Code": response.status_code,
                        "Headers": dict(response.headers),
                        "URL": HF_MODEL_URL,
                        "Token présent": bool(hf_token)
                    })
                return False
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        st.exception(e)
        return False

# Informations système pour le débogage
with st.sidebar:
    st.title("Informations système")
    st.write(f"Python version: {platform.python_version()}")
    st.write(f"Mémoire disponible: {psutil.virtual_memory().available / (1024 * 1024):.2f} MB")
    st.write(f"CPU count: {os.cpu_count()}")
    st.write(f"Working directory: {os.getcwd()}")

    # Test de connexion à Hugging Face avec la nouvelle fonction
    if st.button("Tester la connexion à Hugging Face"):
        test_hugging_face_connection()

st.title("🦜 Classification d'Images avec ConvNeXtTiny")
st.markdown("""
Cette application utilise un modèle ConvNeXtTiny pour classifier les images.
Téléchargez une image ou utilisez un exemple pour voir le modèle en action! Mourad JABRI 2025
""")

@st.cache_resource
def load_model():
    with st.spinner('Chargement du modèle... Cela peut prendre quelques instants.'):
        try:
            # Test de connexion à Hugging Face
            st.info("Test de connexion à Hugging Face...")
            test_result = test_hugging_face_connection()
            if not test_result:
                st.warning("La connexion à Hugging Face a échoué, mais nous allons essayer de charger le modèle quand même.")

            # Obtention des chemins (fonction maintenue pour compatibilité)
            paths = get_model_paths()
            st.write("Chemins recherchés:", paths)

            # Mesure du temps de chargement du modèle
            start_time = time.time()

            # Utilisation de la fonction load_efficientnet_transformer_model qui redirige vers ConvNeXtTiny
            st.info("Chargement du modèle ConvNeXtTiny depuis Hugging Face...")
            model = load_efficientnet_transformer_model()

            end_time = time.time()
            loading_time = end_time - start_time

            if model is None:
                st.error("Le modèle n'a pas pu être chargé (retourné None)")
                
                # Vérifier si le token est présent
                hf_token = get_hugging_face_token()
                if not hf_token:
                    st.error("Aucun token Hugging Face trouvé. Veuillez l'ajouter dans les secrets Streamlit.")
                    st.info("Pour ajouter un token Hugging Face, allez dans les paramètres de votre application Streamlit Cloud, puis dans l'onglet 'Secrets' et ajoutez: HF_TOKEN = 'votre_token'")
                
                return None, None

            # Vérification du modèle chargé
            st.write("### Vérification du modèle chargé")
            st.write(f"Nombre de couches: {len(model.layers)}")
            st.write(f"Shape de sortie: {model.output_shape}")
            st.write(f"Temps de chargement: {loading_time:.2f} secondes")

            # Afficher le type de modèle chargé
            import tensorflow as tf
            model_size = sum([tf.keras.backend.count_params(w) for w in model.weights])
            st.write(f"Taille du modèle: {model_size/1000000:.2f} M paramètres")

            if model_size < 10_000_000:  # 10M paramètres
                st.success("✅ Modèle ConvNeXtTiny chargé")
            else:
                st.info("✅ Modèle standard chargé")

            categories = load_categories()
            st.success("Modèle et catégories chargés avec succès!")
            return model, categories

        except Exception as e:
            st.error(f"Erreur détaillée lors du chargement: {str(e)}")
            st.error("Traceback complet:")
            st.code(traceback.format_exc())
            
            # Vérifier si le token est présent
            hf_token = get_hugging_face_token()
            if not hf_token:
                st.error("Aucun token Hugging Face trouvé. Veuillez l'ajouter dans les secrets Streamlit.")
                st.info("Pour ajouter un token Hugging Face, allez dans les paramètres de votre application Streamlit Cloud, puis dans l'onglet 'Secrets' et ajoutez: HF_TOKEN = 'votre_token'")
            
            return None, None

# Ajoutez ici le reste de votre application, y compris le chargement du modèle, 
# l'interface utilisateur pour télécharger les images, etc.
# Par exemple:

# Chargement du modèle
model, categories = load_model()

# Interface utilisateur pour l'upload d'images
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Traitement de l'image téléchargée
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)
        
        # Prédiction
        if model is not None and categories is not None:
            with st.spinner("Classification en cours..."):
                predictions = predict_image(model, image, categories)
                
                # Afficher les résultats
                st.subheader("Résultats de la classification")
                # Visualisation des résultats
                fig = plot_prediction_bars(predictions)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image: {str(e)}")
        st.code(traceback.format_exc())
