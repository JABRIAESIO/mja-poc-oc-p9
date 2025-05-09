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

from models.model_loader import load_efficientnet_transformer_model, load_categories, get_model_paths
from models.inference import predict_image
from utils.visualization import plot_prediction_bars
from utils.preprocessing import resize_and_pad_image, apply_data_augmentation

st.set_page_config(
    page_title="Classification d'Images - ConvNeXtTiny",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Informations système pour le débogage
with st.sidebar:
    st.title("Informations système")
    st.write(f"Python version: {platform.python_version()}")
    st.write(f"Mémoire disponible: {psutil.virtual_memory().available / (1024 * 1024):.2f} MB")
    st.write(f"CPU count: {os.cpu_count()}")
    st.write(f"Working directory: {os.getcwd()}")
    
    # Test de connexion à Hugging Face
    if st.button("Tester la connexion à Hugging Face"):
        try:
            from models.model_loader import HF_MODEL_URL
            with st.spinner("Test en cours..."):
                response = requests.head(HF_MODEL_URL, timeout=10)
                if response.status_code == 200:
                    size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                    st.success(f"✅ Connexion réussie! \nTaille du modèle: {size_mb:.2f} MB")
                else:
                    st.error(f"❌ Erreur HTTP {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

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
            try:
                from models.model_loader import HF_MODEL_URL
                response = requests.head(HF_MODEL_URL, timeout=10)
                if response.status_code == 200:
                    size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                    st.success(f"✅ Connexion à Hugging Face réussie! Taille du modèle: {size_mb:.2f} MB")
                else:
                    st.error(f"❌ Erreur de connexion à Hugging Face: Code {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur lors du test de connexion à Hugging Face: {str(e)}")
                
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
            return None, None
