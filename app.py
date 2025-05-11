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

# Configuration de la page avec préoccupation d'accessibilité
st.set_page_config(
    page_title="Classification d'Images - ConvNeXtTiny",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajout de CSS pour améliorer l'accessibilité
st.markdown("""
<style>
    /* Augmenter le contraste des textes */
    .stMarkdown, .stText, .stHeading, .stButton, .stRadio, h1, h2, h3, p {
        color: #191919 !important;
    }
    
    /* Augmenter la taille de la police pour une meilleure lisibilité */
    .stMarkdown, .stText, p {
        font-size: 16px !important;
    }
    
    /* Meilleur contraste pour les titres */
    h1, h2, h3, .stHeading {
        font-weight: 600 !important;
    }
    
    /* Assurer que les liens sont facilement identifiables */
    a {
        color: #0056b3 !important;
        text-decoration: underline !important;
    }
    
    /* Focus visible pour les éléments interactifs */
    button:focus, input:focus, select:focus {
        outline: 3px solid #4299e1 !important;
    }
    
    /* Améliorer la visibilité des boutons */
    .stButton button {
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Meilleur espacement pour les éléments de formulaire */
    .stRadio, .stSelectbox, .stFileUploader {
        margin-bottom: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

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

        # Faire la requête - Utiliser GET au lieu de HEAD et suivre les redirections
        with st.spinner("Test de connexion en cours. Veuillez patienter..."):
            # Test direct sur le modèle avec GET et streaming
            response = requests.get(
                HF_MODEL_URL,
                headers=headers,
                stream=True,  # Important pour ne pas télécharger tout le contenu
                timeout=10,
                allow_redirects=True  # Important: suivre les redirections
            )
            
            # Lire juste un peu de contenu pour confirmer l'accès
            for chunk in response.iter_content(chunk_size=1024):
                break
                
            st.write(f"URL finale après redirection: {response.url}")
            
            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                st.success(f"✅ Connexion réussie! Taille du modèle: {size_mb:.2f} MB")
                return True
            else:
                st.error(f"❌ Erreur HTTP {response.status_code}")
                
                # Afficher des informations de débogage
                with st.expander("Détails de l'erreur (pour le dépannage)"):
                    st.write({
                        "Code d'état": response.status_code,
                        "URL demandée": HF_MODEL_URL,
                        "URL finale": response.url,
                        "En-têtes": dict(response.headers),
                        "Token présent": "Oui" if bool(hf_token) else "Non"
                    })
                return False
    except Exception as e:
        st.error(f"❌ Erreur de connexion: {str(e)}")
        with st.expander("Détails techniques de l'erreur"):
            st.exception(e)
        return False

# Informations système pour le débogage
with st.sidebar:
    st.title("Informations système")
    
    # Afficher les informations avec des étiquettes claires
    system_info = {
        "Version Python": platform.python_version(),
        "Mémoire disponible": f"{psutil.virtual_memory().available / (1024 * 1024):.2f} MB",
        "Nombre de CPU": os.cpu_count(),
        "Répertoire de travail": os.getcwd()
    }
    
    for label, value in system_info.items():
        st.markdown(f"**{label}:** {value}")

    # Afficher l'URL et le chemin du modèle pour débogage
    paths = get_model_paths()
    st.markdown("### Informations modèle")
    st.markdown(f"**URL Hugging Face:** {HF_MODEL_URL}")
    st.markdown(f"**Chemin local:** {paths['convnext_model']}")

    # Test de connexion à Hugging Face avec la nouvelle fonction
    st.button("Tester la connexion à Hugging Face", help="Vérifie si l'application peut accéder au modèle sur Hugging Face", on_click=test_hugging_face_connection)

# Titre principal avec description claire
st.title("🦜 Classification d'Images avec ConvNeXtTiny")
st.markdown("""
Cette application permet de classifier des images selon différentes catégories en utilisant un modèle ConvNeXtTiny.

**Comment utiliser cette application:**
1. Choisissez une image (en la téléchargeant ou en sélectionnant un exemple)
2. Le modèle analysera automatiquement l'image
3. Les résultats de classification s'afficheront ci-dessous

Développé par Mourad JABRI, 2025
""")

def load_example_images():
    """
    Charge les exemples d'images disponibles dans le dossier assets/examples
    
    Returns:
        list: Liste des chemins d'accès aux images exemple
    """
    examples_dir = os.path.join("assets", "examples")
    if not os.path.exists(examples_dir):
        st.warning(f"⚠️ Le dossier d'exemples '{examples_dir}' n'existe pas.")
        # Afficher le contenu du répertoire actuel pour le débogage
        with st.expander("Informations de dépannage"):
            st.write("Contenu du répertoire actuel:", os.listdir("."))
        return []
    
    image_files = [f for f in os.listdir(examples_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        st.warning(f"⚠️ Aucune image trouvée dans le dossier '{examples_dir}'")
        return []
    
    return [os.path.join(examples_dir, f) for f in image_files]

@st.cache_resource
def load_model():
    """
    Charge le modèle de classification et les catégories.
    Cette fonction est mise en cache pour éviter de recharger le modèle à chaque interaction.
    
    Returns:
        tuple: (modèle, dictionnaire des catégories) ou (None, None) en cas d'erreur
    """
    with st.spinner('Chargement du modèle en cours. Cela peut prendre quelques instants...'):
        try:
            # Test de connexion à Hugging Face
            st.info("Test de connexion à Hugging Face...")
            test_result = test_hugging_face_connection()
            if not test_result:
                st.warning("La connexion à Hugging Face a échoué, mais nous allons essayer de charger le modèle quand même.")

            # Obtention des chemins
            paths = get_model_paths()
            st.write("Chemins recherchés pour le modèle:", paths)

            # Mesure du temps de chargement du modèle
            start_time = time.time()

            # Utilisation de la fonction load_efficientnet_transformer_model qui redirige vers ConvNeXtTiny
            st.info("Chargement du modèle ConvNeXtTiny depuis Hugging Face...")
            model = load_efficientnet_transformer_model()

            end_time = time.time()
            loading_time = end_time - start_time

            if model is None:
                st.error("Le modèle n'a pas pu être chargé correctement")

                # Vérifier si le token est présent
                hf_token = get_hugging_face_token()
                if not hf_token:
                    st.error("Aucun token Hugging Face trouvé. Veuillez l'ajouter dans les secrets Streamlit.")
                    st.info("Pour ajouter un token Hugging Face, allez dans les paramètres de votre application Streamlit Cloud, puis dans l'onglet 'Secrets' et ajoutez: HF_TOKEN = 'votre_token'")

                return None, None

            # Vérification du modèle chargé
            st.write("### Vérification du modèle chargé")
            # Afficher les informations dans un format bien organisé
            model_info = {
                "Nombre de couches": len(model.layers),
                "Format de sortie": str(model.output_shape),
                "Temps de chargement": f"{loading_time:.2f} secondes"
            }
            
            for label, value in model_info.items():
                st.markdown(f"**{label}:** {value}")

            # Afficher le type de modèle chargé
            import tensorflow as tf
            model_size = sum([tf.keras.backend.count_params(w) for w in model.weights])
            st.write(f"**Taille du modèle:** {model_size/1000000:.2f} millions de paramètres")

            if model_size < 10_000_000:  # 10M paramètres
                st.success("✅ Modèle ConvNeXtTiny chargé avec succès")
            else:
                st.info("✅ Modèle standard chargé avec succès")

            categories = load_categories()
            st.success("Modèle et catégories chargés avec succès!")
            return model, categories

        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle")
            with st.expander("Détails techniques de l'erreur"):
                st.error(f"Description détaillée: {str(e)}")
                st.code(traceback.format_exc())

            # Vérifier si le token est présent
            hf_token = get_hugging_face_token()
            if not hf_token:
                st.error("Aucun token Hugging Face trouvé. Veuillez l'ajouter dans les secrets Streamlit.")
                st.info("Pour ajouter un token Hugging Face, allez dans les paramètres de votre application Streamlit Cloud, puis dans l'onglet 'Secrets' et ajoutez: HF_TOKEN = 'votre_token'")

            return None, None

# Chargement du modèle
model, categories = load_model()

# Interface pour sélectionner entre upload et exemples
st.header("Sélection de l'image")
source_option = st.radio(
    "Comment souhaitez-vous fournir une image?",
    ["Télécharger une image", "Utiliser un exemple"],
    help="Choisissez si vous voulez télécharger votre propre image ou utiliser une image d'exemple"
)

if source_option == "Télécharger une image":
    # Interface utilisateur pour l'upload d'images
    uploaded_file = st.file_uploader(
        "Choisissez une image à classifier (JPG, JPEG ou PNG)",
        type=["jpg", "jpeg", "png"],
        help="L'image sera analysée par le modèle de classification"
    )
    
    if uploaded_file is not None:
        # Traitement de l'image téléchargée
        try:
            image = Image.open(uploaded_file)
            # Afficher l'image avec une description accessible
            st.image(
                image, 
                caption=f"Image téléchargée: {uploaded_file.name}", 
                use_column_width=True,
                output_format="PNG"  # Format stable pour l'affichage
            )
            
            # Prédiction
            if model is not None and categories is not None:
                with st.spinner("Classification en cours... Veuillez patienter."):
                    predictions = predict_image(model, image, categories)
                    
                    # Afficher les résultats
                    st.header("Résultats de la classification")
                    
                    # Afficher un tableau de résultats textuels pour accessibilité
                    st.markdown("### Probabilités par catégorie")
                    results_table = []
                    
                    # Trier les prédictions par probabilité décroissante
                    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    
                    for category, probability in sorted_preds:
                        results_table.append({
                            "Catégorie": category,
                            "Probabilité": f"{probability*100:.2f}%"
                        })
                    
                    st.table(results_table)
                    
                    # Visualisation graphique (avec texte alternatif)
                    st.markdown("### Visualisation graphique")
                    fig = plot_prediction_bars(predictions)
                    st.pyplot(fig)
                    
                    # Ajouter une description textuelle du résultat principal pour accessibilité
                    top_category = sorted_preds[0][0]
                    top_probability = sorted_preds[0][1]
                    st.markdown(f"""
                    **Résultat principal:** L'image est classifiée comme **{top_category}** 
                    avec une probabilité de **{top_probability*100:.2f}%**.
                    """)
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image")
            with st.expander("Détails techniques de l'erreur"):
                st.error(f"Description: {str(e)}")
                st.code(traceback.format_exc())
else:
    # Chargement et affichage des exemples
    examples = load_example_images()
    
    if examples:
        # Sélection d'un exemple avec descriptions accessibles
        example_options = [os.path.basename(ex) for ex in examples]
        selected_example_name = st.selectbox(
            "Sélectionnez une image exemple",
            example_options,
            help="Choisissez parmi les images d'exemple disponibles"
        )
        
        # Retrouver le chemin complet
        selected_example = next((ex for ex in examples if os.path.basename(ex) == selected_example_name), None)
        
        if selected_example:
            try:
                image = Image.open(selected_example)
                # Afficher l'image avec une description accessible
                st.image(
                    image, 
                    caption=f"Exemple: {os.path.basename(selected_example)}", 
                    use_column_width=True,
                    output_format="PNG"
                )
                
                # Prédiction sur l'exemple
                if model is not None and categories is not None:
                    with st.spinner("Classification en cours... Veuillez patienter."):
                        predictions = predict_image(model, image, categories)
                        
                        # Afficher les résultats
                        st.header("Résultats de la classification")
                        
                        # Afficher un tableau de résultats textuels pour accessibilité
                        st.markdown("### Probabilités par catégorie")
                        results_table = []
                        
                        # Trier les prédictions par probabilité décroissante
                        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        
                        for category, probability in sorted_preds:
                            results_table.append({
                                "Catégorie": category,
                                "Probabilité": f"{probability*100:.2f}%"
                            })
                        
                        st.table(results_table)
                        
                        # Visualisation graphique (avec texte alternatif)
                        st.markdown("### Visualisation graphique")
                        fig = plot_prediction_bars(predictions)
                        st.pyplot(fig)
                        
                        # Ajouter une description textuelle du résultat principal pour accessibilité
                        top_category = sorted_preds[0][0]
                        top_probability = sorted_preds[0][1]
                        st.markdown(f"""
                        **Résultat principal:** L'image est classifiée comme **{top_category}** 
                        avec une probabilité de **{top_probability*100:.2f}%**.
                        """)
            except Exception as e:
                st.error(f"Erreur lors du traitement de l'image exemple")
                with st.expander("Détails techniques de l'erreur"):
                    st.error(f"Description: {str(e)}")
                    st.code(traceback.format_exc())
    else:
        st.error("Aucun exemple d'image n'a été trouvé. Veuillez télécharger votre propre image.")
        with st.expander("Informations de dépannage"):
            st.write("Répertoire de travail:", os.getcwd())
            st.write("Contenu du répertoire:", os.listdir("."))
            if os.path.exists("assets"):
                st.write("Contenu du répertoire assets:", os.listdir("assets"))

# Pied de page avec informations complémentaires
st.markdown("---")
st.markdown("""
### À propos de cette application
Cette application de classification d'images utilise un modèle ConvNeXtTiny entraîné sur un dataset Flipkart.
Pour plus d'informations, consultez [le modèle sur Hugging Face](https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification).

Pour signaler des problèmes ou contribuer, contactez l'auteur.
""")
