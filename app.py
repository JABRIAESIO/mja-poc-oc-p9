import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import glob
import random
import traceback
import json
import requests
import sys
import platform
import psutil
import shutil  # AJOUT DEBUG - pour vérification espace disque

# Configuration essentielle AVANT tout autre code
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Configuration de la page
st.set_page_config(
    page_title="Classifieur d'Images - Flipkart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports après configuration
from models.model_loader import load_categories, get_model_paths, get_hugging_face_token, HF_MODEL_URL
from models.inference import predict_image, plot_prediction_bars
from utils.preprocessing import preprocess_image_for_convnext, resize_and_pad_image, apply_data_augmentation

# NOUVEAU : Import du module EDA
from eda_module import display_eda_mode

# Nouvelle fonction pour charger le modèle depuis Hugging Face
def load_model_from_huggingface():
    """Charge le modèle depuis Hugging Face avec gestion d'erreurs robuste"""
    try:
        # Import conditionnel pour éviter les erreurs
        from models.model_loader import load_efficientnet_transformer_model

        with st.spinner('Téléchargement du modèle depuis Hugging Face...'):
            model = load_efficientnet_transformer_model()
            return model
    except ImportError as e:
        st.error(f"Erreur d'import : {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

# CSS pour améliorer l'accessibilité
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
        outline-offset: 2px !important;
    }

    /* Améliorer la visibilité des boutons */
    .stButton button {
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        transition: background-color 0.3s ease !important;
    }

    /* Amélioration des boutons au hover */
    .stButton button:hover {
        background-color: #0056b3 !important;
        color: white !important;
    }

    /* Meilleur espacement pour les éléments de formulaire */
    .stRadio, .stSelectbox, .stFileUploader {
        margin-bottom: 1.5rem !important;
    }

    /* Messages d'erreur avec meilleur contraste */
    .stError, .element-container .stError {
        background-color: #ffd6d6 !important;
        border-left: 5px solid #ff4b4b !important;
        padding: 10px !important;
        margin-bottom: 1rem !important;
    }

    /* Messages de succès avec meilleur contraste */
    .stSuccess, .element-container .stSuccess {
        background-color: #d4edda !important;
        border-left: 5px solid #00cc88 !important;
        padding: 10px !important;
        margin-bottom: 1rem !important;
    }

    /* Messages d'info avec meilleur contraste */
    .stInfo, .element-container .stInfo {
        background-color: #e7f3ff !important;
        border-left: 5px solid #0068c9 !important;
        padding: 10px !important;
        margin-bottom: 1rem !important;
    }

    /* Amélioration de l'accessibilité des tableaux */
    .stTable table {
        border-collapse: collapse !important;
        width: 100% !important;
    }

    .stTable th, .stTable td {
        border: 1px solid #ddd !important;
        padding: 12px !important;
        text-align: left !important;
    }

    .stTable th {
        background-color: #f5f5f5 !important;
        font-weight: bold !important;
    }

    /* Amélioration de l'accessibilité du spinner */
    .stSpinner {
        color: #0068c9 !important;
    }

    /* Indicateur de progression accessible */
    .stProgress .st-bo {
        background-color: #00cc88 !important;
    }

    /* Radio buttons plus grands pour faciliter l'utilisation */
    .stRadio > div {
        gap: 0.5rem !important;
    }

    /* Skip navigation link (caché visuellement mais accessible aux lecteurs d'écran) */
    .skip-nav {
        position: absolute !important;
        top: -40px !important;
        left: 0 !important;
        background: #000 !important;
        color: #fff !important;
        padding: 8px !important;
        text-decoration: none !important;
        z-index: 1000 !important;
    }

    .skip-nav:focus {
        top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def test_hugging_face_connection():
    """Teste la connexion à Hugging Face en vérifiant l'accès au modèle."""
    try:
        hf_token = get_hugging_face_token()
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            st.info("🔑 Token d'authentification trouvé")
        else:
            st.warning("⚠️ Aucun token d'authentification trouvé")

        with st.spinner("Test de connexion en cours..."):
            response = requests.get(
                HF_MODEL_URL,
                headers=headers,
                stream=True,
                timeout=10,
                allow_redirects=True
            )

            # Lire juste un peu de contenu pour confirmer l'accès
            for chunk in response.iter_content(chunk_size=1024):
                break

            if response.status_code == 200:
                size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
                st.success(f"✅ Connexion réussie! Taille du modèle: {size_mb:.2f} MB")
                return True
            else:
                st.error(f"❌ Erreur HTTP {response.status_code}")
                return False
    except Exception as e:
        st.error(f"❌ Erreur de connexion: {str(e)}")
        return False

def load_example_images():
    """Charge les exemples d'images disponibles dans le dossier assets/examples"""
    examples_dir = os.path.join("assets", "examples")
    if not os.path.exists(examples_dir):
        st.warning(f"⚠️ Le dossier d'exemples '{examples_dir}' n'existe pas.")
        return []

    image_files = [f for f in os.listdir(examples_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        st.warning(f"⚠️ Aucune image trouvée dans le dossier '{examples_dir}'")
        return []

    return [os.path.join(examples_dir, f) for f in image_files]

# Fonction load_model() modifiée selon les recommandations
@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle avec une gestion robuste des erreurs"""
    try:
        with st.spinner('Chargement du modèle (2-3 minutes pour la première exécution)...'):
            # Obtention des chemins
            paths = get_model_paths()

            # AJOUT DEBUG - Vérification de l'espace disque
            total, used, free = shutil.disk_usage("/")
            st.write(f"Espace disque disponible : {free // (2**30)} Go")

            # AJOUT DEBUG - Vérification de la présence effective du fichier
            if os.path.exists(paths['convnext_model']):
                st.success(f"Fichier modèle trouvé : {paths['convnext_model']}")
                st.write(f"Taille : {os.path.getsize(paths['convnext_model']) / 1e6:.2f} MB")
            else:
                st.error("Fichier modèle absent !")

            model = load_model_from_huggingface()
            categories = load_categories()

            if model is None:
                st.error("""
                **Échec critique** : Le modèle n'a pas pu être chargé.
                Causes possibles :
                - Problème de connexion avec Hugging Face
                - Format de modèle incompatible
                - Espace disque insuffisant
                """)
                st.stop()

            return model, categories

    except Exception as e:
        st.error(f"Erreur irrécupérable : {str(e)}")
        st.stop()

def display_prediction_results(result):
    """Affiche les résultats de prédiction de manière organisée et accessible"""
    if "error" in result:
        st.error(f"Erreur lors de la prédiction: {result['error']}")
        return

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    all_predictions = result["all_predictions"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Résultat de la classification")

        # Utiliser des éléments sémantiques pour une meilleure accessibilité
        st.markdown("### Prédiction principale")
        st.success(f"**Catégorie prédite**: {predicted_class}")

        # Barre de progression avec label associé
        st.markdown(f"**Niveau de confiance**: {confidence*100:.2f}%")
        st.progress(confidence)

        # Tableau accessible avec headers explicites
        st.markdown("### Toutes les probabilités")
        results_table = []

        for pred in all_predictions:
            results_table.append({
                "Catégorie": pred["class_name"],
                "Probabilité": f"{pred['probability']*100:.2f}%"
            })

        # Tableau avec titre pour l'accessibilité
        st.markdown("Tableau des probabilités par catégorie :")
        st.table(results_table)

    with col2:
        st.markdown("### Visualisation graphique")

        # Texte alternatif pour le graphique
        prediction_dict = {pred["class_name"]: pred["probability"] for pred in all_predictions}

        # Description textuelle du graphique pour l'accessibilité
        st.markdown("**Description du graphique**: Graphique à barres horizontales montrant les probabilités de classification pour chaque catégorie.")

        fig = plot_prediction_bars(prediction_dict)

        # Alt text personnalisé pour le graphique
        st.pyplot(fig, use_container_width=True)

        # Description détaillée sous le graphique
        st.markdown(f"**Résumé du graphique**: La catégorie '{predicted_class}' a la probabilité la plus élevée ({confidence*100:.2f}%). Les autres catégories ont des probabilités inférieures.")

    # Informations sur les performances dans un expander
    with st.expander("Informations sur les performances"):
        st.markdown(f"""
        - **Temps de prétraitement**: {result['preprocess_time']*1000:.2f} ms
        - **Temps d'inférence**: {result['inference_time']*1000:.2f} ms
        - **Temps total**: {result['total_time']*1000:.2f} ms
        """)

def process_uploaded_image(uploaded_file, model, categories):
    """Traite une image téléchargée et affiche les résultats"""
    try:
        image = Image.open(uploaded_file)
        # Afficher l'image avec description alternative accessible
        st.image(
            image,
            caption=f"Image téléchargée: {uploaded_file.name}",
            width=400,
            output_format="PNG"
        )

        # Description alternative pour l'accessibilité
        st.markdown(f"**Description de l'image**: {uploaded_file.name} - Image téléchargée pour classification")

        if model is not None and categories is not None:
            with st.spinner("Classification en cours..."):
                result = predict_image(model, image, categories)
                display_prediction_results(result)
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image")
        with st.expander("Détails techniques de l'erreur"):
            st.error(f"Description: {str(e)}")
            st.code(traceback.format_exc())

def process_example_image(selected_example, model, categories):
    """Traite une image d'exemple et affiche les résultats"""
    try:
        image = Image.open(selected_example)
        st.image(
            image,
            caption=f"Exemple: {os.path.basename(selected_example)}",
            width=400,
            output_format="PNG"
        )

        # Description alternative pour l'accessibilité
        st.markdown(f"**Description de l'image**: {os.path.basename(selected_example)} - Image d'exemple pour classification")

        if model is not None and categories is not None:
            with st.spinner("Classification en cours..."):
                result = predict_image(model, image, categories)
                display_prediction_results(result)
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image exemple")
        with st.expander("Détails techniques de l'erreur"):
            st.error(f"Description: {str(e)}")
            st.code(traceback.format_exc())

def display_classification_interface(model, categories):
    """Affiche l'interface de classification (code existant séparé)"""
    # Interface pour sélectionner entre upload et exemples
    st.header("Sélection de l'image")

    # Utilisation de st.container pour une meilleure organisation
    with st.container():
        st.subheader("Mode de saisie")
        source_option = st.radio(
            "Comment souhaitez-vous fournir une image?",
            ["Télécharger une image", "Utiliser un exemple"],
            help="Choisissez si vous voulez télécharger votre propre image ou utiliser une image d'exemple"
        )

    if source_option == "Télécharger une image":
        uploaded_file = st.file_uploader(
            "Choisissez une image à classifier (JPG, JPEG ou PNG)",
            type=["jpg", "jpeg", "png"],
            help="L'image sera analysée par le modèle de classification"
        )

        if uploaded_file is not None:
            process_uploaded_image(uploaded_file, model, categories)
    else:
        examples = load_example_images()

        if examples:
            example_options = [os.path.basename(ex) for ex in examples]
            selected_example_name = st.selectbox(
                "Sélectionnez une image exemple",
                example_options,
                help="Choisissez parmi les images d'exemple disponibles"
            )

            selected_example = next((ex for ex in examples if os.path.basename(ex) == selected_example_name), None)

            if selected_example:
                process_example_image(selected_example, model, categories)
        else:
            st.warning("Aucun exemple d'image n'a été trouvé. Veuillez télécharger votre propre image.")

def main():
    # Ajouter un lien de navigation pour l'accessibilité
    st.markdown('<a href="#main-content" class="skip-nav">Aller au contenu principal</a>', unsafe_allow_html=True)

    # Sidebar avec informations système
    with st.sidebar:
        st.title("Informations système")

        system_info = {
            "Version Python": platform.python_version(),
            "Mémoire disponible": f"{psutil.virtual_memory().available / (1024 * 1024):.2f} MB",
            "Nombre de CPU": os.cpu_count(),
            "Répertoire de travail": os.getcwd()
        }

        for label, value in system_info.items():
            st.markdown(f"**{label}:** {value}")

        paths = get_model_paths()
        st.markdown("### Informations modèle")
        st.markdown(f"**URL Hugging Face:** {HF_MODEL_URL}")
        st.markdown(f"**Chemin local:** {paths['convnext_model']}")

        if st.button("Tester la connexion à Hugging Face"):
            test_hugging_face_connection()

        # NOUVEAU : Navigation principale dans la sidebar
        st.markdown("---")
        st.subheader("Navigation")
        app_mode = st.radio(
            "Choisir le mode :",
            ["🔮 Classification", "📊 Analyse des Données"],
            help="Mode Classification : classifiez vos images\nMode Analyse : explorez les données d'entraînement"
        )

    # Titre principal avec ancre pour l'accessibilité
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.title("🛒 Classifieur d'Images - Flipkart")
    st.markdown("""
    Cette application permet de classifier des images selon différentes catégories en utilisant un modèle ConvNeXtTiny.

    **Comment utiliser cette application:**
    1. Téléchargez une image de produit
    2. Le modèle analysera automatiquement l'image
    3. Les résultats de classification s'afficheront ci-dessous

    Développé dans le cadre du projet 9 de la formation OpenClassrooms "Machine Learning Engineer".
    """)

    # Chargement du modèle avec la nouvelle fonction
    model, categories = load_model()

    # NOUVEAU : Conditionner l'affichage selon le mode sélectionné
    if app_mode == "🔮 Classification":
        # Mode Classification (code existant)
        display_classification_interface(model, categories)
        
    elif app_mode == "📊 Analyse des Données":
        # NOUVEAU : Mode EDA
        display_eda_mode()

    # Pied de page
    st.markdown("---")
    st.markdown("""
    ### À propos de cette application
    Cette application de classification d'images utilise un modèle ConvNeXtTiny entraîné sur un dataset Flipkart.
    Elle a été développée dans le cadre du projet 9 de la formation OpenClassrooms "Machine Learning Engineer".

    Pour plus d'informations sur le modèle, consultez [Hugging Face](https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification).
    """)

if __name__ == "__main__":
    main()
