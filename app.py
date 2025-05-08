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

st.title("🦜 Classification d'Images avec ConvNeXtTiny")
st.markdown("""
Cette application utilise un modèle ConvNeXtTiny pour classifier les images.
Téléchargez une image ou utilisez un exemple pour voir le modèle en action! Mourad JABRI 2025
""")

@st.cache_resource
def load_model():
    with st.spinner('Chargement du modèle... Cela peut prendre quelques instants.'):
        try:
            paths = get_model_paths()
            st.write("Chemins recherchés:", paths)

            # Utilisation de la fonction load_efficientnet_transformer_model qui redirige maintenant vers ConvNeXtTiny
            model = load_efficientnet_transformer_model()

            if model is None:
                st.error("Le modèle n'a pas pu être chargé (retourné None)")
                return None, None

            # Vérification du modèle chargé
            st.write("### Vérification du modèle chargé")
            st.write(f"Nombre de couches: {len(model.layers)}")
            st.write(f"Shape de sortie: {model.output_shape}")

            # Afficher le type de modèle chargé
            import tensorflow as tf
            model_size = sum([tf.keras.backend.count_params(w) for w in model.weights])
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

try:
    model, categories = load_model()
    if model is None:
        st.error("Impossible de charger le modèle. Veuillez vérifier que le modèle est disponible.")
        st.info("Informations de débogage:")
        paths = get_model_paths()
        for key, path in paths.items():
            st.write(f"{key}: {path}")
            st.write(f"  - Existe: {os.path.exists(path)}")
        st.stop()
    else:
        st.write(f"Shape de sortie du modèle: {model.output_shape}")
        st.write(f"Nombre de couches: {len(model.layers)}")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

with st.sidebar:
    st.header("📝 À propos")
    st.markdown("""
    Cette application utilise un modèle ConvNeXtTiny
    pour classifier les images parmi 7 catégories (d'après les données Flipkart).
    """)

    st.header("🔍 Comment utiliser")
    st.markdown("""
    1. Téléchargez une image
    2. Ou utilisez un exemple fourni
    3. Observez le résultat de la classification
    """)

    show_preprocessing = st.checkbox("Montrer les étapes de prétraitement", value=False)
    augmentation_options = st.expander("Options d'augmentation (pour tester)")
    with augmentation_options:
        apply_aug = st.checkbox("Appliquer une augmentation", value=False)
        aug_type = st.selectbox(
            "Type d'augmentation",
            ["rotation", "flip", "brightness", "contrast", "color"],
            disabled=not apply_aug
        )

st.header("📤 Téléchargez votre image")
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

examples_dir = os.path.join("assets", "examples")
example_files = []
image_mapping = {}

if os.path.exists(examples_dir):
    mapping_path = os.path.join(examples_dir, "image_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            image_mapping = json.load(f)

    all_files = os.listdir(examples_dir)
    example_files = []
    for f in all_files:
        if (f.endswith('.jpg') and
            len(f) == 36 and
            all(c in '0123456789abcdef' for c in f[:-4]) and
            os.path.exists(os.path.join(examples_dir, f))):
            example_files.append(f)

    if image_mapping:
        example_files.sort(key=lambda x: image_mapping.get(x, {}).get('category', 'z'))

if example_files:
    st.header("🖼️ Ou utilisez un exemple")
    st.markdown("Cliquez sur un ID pour utiliser l'image correspondante :")
    cols = st.columns(3)
    for i, example_id in enumerate(example_files[:12]):
        with cols[i % 3]:
            if st.button(example_id, key=f"btn_{i}", help="Cliquer pour utiliser cette image"):
                uploaded_file = os.path.join(examples_dir, example_id)

image_to_process = None

if uploaded_file is not None:
    st.header("🔍 Résultat de la Classification")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image analysée")
        image = Image.open(uploaded_file)
        st.image(image, caption="Image sélectionnée", use_column_width=True)

        if show_preprocessing:
            st.subheader("Prétraitement")
            preproc_col1, preproc_col2 = st.columns(2)
            with preproc_col1:
                st.markdown("**Image originale**")
                st.image(image, caption="Avant prétraitement", width=200)

            preprocessed_image = resize_and_pad_image(image, target_size=(224, 224))  # Taille pour ConvNeXtTiny
            if apply_aug:
                preprocessed_image = apply_data_augmentation(preprocessed_image, aug_type)

            with preproc_col2:
                st.markdown("**Image prétraitée**")
                st.image(preprocessed_image, caption="Après prétraitement (224x224)", width=200)

            image_to_process = preprocessed_image
        else:
            image_to_process = resize_and_pad_image(image, target_size=(224, 224))  # Taille pour ConvNeXtTiny
            if apply_aug:
                image_to_process = apply_data_augmentation(image_to_process, aug_type)

    with col2:
        st.subheader("Résultats")
        with st.spinner('Analyse en cours...'):
            try:
                results = predict_image(model, image_to_process, categories)
                if "error" in results:
                    st.error(f"Erreur lors de la prédiction: {results['error']}")
                else:
                    st.success(f"**Catégorie prédite:** {results['predicted_class']}")
                    st.info(f"**Confiance:** {results['confidence']*100:.2f}%")
                    st.info(f"**Temps d'inférence:** {results['inference_time']*1000:.2f} ms")

                    if isinstance(uploaded_file, str) and image_mapping and uploaded_file is not None:
                        filename = os.path.basename(uploaded_file)
                        img_info = image_mapping.get(filename, {})
                        true_category = img_info.get('category', None)
                        if true_category:
                            st.markdown("---")
                            st.markdown("### Évaluation de la prédiction")
                            st.info(f"**Catégorie réelle:** {true_category}")
                            if results['predicted_class'] == true_category:
                                st.success("✅ **Prédiction correcte!**")
                            else:
                                st.error("❌ **Prédiction incorrecte**")
                                st.write(f"Prédit: {results['predicted_class']} | Réel: {true_category}")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")
                st.code(traceback.format_exc())

    if uploaded_file is not None and 'results' in locals() and not results.get("error"):
        st.header("📊 Probabilités pour chaque classe")
        try:
            fig = plot_prediction_bars(results["all_predictions"], title="Probabilités par classe")
            st.pyplot(fig)
            with st.expander("Voir les probabilités détaillées"):
                for pred in results["all_predictions"]:
                    st.write(f"**{pred['class_name']}**")
                    st.progress(float(pred['probability']))
                    st.write(f"{pred['probability']*100:.2f}%")
                    st.write("---")
        except Exception as e:
            st.error(f"Erreur lors de la création de la visualisation: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Développé avec ❤️ pour le projet P9 OpenClassrooms - Classification d'images</p>
    <p>Modèle ConvNeXtTiny entraîné sur des données Flipkart avec 7 catégories de produits</p>
</div>
""", unsafe_allow_html=True)
