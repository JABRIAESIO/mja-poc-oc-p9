import streamlit as st
from PIL import Image
import time
import numpy as np
import os

# Configurer la page Streamlit
st.set_page_config(
    page_title="Classifieur d'Images - Flipkart",
    page_icon="🛒",
    layout="wide"
)

# Afficher le titre et la description même avant le chargement du modèle
st.title("🛒 Classifieur d'Images - Flipkart")
st.markdown("""
Cette application permet de classer des images de produits en différentes catégories.
Chargez une image et le modèle détectera la catégorie du produit.
""")

# Créer un placeholder pour les messages de chargement
loading_placeholder = st.empty()

# Importer les fonctions de model_loader avec le placeholder
# Cela permet d'afficher l'interface pendant le chargement du modèle
from models.model_loader import load_efficientnet_transformer_model, load_categories, preprocess_image_for_convnext

# Charger les catégories
categories = load_categories()

# Charger le modèle avec affichage de progression
loading_placeholder.info("Initialisation du modèle en cours... Cela peut prendre quelques minutes.")
model = load_efficientnet_transformer_model(loading_placeholder)

# Fonction de prédiction
def predict_image(image):
    try:
        # Prétraiter l'image
        img_array = preprocess_image_for_convnext(image)
        
        # Faire la prédiction
        prediction = model.predict(img_array)
        
        # Obtenir la classe prédite
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Obtenir le nom de la catégorie
        category_name = categories.get(predicted_class, f"Catégorie {predicted_class}")
        
        return {
            "category_id": predicted_class,
            "category_name": category_name,
            "confidence": confidence
        }
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

# Interface utilisateur pour le téléchargement d'images
st.header("📤 Téléchargez une image de produit")
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

# Si le modèle est chargé et qu'une image est téléchargée
if model is not None and uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Prédiction
    with col2:
        st.subheader("Résultat de la classification")
        with st.spinner("Classification en cours..."):
            result = predict_image(image)
        
        if result:
            st.success(f"Catégorie prédite: **{result['category_name']}**")
            st.progress(result["confidence"])
            st.info(f"Confiance: {result['confidence']*100:.2f}%")
elif model is None:
    st.warning("Le modèle n'a pas pu être chargé. Veuillez réessayer plus tard.")

# Afficher des informations sur le modèle
st.sidebar.header("À propos du modèle")
st.sidebar.markdown("""
Ce modèle utilise une architecture ConvNeXt Tiny fine-tunée sur un dataset 
de produits Flipkart pour classifier les images en 7 catégories.
""")

st.sidebar.subheader("Catégories supportées")
for cat_id, cat_name in categories.items():
    st.sidebar.write(f"- {cat_name}")

# Afficher des informations sur l'application
st.sidebar.header("À propos de l'application")
st.sidebar.markdown("""
Cette application a été développée dans le cadre du projet 9 
de la formation OpenClassrooms "Machine Learning Engineer".
""")
