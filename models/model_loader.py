import os
import json
import requests
import tempfile
import tensorflow as tf
import keras
from keras.models import load_model
import keras.backend as K
import numpy as np
import time
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download

# Utilise TensorFlow comme backend pour Keras 3
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Importer depuis utils.preprocessing au lieu d'autres modules
from utils.preprocessing import preprocess_image_for_convnext, resize_and_pad_image

# Configuration du modèle
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final_fixed.keras"
MODEL_FILENAME = "model_final_fixed.keras"
REPO_ID = "mourad42008/convnext-tiny-flipkart-classification"

def get_hugging_face_token():
    """
    Récupère le token d'authentification Hugging Face depuis les secrets Streamlit
    ou une variable d'environnement.

    Returns:
        str: Le token d'authentification ou une chaîne vide si non trouvé
    """
    try:
        return st.secrets.get("HF_TOKEN", "")
    except:
        return os.environ.get("HF_TOKEN", "")

def get_model_paths():
    """
    Obtient les chemins vers les fichiers du modèle.

    Returns:
        Dictionnaire contenant les chemins des fichiers
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "models", "saved")
    os.makedirs(models_dir, exist_ok=True)
    
    convnext_model_path = os.path.join(models_dir, MODEL_FILENAME)
    category_mapping = os.path.join(models_dir, "category_mapping.json")
    
    return {
        "models_dir": models_dir,
        "convnext_model": convnext_model_path,
        "category_mapping": category_mapping
    }

# Variable globale pour stocker le placeholder de chargement
loading_placeholder = None

def set_loading_placeholder(placeholder):
    """
    Définit le placeholder pour afficher les messages de progression

    Args:
        placeholder: Élément Streamlit pour afficher les messages de progression
    """
    global loading_placeholder
    loading_placeholder = placeholder

def update_loading_status(message, status="info"):
    """
    Met à jour le statut de chargement dans l'interface Streamlit

    Args:
        message: Message à afficher
        status: Type de message ('info', 'success', 'error', 'warning')
    """
    global loading_placeholder
    if loading_placeholder is not None:
        if status == "info":
            loading_placeholder.info(message)
        elif status == "success":
            loading_placeholder.success(message)
        elif status == "error":
            loading_placeholder.error(message)
        elif status == "warning":
            loading_placeholder.warning(message)
    else:
        print(message)

@st.cache_resource(show_spinner=False)
def load_model_from_huggingface():
    """
    Charge le modèle depuis Hugging Face de façon simplifiée.
    
    Returns:
        Modèle Keras chargé ou None en cas d'erreur
    """
    try:
        # Vérifie d'abord si le modèle existe localement
        paths = get_model_paths()
        local_model_path = paths["convnext_model"]
        
        if os.path.exists(local_model_path):
            update_loading_status(f"Chargement du modèle local depuis {local_model_path}...")
            try:
                model = keras.models.load_model(local_model_path, compile=False)
                update_loading_status("Modèle local chargé avec succès!", "success")
                return model
            except Exception as e:
                update_loading_status(f"Erreur chargement local: {e}", "warning")
                update_loading_status("Tentative de téléchargement depuis Hugging Face...", "info")
        
        # Télécharge depuis Hugging Face
        update_loading_status("Téléchargement du modèle depuis Hugging Face...")
        
        # Obtenir le token si disponible
        hf_token = get_hugging_face_token()
        if hf_token:
            update_loading_status("Token d'authentification trouvé", "info")
        
        # Télécharger et charger directement le modèle
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            force_download=True,
            token=hf_token if hf_token else None
        )
        
        update_loading_status("Chargement du modèle téléchargé...")
        model = keras.models.load_model(model_path, compile=False)
        update_loading_status("Modèle chargé avec succès!", "success")
        
        # Sauvegarder localement pour une utilisation future
        try:
            update_loading_status("Sauvegarde du modèle localement...")
            model.save(local_model_path)
            update_loading_status("Modèle sauvegardé localement!", "success")
        except Exception as e:
            update_loading_status(f"Erreur sauvegarde locale: {e}", "warning")
        
        return model
        
    except Exception as e:
        update_loading_status(f"Erreur lors du chargement du modèle: {e}", "error")
        return None

def load_efficientnet_transformer_model(progress_placeholder=None):
    """
    Charge le modèle ConvNeXtTiny (nom maintenu pour compatibilité).

    Args:
        progress_placeholder: Un placeholder Streamlit pour afficher la progression

    Returns:
        Modèle Keras chargé
    """
    # Définir le placeholder de chargement si fourni
    if progress_placeholder is not None:
        set_loading_placeholder(progress_placeholder)

    update_loading_status("Chargement du modèle ConvNeXtTiny...", "info")
    
    model = load_model_from_huggingface()
    
    if model is None:
        update_loading_status("Impossible de charger le modèle. Vérifiez la connexion et les chemins.", "error")
    else:
        update_loading_status("Modèle chargé avec succès!", "success")
        # Informations sur le modèle
        if hasattr(model, 'layers'):
            update_loading_status(f"Nombre de couches: {len(model.layers)}", "info")
        update_loading_status(f"Type du modèle: {type(model)}", "info")

    return model

def load_categories():
    """
    Charge les catégories de classification.

    Returns:
        Dictionnaire des catégories {indice: nom}
    """
    paths = get_model_paths()
    category_file = paths["category_mapping"]

    # Si le fichier de mapping existe, le charger
    if os.path.exists(category_file):
        try:
            with open(category_file, 'r') as f:
                mapping_data = json.load(f)

            # Vérifier la structure du fichier
            if "categories" in mapping_data:
                return {int(k): v for k, v in mapping_data["categories"].items()}
        except Exception as e:
            print(f"Erreur lors du chargement des catégories: {e}")

    # Catégories par défaut
    default_categories = {
        0: "Baby Care",
        1: "Beauty and Personal Care",
        2: "Computers",
        3: "Home Decor & Festive Needs",
        4: "Home Furnishing",
        5: "Kitchen & Dining",
        6: "Watches"
    }

    # Sauvegarder les catégories par défaut pour une utilisation future
    try:
        os.makedirs(os.path.dirname(category_file), exist_ok=True)
        with open(category_file, 'w') as f:
            json.dump({"categories": default_categories}, f, indent=2)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des catégories: {e}")

    return default_categories

def preprocess_image(image, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence.
    Cette fonction est maintenue pour compatibilité, mais utilise maintenant
    preprocess_image_for_convnext en interne.

    Args:
        image: Image PIL
        target_size: Taille cible (défaut: 224x224)

    Returns:
        Tableau numpy normalisé prêt pour l'inférence
    """
    print("ATTENTION: preprocess_image est obsolète, utilisez preprocess_image_for_convnext à la place")
    return preprocess_image_for_convnext(image, target_size)

# Si exécuté directement, tester le chargement du modèle
if __name__ == "__main__":
    test_placeholder = None
    if 'st' in globals():
        test_placeholder = st.empty()

    model = load_efficientnet_transformer_model(test_placeholder)
    categories = load_categories()
    print(f"Catégories: {categories}")

    if model:
        print(f"Modèle chargé avec succès. Nombre de couches: {len(model.layers) if hasattr(model, 'layers') else 'N/A'}")
