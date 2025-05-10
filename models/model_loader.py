import os
import json
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import time
from PIL import Image
import streamlit as st  # Ajout de l'import streamlit

# Importer depuis utils.preprocessing au lieu d'autres modules
from utils.preprocessing import preprocess_image_for_convnext, resize_and_pad_image

# Configuration pour permettre la désérialisation des couches Lambda
tf.keras.config.enable_unsafe_deserialization()

# URL du modèle sur Hugging Face
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final.keras"

def get_hugging_face_token():
    """
    Récupère le token d'authentification Hugging Face depuis les secrets Streamlit
    ou une variable d'environnement.
    
    Returns:
        str: Le token d'authentification ou une chaîne vide si non trouvé
    """
    # Essayer d'obtenir le token depuis les secrets Streamlit
    try:
        return st.secrets.get("HF_TOKEN", "")
    except:
        # Si ce n'est pas possible, essayer depuis les variables d'environnement
        return os.environ.get("HF_TOKEN", "")

def get_model_paths():
    """
    Obtient les chemins vers les fichiers du modèle.
    
    Returns:
        Dictionnaire contenant les chemins des fichiers
    """
    # Répertoire racine du projet
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Répertoire pour les modèles
    models_dir = os.path.join(root_dir, "models", "saved")
    os.makedirs(models_dir, exist_ok=True)
    
    # Fichier du modèle ConvNeXt
    convnext_model_path = os.path.join(models_dir, "model_convnext_tiny.keras")
    
    # Mapping des catégories
    category_mapping = os.path.join(models_dir, "category_mapping.json")
    
    # Retourner les chemins
    return {
        "models_dir": models_dir,
        "convnext_model": convnext_model_path,
        "category_mapping": category_mapping
    }

def load_model_from_huggingface():
    """
    Charge le modèle depuis Hugging Face.
    
    Returns:
        Modèle Keras chargé
    """
    try:
        # Obtenir les chemins
        paths = get_model_paths()
        model_path = paths["convnext_model"]
        
        # Si le modèle existe déjà localement, le charger
        if os.path.exists(model_path):
            print(f"Chargement du modèle local depuis {model_path}")
            return load_model(model_path)
        
        # Sinon, télécharger le modèle depuis Hugging Face
        print(f"Téléchargement du modèle depuis Hugging Face: {HF_MODEL_URL}")
        
        # Créer un fichier temporaire pour le téléchargement
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Obtenir le token d'authentification
        hf_token = get_hugging_face_token()
        
        # Préparer les headers avec le token si disponible
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            print("Token d'authentification Hugging Face trouvé et utilisé")
        else:
            print("Aucun token d'authentification Hugging Face trouvé")
        
        # Télécharger le modèle avec authentification si nécessaire
        response = requests.get(HF_MODEL_URL, headers=headers, stream=True, timeout=300)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        
        # Enregistrer le modèle dans le fichier temporaire
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Charger le modèle
        model = load_model(temp_path)
        
        # Sauvegarder le modèle localement pour une utilisation future
        model.save(model_path)
        
        # Supprimer le fichier temporaire
        os.unlink(temp_path)
        
        return model
    
    except Exception as e:
        import traceback
        print(f"Erreur lors du chargement du modèle: {e}")
        print(traceback.format_exc())
        
        # Ajouter plus de détails sur l'erreur si on utilise Streamlit
        try:
            if 'response' in locals() and response is not None:
                print(f"Code d'état HTTP: {response.status_code}")
                print(f"En-têtes de réponse: {dict(response.headers)}")
                print(f"Début du contenu de la réponse: {response.text[:500] if hasattr(response, 'text') else 'Non disponible'}")
        except:
            pass
            
        return None

def load_efficientnet_transformer_model():
    """
    Charge le modèle EfficientNet-Transformer (maintenu pour compatibilité,
    mais charge réellement ConvNeXtTiny).

    Returns:
        Modèle Keras chargé
    """
    print("Chargement du modèle ConvNeXtTiny...")
    model = load_model_from_huggingface()

    if model is None:
        print("Impossible de charger le modèle. Vérifiez la connexion et les chemins.")
    else:
        print("Modèle chargé avec succès!")

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
    model = load_efficientnet_transformer_model()
    categories = load_categories()
    print(f"Catégories: {categories}")

    if model:
        print(f"Modèle chargé avec succès. Nombre de couches: {len(model.layers)}")
