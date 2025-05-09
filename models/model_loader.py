import os
import json
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Configuration pour permettre la désérialisation des couches Lambda
tf.keras.config.enable_unsafe_deserialization()

# URL du modèle sur Hugging Face
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final.keras"

def is_cloud_environment():
    """
    Détecte si l'application s'exécute dans un environnement cloud.
    """
    cloud_indicators = [
        os.environ.get('STREAMLIT_SHARING', ''),
        os.environ.get('RAILWAY_STATIC_URL', ''),
        os.environ.get('HEROKU_APP_ID', ''),
        os.path.exists('/.dockerenv'),
    ]
    return any(cloud_indicators)

def get_default_categories():
    """Retourne les catégories correctes de Flipkart alignées avec le fichier categories.json."""
    return {
        0: "Baby Care",
        1: "Beauty and Personal Care",
        2: "Computers",
        3: "Home Decor & Festive Needs",
        4: "Home Furnishing",
        5: "Kitchen & Dining",
        6: "Watches"
    }

def download_model_from_hf():
    """
    Télécharge le modèle depuis Hugging Face.
    
    Returns:
        str: Chemin vers le fichier temporaire contenant le modèle téléchargé
        None: En cas d'erreur
    """
    try:
        print(f"Téléchargement du modèle depuis Hugging Face: {HF_MODEL_URL}")
        response = requests.get(HF_MODEL_URL, stream=True)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.keras')
        temp_path = temp_file.name

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            temp_file.write(chunk)
        temp_file.close()

        print(f"Modèle téléchargé avec succès vers {temp_path}")
        return temp_path
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle: {e}")
        return None

def load_simplified_model():
    """
    Charge le modèle ConvNeXtTiny depuis Hugging Face.

    Returns:
        Le modèle chargé ou None en cas d'erreur
    """
    print("Téléchargement du modèle ConvNeXtTiny depuis Hugging Face...")
    model_path = download_model_from_hf()
    
    if model_path is None:
        print("Échec du téléchargement du modèle depuis Hugging Face")
        return None
    
    try:
        model = load_model(model_path)
        print("Modèle ConvNeXtTiny chargé depuis le fichier téléchargé")
        os.unlink(model_path)  # Supprime le fichier temporaire
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle téléchargé: {e}")
        if os.path.exists(model_path):
            os.unlink(model_path)
        return None

def load_convnext_model():
    """
    Charge le modèle ConvNeXtTiny depuis Hugging Face.

    Returns:
        Le modèle chargé ou None en cas d'erreur
    """
    print("Téléchargement du modèle ConvNeXtTiny depuis Hugging Face...")
    model_path = download_model_from_hf()
    
    if model_path is None:
        print("Échec du téléchargement du modèle depuis Hugging Face")
        return None
    
    try:
        model = load_model(model_path)
        print("Modèle ConvNeXtTiny chargé depuis le fichier téléchargé")
        os.unlink(model_path)  # Supprime le fichier temporaire
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle téléchargé: {e}")
        if os.path.exists(model_path):
            os.unlink(model_path)
        return None

def load_categories():
    """
    Retourne les catégories par défaut pour le modèle.
    
    Returns:
        dict: Dictionnaire des catégories avec indices comme clés et noms comme valeurs
    """
    print("Utilisation des catégories par défaut (7 catégories Flipkart)")
    categories = get_default_categories()
    print(f"Catégories chargées: {categories}")
    return categories

def get_model_summary(model):
    """
    Génère un résumé du modèle.
    
    Args:
        model: Modèle TensorFlow/Keras
        
    Returns:
        str: Résumé du modèle sous forme de chaîne de caractères
    """
    if model is None:
        return "Modèle non disponible"

    import io
    summary_string = io.StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    return summary_string.getvalue()

def preprocess_image_for_convnext(image, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence avec ConvNeXtTiny.

    Args:
        image: Image PIL à prétraiter
        target_size: Taille cible pour le redimensionnement

    Returns:
        Image prétraitée sous forme de tableau NumPy
    """
    import numpy as np
    from PIL import Image

    # Redimensionner l'image
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)

    # Convertir en tableau NumPy
    img_array = np.array(image).astype(np.float32)

    # Normaliser l'image à [-1, 1] comme dans le script d'entraînement ConvNeXtTiny
    img_array = img_array / 127.5 - 1.0

    # Ajouter la dimension de batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
