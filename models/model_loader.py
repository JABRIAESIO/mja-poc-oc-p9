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
import streamlit as st  # Ajout de l'import streamlit

# Utilise TensorFlow comme backend pour Keras 3
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Importer depuis utils.preprocessing au lieu d'autres modules
from utils.preprocessing import preprocess_image_for_convnext, resize_and_pad_image

# Configuration pour permettre la désérialisation des couches Lambda
keras.config.enable_unsafe_deserialization()

# URL du modèle sur Hugging Face
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final.h5"  # MODIFICATION 1: .keras → .h5
# Nom du fichier pour la sauvegarde locale - CORRIGÉ pour correspondre au fichier sur HF
MODEL_FILENAME = "model_final.h5"  # MODIFICATION 2: .savedmodel → .h5

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

    # Fichier du modèle - CORRIGÉ pour utiliser le nouveau nom adapté au format SavedModel
    convnext_model_path = os.path.join(models_dir, MODEL_FILENAME)

    # Mapping des catégories
    category_mapping = os.path.join(models_dir, "category_mapping.json")

    # Retourner les chemins
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
        print(message)  # Fallback à un print normal si placeholder non défini

@st.cache_resource(show_spinner=False)
def load_model_from_huggingface():
    """
    Charge le modèle depuis Hugging Face avec affichage de l'état de progression
    dans l'interface Streamlit.

    Returns:
        Modèle Keras chargé
    """
    try:
        # Obtenir les chemins
        paths = get_model_paths()
        model_path = paths["convnext_model"]

        # Si le modèle existe déjà localement, le charger
        if os.path.exists(model_path):
            update_loading_status(f"Chargement du modèle local depuis {model_path}...")
            try:
                # Pour un SavedModel, utiliser keras.models.load_model directement
                model = keras.models.load_model(model_path)
                update_loading_status("Modèle local chargé avec succès!", "success")
                return model
            except ValueError as e:
                update_loading_status(f"Erreur standard de chargement: {e}", "warning")
                update_loading_status("Tentative de chargement avec TFSMLayer...")
                try:
                    # Utiliser TFSMLayer pour charger un SavedModel format
                    from keras.layers import TFSMLayer
                    from keras.models import Sequential

                    # Créer un modèle qui charge le SavedModel en tant que couche
                    model = Sequential([
                        TFSMLayer(model_path, call_endpoint='serving_default')
                    ])
                    update_loading_status("Modèle chargé avec TFSMLayer!", "success")
                    return model
                except Exception as inner_e:
                    update_loading_status(f"Erreur avec TFSMLayer: {inner_e}", "error")

                    # Dernière tentative avec un chargement différent
                    try:
                        update_loading_status("Tentative de chargement direct avec tf.saved_model.load...", "info")
                        model = tf.saved_model.load(model_path)
                        update_loading_status("Modèle chargé avec tf.saved_model.load!", "success")
                        return model
                    except Exception as sm_e:
                        update_loading_status(f"Erreur avec tf.saved_model.load: {sm_e}", "error")
                        return None

        # Sinon, télécharger le modèle depuis Hugging Face
        update_loading_status(f"Téléchargement du modèle depuis Hugging Face...", "info")

        # Créer un fichier temporaire pour le téléchargement
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        # Obtenir le token d'authentification
        hf_token = get_hugging_face_token()

        # Préparer les headers avec le token si disponible
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            update_loading_status("Token d'authentification Hugging Face trouvé et utilisé", "info")
        else:
            update_loading_status("Aucun token d'authentification Hugging Face trouvé", "warning")

        # Télécharger le modèle avec authentification si nécessaire
        response = requests.get(
            HF_MODEL_URL,
            headers=headers,
            stream=True,
            timeout=300,
            allow_redirects=True  # Suivre les redirections automatiquement
        )

        # Afficher des informations de débogage
        update_loading_status(f"Code HTTP: {response.status_code}", "info")
        update_loading_status(f"URL finale après redirection: {response.url}", "info")

        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP

        # Enregistrer le modèle dans le fichier temporaire avec indication de progression
        update_loading_status("Téléchargement du fichier modèle (114 MB)...", "info")
        content_length = int(response.headers.get('Content-Length', 0)) or None
        if content_length:
            update_loading_status(f"Taille totale: {content_length/1024/1024:.1f} MB", "info")

        downloaded = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if content_length and downloaded % (5*1024*1024) == 0:  # Mise à jour tous les 5 MB
                    update_loading_status(f"Téléchargement en cours... {downloaded/1024/1024:.1f} MB / {content_length/1024/1024:.1f} MB", "info")

        update_loading_status("Téléchargement terminé. Chargement du modèle...", "info")

        # Vérifier si le fichier téléchargé est un SavedModel ou autre format
        try:
            # Essayer d'abord avec tf.saved_model.load pour détecter si c'est un SavedModel
            update_loading_status("Vérification du format du modèle...", "info")
            saved_model = tf.saved_model.contains_saved_model(temp_path)
            if saved_model:
                update_loading_status("Modèle détecté comme format SavedModel", "info")
                try:
                    model = tf.saved_model.load(temp_path)
                    update_loading_status("Modèle chargé avec tf.saved_model.load!", "success")
                except Exception as sm_e:
                    update_loading_status(f"Erreur avec tf.saved_model.load: {sm_e}", "error")
                    # Essayer avec une approche plus standard pour les SavedModel
                    try:
                        model = keras.models.load_model(temp_path)
                        update_loading_status("Modèle chargé avec keras.models.load_model!", "success")
                    except Exception as keras_e:
                        update_loading_status(f"Erreur avec keras.models.load_model: {keras_e}", "error")
                        return None
            else:
                # Si ce n'est pas un SavedModel, essayer avec les méthodes standard Keras
                update_loading_status("Modèle n'est pas un SavedModel, tentative avec load_model standard...", "info")
                try:
                    model = keras.models.load_model(temp_path)
                    update_loading_status("Modèle chargé avec keras.models.load_model standard!", "success")
                except Exception as keras_e:
                    update_loading_status(f"Erreur avec load_model standard: {keras_e}", "error")
                    return None
        except Exception as format_e:
            update_loading_status(f"Erreur lors de la vérification du format: {format_e}", "error")
            # Dernière tentative avec la méthode standard
            try:
                model = keras.models.load_model(temp_path)
                update_loading_status("Modèle chargé avec keras.models.load_model (dernière tentative)!", "success")
            except Exception as last_e:
                update_loading_status(f"Échec de toutes les tentatives de chargement: {last_e}", "error")
                return None

        # Sauvegarder le modèle localement pour une utilisation future
        try:
            update_loading_status(f"Sauvegarde du modèle vers {model_path}...", "info")
            # Utiliser keras.models.save pour un format compatible
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                tf.saved_model.save(model, model_path)
            update_loading_status("Modèle sauvegardé localement avec succès!", "success")
        except Exception as save_e:
            update_loading_status(f"Erreur lors de la sauvegarde du modèle: {save_e}", "warning")
            # Copier le fichier temporaire comme alternative
            import shutil
            try:
                update_loading_status("Tentative de copie du fichier temporaire...", "info")
                shutil.copy2(temp_path, model_path)
                update_loading_status("Fichier temporaire copié avec succès!", "success")
            except Exception as copy_e:
                update_loading_status(f"Erreur lors de la copie du fichier: {copy_e}", "error")

        # Supprimer le fichier temporaire
        try:
            os.unlink(temp_path)
        except:
            pass

        return model

    except Exception as e:
        import traceback
        update_loading_status(f"Erreur lors du chargement du modèle: {e}", "error")
        update_loading_status(traceback.format_exc(), "error")

        # Ajouter plus de détails sur l'erreur
        try:
            if 'response' in locals() and response is not None:
                update_loading_status(f"Code d'état HTTP: {response.status_code}", "error")
                update_loading_status(f"URL finale: {response.url}", "error")
        except:
            pass

        return None

def load_efficientnet_transformer_model(progress_placeholder=None):
    """
    Charge le modèle EfficientNet-Transformer (maintenu pour compatibilité,
    mais charge réellement ConvNeXtTiny).

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
    # Création d'un placeholder pour les tests
    test_placeholder = None
    if 'st' in globals():
        test_placeholder = st.empty()

    model = load_efficientnet_transformer_model(test_placeholder)
    categories = load_categories()
    print(f"Catégories: {categories}")

    if model:
        print(f"Modèle chargé avec succès. Nombre de couches: {len(model.layers) if hasattr(model, 'layers') else 'N/A (SavedModel format)'}")
