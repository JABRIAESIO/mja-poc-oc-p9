import os
import json
import requests
import tempfile
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Embedding
import tensorflow.keras.backend as K


# Configuration pour permettre la désérialisation des couches Lambda
tf.keras.config.enable_unsafe_deserialization()

# URL du modèle sur Hugging Face (à mettre à jour avec votre nouveau modèle ConvNeXtTiny)
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final.keras"
# add catégories. ??
 
# Classes personnalisées du modèle précédent (conservées pour compatibilité)
class ReshapeLayer(layers.Layer):
    """Couche personnalisée pour reshape"""
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.reshape(inputs, [batch_size] + list(self.target_shape))

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({'target_shape': self.target_shape})
        return config

class CLSTokenLayer(layers.Layer):
    """Couche personnalisée pour ajouter un CLS token"""
    def __init__(self, feature_dim, **kwargs):
        super(CLSTokenLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.feature_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        super(CLSTokenLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, inputs], axis=1)

    def get_config(self):
        config = super(CLSTokenLayer, self).get_config()
        config.update({'feature_dim': self.feature_dim})
        return config

class ExtractCLSLayer(layers.Layer):
    """Couche personnalisée pour extraire le CLS token"""
    def call(self, inputs):
        return inputs[:, 0, :]

    def get_config(self):
        return super(ExtractCLSLayer, self).get_config()

class PositionalEmbeddingLayer(layers.Layer):
    """Couche personnalisée pour ajouter des embeddings positionnels"""
    def __init__(self, seq_length, feature_dim, **kwargs):
        super(PositionalEmbeddingLayer, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.embedding = Embedding(input_dim=seq_length, output_dim=feature_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        positions = tf.range(start=0, limit=self.seq_length, delta=1)
        pos_embedding = self.embedding(positions)
        pos_embedding = tf.expand_dims(pos_embedding, axis=0)
        pos_embedding = tf.tile(pos_embedding, [batch_size, 1, 1])
        return inputs + pos_embedding

    def get_config(self):
        config = super(PositionalEmbeddingLayer, self).get_config()
        config.update({
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim
        })
        return config


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

def get_model_paths(base_dir=None):
    """
    Détermine les chemins vers les fichiers du modèle.
    """
    if base_dir is None:
        base_dir = "/data/OC/P9"

    if not os.path.exists(base_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Répertoire des artéfacts du 07/05/2025
    output_dir = os.path.join(base_dir, "convnext-output_07052025_1735")
    # Important: ajouter le sous-dossier "models" pour les fichiers du modèle
    models_subdir = os.path.join(output_dir, "models")

    paths = {
        "base_dir": base_dir,
        "model_dir": os.path.join(base_dir, "models"),
        "output_dir": output_dir,
        "models_subdir": models_subdir,
        # Chemin correct avec le sous-dossier "models"
        "best_model_weights": os.path.join(models_subdir, "model_final.keras"),
        "model_architecture": os.path.join(models_subdir, "model_architecture.json"),
        "simple_model": os.path.join(models_subdir, "model_final.keras"),
        "simplified_model": os.path.join(models_subdir, "model_final.keras"),
        "examples_dir": os.path.join(base_dir, "mja-poc-oc-p9", "assets", "examples"),
        "category_mapping": os.path.join(models_subdir, "categories.json")
    }

    # Vérifier et afficher l'existence des fichiers
    print("Vérification des chemins de modèle:")
    for key, path in paths.items():
        exists = os.path.exists(path)
        print(f"  {key}: {path}")
        print(f"    - Existe: {exists}")

    return paths

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

def load_categories(mapping_path=None):
    """
    Charge les catégories depuis le fichier de mapping ou utilise les catégories par défaut.
    """
    if mapping_path is None:
        paths = get_model_paths()
        mapping_path = paths["category_mapping"]

    try:
        if os.path.exists(mapping_path):
            print(f"Chargement des catégories depuis: {mapping_path}")
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
            
            # Format direct des catégories pour ConvNeXtTiny
            if isinstance(mapping_data, dict) and all(k.isdigit() for k in mapping_data.keys()):
                # Convertir les clés en entiers
                categories = {int(k): v for k, v in mapping_data.items()}
                print(f"Catégories chargées directement: {categories}")
                return categories
            # Format avec clé 'categories'
            elif "categories" in mapping_data:
                categories = {int(idx): name for idx, name in mapping_data['categories'].items()}
                print(f"Catégories chargées depuis la clé 'categories': {categories}")
                return categories
            # Format inverse avec 'category_mapping'
            elif "category_mapping" in mapping_data:
                category_mapping = mapping_data["category_mapping"]
                categories = {v: k for k, v in category_mapping.items()}
                print(f"Catégories chargées depuis le mapping inversé: {categories}")
                return categories
    except Exception as e:
        print(f"Erreur lors du chargement des catégories depuis le fichier: {e}")

    print("Utilisation des catégories par défaut (7 catégories Flipkart)")
    categories = get_default_categories()
    print(f"Catégories chargées: {categories}")
    return categories

def download_model_from_hf():
    """
    Télécharge le modèle depuis Hugging Face.
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

def load_simplified_model(model_path=None):
    """
    Charge le modèle simplifié (maintenant ConvNeXtTiny).

    Args:
        model_path: Chemin vers le modèle simplifié

    Returns:
        Le modèle chargé ou None en cas d'erreur
    """
    cloud_env = is_cloud_environment()

    # Si on est dans le cloud ou si le chemin local n'existe pas
    if cloud_env or not os.path.exists("/data/OC/P9"):
        print("Environnement cloud détecté. Téléchargement du modèle simplifié depuis Hugging Face...")
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
            os.unlink(model_path)
            return None
    else:
        # Chargement local
        paths = get_model_paths()
        if model_path is None:
            model_path = paths["simplified_model"]

        if not os.path.exists(model_path):
            print(f"ERREUR: Modèle simplifié non trouvé: {model_path}")
            return None

        try:
            print(f"Chargement du modèle ConvNeXtTiny depuis: {model_path}")
            model = load_model(model_path)
            print("Modèle ConvNeXtTiny chargé avec succès!")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle simplifié: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_lambda_identity(x):
    """Fonction Lambda identité qui retourne simplement son entrée."""
    return x

def create_lambda_reshape(output_shape):
    """Crée une fonction Lambda de reshape avec une forme spécifiée."""
    def reshape_func(x):
        # Si output_shape est (49, 1280), reshape depuis (batch, 7, 7, 1280) vers (batch, 49, 1280)
        if output_shape == (49, 1280):
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1]
            width = tf.shape(x)[2]
            channels = tf.shape(x)[3]
            return tf.reshape(x, [batch_size, height * width, channels])
        else:
            return x
    return reshape_func

def load_efficientnet_transformer_model(model_architecture_path=None, model_weights_path=None, use_simple_model=False):
    """
    Charge le modèle EfficientNet-Transformer.
    Cette fonction est maintenue pour compatibilité mais utilise maintenant le modèle ConvNeXtTiny.
    """
    print("ATTENTION: L'API a changé - Chargement du nouveau modèle ConvNeXtTiny à la place d'EfficientNet-Transformer")
    return load_convnext_model(model_weights_path)

def load_convnext_model(model_path=None):
    """
    Charge le modèle ConvNeXtTiny.

    Args:
        model_path: Chemin vers le modèle ConvNeXtTiny

    Returns:
        Le modèle chargé ou None en cas d'erreur
    """
    cloud_env = is_cloud_environment()

    # Si on est dans le cloud ou si le chemin local n'existe pas
    if cloud_env or not os.path.exists("/data/OC/P9"):
        print("Environnement cloud détecté. Téléchargement du modèle ConvNeXtTiny depuis Hugging Face...")
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
            os.unlink(model_path)
            return None
    else:
        # Chargement local
        paths = get_model_paths()
        if model_path is None:
            model_path = paths["best_model_weights"]

        if not os.path.exists(model_path):
            print(f"ERREUR: Modèle ConvNeXtTiny non trouvé: {model_path}")
            return None

        try:
            print(f"Chargement du modèle ConvNeXtTiny depuis: {model_path}")
            model = load_model(model_path)
            print("Modèle ConvNeXtTiny chargé avec succès!")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle ConvNeXtTiny: {e}")
            import traceback
            traceback.print_exc()
            return None

def get_model_summary(model):
    """
    Génère un résumé du modèle.
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
