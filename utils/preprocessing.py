import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

def load_image(image_path):
    """
    Charge une image depuis un chemin et la convertit en format approprié.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Image PIL
    """
    try:
        image = Image.open(image_path)
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return None

def resize_and_pad_image(image, target_size=(224, 224), color=(0, 0, 0)):
    """
    Redimensionne une image en conservant ses proportions et en la remplissant si nécessaire.

    Args:
        image: Image PIL
        target_size: Taille cible (width, height)
        color: Couleur de remplissage pour le padding

    Returns:
        Image redimensionnée et rembourrée
    """
    if image is None:
        return None

    # Calculer le ratio pour préserver les proportions
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # Image plus large que la cible
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image plus haute que la cible
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)

    # Redimensionner l'image
    resized_img = image.resize((new_width, new_height), Image.LANCZOS)

    # Créer une nouvelle image avec la taille cible et la remplir avec la couleur spécifiée
    padded_img = Image.new('RGB', target_size, color)

    # Calculer la position pour centrer l'image redimensionnée
    paste_position = ((target_size[0] - new_width) // 2,
                      (target_size[1] - new_height) // 2)

    # Coller l'image redimensionnée
    padded_img.paste(resized_img, paste_position)

    return padded_img

def apply_data_augmentation(image, augmentation_type=None):
    """
    Applique une augmentation de données spécifique à une image.

    Args:
        image: Image PIL
        augmentation_type: Type d'augmentation à appliquer

    Returns:
        Image augmentée
    """
    if image is None:
        return None

    if augmentation_type is None:
        return image

    if augmentation_type == 'rotation':
        angle = np.random.randint(-30, 30)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)

    elif augmentation_type == 'flip':
        return ImageOps.mirror(image)

    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    elif augmentation_type == 'contrast':
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    elif augmentation_type == 'color':
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    else:
        return image

def preprocess_for_efficientnet(image, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence avec EfficientNet.

    Args:
        image: Image PIL ou tableau numpy
        target_size: Dimensions cibles

    Returns:
        Image prétraitée sous forme de tableau numpy
    """
    if image is None:
        return None

    # Convertir en tableau numpy si c'est une image PIL
    if isinstance(image, Image.Image):
        image = image.resize(target_size)
        image = np.array(image)
    else:
        # Si c'est déjà un tableau numpy
        image = cv2.resize(image, target_size)
        # Convertir BGR en RGB si l'image vient de OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Appliquer le prétraitement spécifique à EfficientNet
    image = preprocess_input(image)

    return image

def create_or_load_category_mapping(categories=None, mapping_file_path=None):
    """
    Crée ou charge un mapping de catégories.

    Args:
        categories: Liste des noms de catégories (si None, essaie de charger depuis le fichier)
        mapping_file_path: Chemin vers le fichier de mapping

    Returns:
        Dictionnaire de mapping {id: nom_categorie}
    """
    if mapping_file_path is None:
        from models.model_loader import get_model_paths
        paths = get_model_paths()
        mapping_file_path = paths["category_mapping"]

    # Si un fichier de mapping existe, le charger
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, 'r') as f:
                mapping_info = json.load(f)

            # Adapter selon la structure du fichier
            if "categories" in mapping_info:
                category_names = {int(idx): name for idx, name in mapping_info['categories'].items()}
            else:
                category_names = {int(v): k for k, v in mapping_info.get("category_mapping", {}).items()}

            return category_names
        except Exception as e:
            print(f"Erreur lors du chargement du mapping: {e}")

    # Si aucun fichier n'existe et qu'on a fourni des catégories, créer un nouveau mapping
    if categories is not None:
        category_mapping = {i: name for i, name in enumerate(categories)}

        # Sauvegarder le mapping si un chemin est spécifié
        if mapping_file_path:
            try:
                os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
                with open(mapping_file_path, 'w') as f:
                    json.dump({"categories": category_mapping}, f, indent=2)
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du mapping: {e}")

        return category_mapping

    # Si aucune option n'est disponible, utiliser des noms génériques
    return {i: f"Classe {i}" for i in range(7)}  # Par défaut 7 classes pour votre cas

# Créer des alias pour la compatibilité
preprocess_for_model = preprocess_for_efficientnet
preprocess_for_hqvit = preprocess_for_efficientnet  # Alias pour éviter les erreurs

def get_class_names(categories_dict):
    """
    Obtient une liste ordonnée des noms de classes à partir d'un dictionnaire de catégories.
    """
    # S'assurer que les indices sont triés
    sorted_indices = sorted(categories_dict.keys())
    return [categories_dict[idx] for idx in sorted_indices]

def process_prediction(predictions, categories_dict):
    """
    Traite les prédictions du modèle pour extraire la classe prédite et les probabilités.
    """
    # Obtenir l'index de la classe avec la plus haute probabilité
    predicted_class = np.argmax(predictions)
    
    # Obtenir le nom de la classe
    class_name = categories_dict.get(predicted_class, f"Classe {predicted_class}")
    
    # Obtenir la probabilité de la classe prédite
    probability = float(predictions[predicted_class])
    
    return predicted_class, class_name, probability

# Action spécifique au model convnexttiny 
# Objectif ! renforcer la solidité de l'API contre les importation circulaires

def preprocess_image_for_convnext(image, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence avec le modèle ConvNeXtTiny.

    Args:
        image: Image PIL
        target_size: Taille cible (défaut: 224x224)

    Returns:
        Tableau numpy normalisé prêt pour l'inférence
    """
    # Si l'image est déjà un tableau numpy, la convertir en image PIL
    if isinstance(image, np.ndarray):
        # Convertir le tableau en image PIL
        image = Image.fromarray((image * 255).astype(np.uint8))

    # Redimensionner l'image si nécessaire
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)

    # Convertir en tableau numpy
    img_array = np.array(image).astype(np.float32)

    # Normalisation pour ConvNeXtTiny (de -1 à 1)
    img_array = img_array / 127.5 - 1.0

    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
