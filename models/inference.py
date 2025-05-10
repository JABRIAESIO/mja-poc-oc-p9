import numpy as np
import time
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import traceback

# Importer depuis utils.preprocessing au lieu d'importer de model_loader
from utils.preprocessing import preprocess_image_for_convnext, resize_and_pad_image, apply_data_augmentation

def predict_image(model, image, categories, verbose=True):
    """
    Prédit la classe d'une image avec le modèle ConvNeXtTiny.

    Args:
        model: Modèle Keras chargé
        image: Image PIL
        categories: Dictionnaire des catégories {index: nom}
        verbose: Si True, affiche des détails supplémentaires

    Returns:
        Dictionnaire contenant les résultats de la prédiction
    """
    try:
        start_time = time.time()

        # Prétraitement de l'image
        preprocessed_image = preprocess_image_for_convnext(image)

        # Vérifier la forme de l'image prétraitée
        if verbose:
            print(f"Shape de l'image prétraitée: {preprocessed_image.shape}")
            print(f"Type de l'image prétraitée: {preprocessed_image.dtype}")
            print(f"Valeurs min/max: {preprocessed_image.min():.4f}/{preprocessed_image.max():.4f}")

        # Mesurer le temps de prétraitement
        preprocess_time = time.time() - start_time

        # Inférence
        inference_start = time.time()
        predictions = model.predict(preprocessed_image, verbose=0)
        inference_time = time.time() - inference_start

        # Temps total (prétraitement + inférence)
        total_time = time.time() - start_time

        # Vérifier la forme des prédictions
        if verbose:
            print(f"Shape des prédictions: {predictions.shape}")
            print(f"Prédictions brutes: {predictions[0]}")

        # Obtenir l'indice de la classe prédite
        predicted_class_idx = np.argmax(predictions[0])

        # Obtenir le nom de la classe prédite
        if predicted_class_idx in categories:
            predicted_class = categories[predicted_class_idx]
        else:
            # Si l'indice n'est pas dans les catégories, utiliser un nom générique
            predicted_class = f"Classe {predicted_class_idx}"
            if verbose:
                print(f"ATTENTION: Indice de classe {predicted_class_idx} non trouvé dans le mapping")
                print(f"Catégories disponibles: {categories}")

        # Calculer la confiance de la prédiction
        confidence = float(predictions[0][predicted_class_idx])

        # Préparer les prédictions détaillées pour toutes les classes
        all_predictions = []
        for i in range(len(predictions[0])):
            class_name = categories.get(i, f"Classe {i}")
            all_predictions.append({
                "class_index": i,
                "class_name": class_name,
                "probability": float(predictions[0][i])
            })

        # Trier les prédictions par probabilité décroissante
        all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)

        # Retourner les résultats
        return {
            "predicted_class": predicted_class,
            "predicted_class_idx": int(predicted_class_idx),
            "confidence": confidence,
            "preprocess_time": float(preprocess_time),
            "inference_time": float(inference_time),
            "total_time": float(total_time),
            "all_predictions": all_predictions
        }

    except Exception as e:
        # En cas d'erreur, retourner un dictionnaire d'erreur
        error_trace = traceback.format_exc()

        if verbose:
            print(f"ERREUR lors de la prédiction: {str(e)}")
            print(error_trace)

        return {
            "error": str(e),
            "error_trace": error_trace
        }

def plot_prediction_bars(predictions, title="Probabilités par classe", figsize=(10, 6)):
    """
    Génère un graphique à barres des probabilités de prédiction.

    Args:
        predictions: Liste des prédictions (dictionnaires avec class_name et probability)
        title: Titre du graphique
        figsize: Taille de la figure

    Returns:
        Figure matplotlib
    """
    # Extraire les noms de classes et probabilités
    class_names = [pred['class_name'] for pred in predictions]
    probabilities = [pred['probability'] * 100 for pred in predictions]

    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)

    # Générer le graphique à barres
    bars = ax.barh(class_names, probabilities, color='skyblue')

    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height()/2,
            f"{probabilities[i]:.2f}%",
            va='center'
        )

    # Ajouter des lignes de grille pour faciliter la lecture
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Ajouter des titres et étiquettes
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Probabilité (%)', fontsize=12)
    ax.set_ylabel('Classe', fontsize=12)

    # Limiter l'axe x de 0 à 100%
    ax.set_xlim(0, 105)

    # Mettre en évidence la classe avec la plus haute probabilité
    bars[0].set_color('dodgerblue')

    # Ajuster la mise en page
    plt.tight_layout()

    return fig

def get_top_n_predictions(predictions, n=3):
    """
    Récupère les N prédictions principales.

    Args:
        predictions: Dictionnaire de résultats de prédiction
        n: Nombre de prédictions principales à retourner

    Returns:
        Liste des N principales prédictions
    """
    if "error" in predictions:
        return []

    return predictions["all_predictions"][:n]

def extract_features(model, image, layer_name=None):
    """
    Extrait les caractéristiques d'une image depuis une couche spécifique du modèle.
    Utile pour la visualisation ou l'analyse des caractéristiques.

    Args:
        model: Modèle Keras
        image: Image prétraitée (tableau numpy)
        layer_name: Nom de la couche (si None, utilise la sortie avant la dernière couche)

    Returns:
        Caractéristiques extraites
    """
    try:
        # Si layer_name est None, trouver l'avant-dernière couche
        if layer_name is None:
            # Généralement l'avant-dernière couche pour la classification
            layer_name = model.layers[-2].name

        # Créer un modèle qui renvoie les sorties jusqu'à la couche spécifiée
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )

        # Prétraiter l'image si nécessaire
        if isinstance(image, Image.Image):
            image = preprocess_image_for_convnext(image)

        # Extraire les caractéristiques
        features = feature_model.predict(image)

        return features

    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {e}")
        return None

def get_model_summary_dict(model):
    """
    Génère un résumé du modèle sous forme de dictionnaire.

    Args:
        model: Modèle Keras

    Returns:
        Dictionnaire contenant le résumé du modèle
    """
    if model is None:
        return {"error": "Modèle non disponible"}

    try:
        # Extraire les informations de base
        summary = {
            "name": model.name,
            "layers_count": len(model.layers),
            "parameters_count": model.count_params(),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "layers": []
        }

        # Ajouter des informations sur chaque couche
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "trainable": layer.trainable,
                "parameters": layer.count_params(),
                "input_shape": str(layer.input_shape),
                "output_shape": str(layer.output_shape)
            }
            summary["layers"].append(layer_info)

        return summary

    except Exception as e:
        return {"error": f"Erreur lors de la génération du résumé: {str(e)}"}

# Si exécuté directement, tester les fonctions
if __name__ == "__main__":
    from tensorflow.keras.applications import EfficientNetB0
    
    print("Test du script d'inférence...")
    
    # Créer un modèle de test
    test_model = EfficientNetB0(weights='imagenet', include_top=True)
    
    # Créer une image de test
    test_image = Image.new('RGB', (300, 200), color='black')
    
    # Dictionnaire de catégories pour le test
    test_categories = {i: f"Test Cat {i}" for i in range(1000)}
    
    # Tester la prédiction
    result = predict_image(test_model, test_image, test_categories, verbose=True)
    
    if "error" in result:
        print(f"Erreur lors du test: {result['error']}")
    else:
        print(f"Test réussi! Classe prédite: {result['predicted_class']}")
        print(f"Temps d'inférence: {result['inference_time']*1000:.2f} ms")
    
    print("Test terminé.")
