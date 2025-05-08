import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Pour utiliser matplotlib sans affichage graphique

def plot_prediction_bars(predictions, title="Probabilités par classe"):
    """
    Génère un graphique à barres horizontales pour les prédictions de classes.
    Adapté pour votre format de données avec 'class_name' et 'probability'.
    
    Args:
        predictions: Liste de dictionnaires contenant les prédictions
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    if not predictions:
        return None
    
    # Extraire les noms de classes et les probabilités
    class_names = []
    probabilities = []
    
    for pred in predictions:
        class_names.append(pred["class_name"])
        probabilities.append(pred["probability"])
    
    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Définir les couleurs en fonction des probabilités
    colors = ['#2ecc71' if p == max(probabilities) else '#3498db' for p in probabilities]
    
    # Tracer les barres horizontales
    bars = ax.barh(class_names, probabilities, color=colors)
    
    # Configurer les limites et les étiquettes
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité", fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Ajouter les valeurs sur les barres
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{width:.3f}",
                ha='left', va='center', fontsize=10)
    
    # Améliorer l'apparence
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Ajuster les marges
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names, title="Matrice de confusion"):
    """
    Génère une matrice de confusion comme visualisation.

    Args:
        cm: Matrice de confusion (array numpy)
        class_names: Liste des noms de classes
        title: Titre du graphique

    Returns:
        Figure matplotlib
    """
    # Normaliser la matrice
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Utiliser seaborn pour un meilleur rendu
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    # Configurer les étiquettes
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title(title)

    plt.tight_layout()
    return fig

def plot_metrics_radar(metrics, class_names, metrics_names=["precision", "recall", "f1-score"]):
    """
    Génère un graphique radar pour visualiser les métriques par classe.

    Args:
        metrics: Dictionnaire de métriques par classe
        class_names: Liste des noms de classes
        metrics_names: Liste des noms de métriques à inclure

    Returns:
        Figure matplotlib
    """
    # Nombre de variables
    num_vars = len(metrics_names)

    # Calculer les angles pour chaque métrique
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Fermer le graphique

    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Préparer les couleurs
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))

    # Tracer chaque classe
    for i, class_name in enumerate(class_names):
        if class_name in metrics:
            # Récupérer les valeurs des métriques pour cette classe
            values = [metrics[class_name].get(metric, 0) for metric in metrics_names]
            values += values[:1]  # Fermer le graphique

            # Tracer la ligne
            ax.plot(angles, values, color=colors[i], linewidth=2, label=class_name)
            ax.fill(angles, values, color=colors[i], alpha=0.1)

    # Configurer les étiquettes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)

    # Ajouter une légende
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title("Performance par classe", size=15, color='navy', y=1.1)
    return fig

def plot_training_history(history, metrics=["accuracy", "loss"]):
    """
    Génère un graphique de l'historique d'entraînement.

    Args:
        history: Objet history de Keras ou dictionnaire
        metrics: Liste des métriques à inclure

    Returns:
        Figure matplotlib
    """
    # Convertir l'objet history en dictionnaire si nécessaire
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history

    # Calculer le nombre de sous-graphiques nécessaires
    n_metrics = sum(1 for m in metrics if m in history_dict)

    if n_metrics == 0:
        return None

    # Créer la figure avec les sous-graphiques
    fig, axes = plt.subplots(1, n_metrics, figsize=(15, 5))

    # Si un seul sous-graphique, transformer axes en liste
    if n_metrics == 1:
        axes = [axes]

    plot_idx = 0
    for metric in metrics:
        if metric in history_dict:
            # Tracer la métrique d'entraînement
            axes[plot_idx].plot(history_dict[metric], label=f'Train')

            # Tracer la métrique de validation si disponible
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                axes[plot_idx].plot(history_dict[val_metric], label=f'Validation')

            # Configurer les étiquettes
            axes[plot_idx].set_title(f"{metric.capitalize()}")
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel(metric.capitalize())
            axes[plot_idx].legend()

            plot_idx += 1

    plt.tight_layout()
    return fig

def plot_model_architecture(model=None):
    """
    Génère une représentation visuelle de l'architecture du modèle.

    Args:
        model: Modèle TensorFlow

    Returns:
        Figure matplotlib
    """
    if model is None:
        # Créer un diagramme simple de l'architecture générique
        fig, ax = plt.subplots(figsize=(10, 8))

        # Désactiver les axes
        ax.axis('off')

        # Position Y des différentes couches
        y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]

        # Largeurs des boîtes
        box_width = 0.6

        # Dessiner les boîtes d'architecture
        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[0] - 0.05), box_width, 0.1,
                                  facecolor='lightblue', edgecolor='blue', alpha=0.7))
        ax.text(0.5, y_positions[0], "Image d'entrée (224×224×3)",
               ha='center', va='center', fontsize=12)

        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[1] - 0.05), box_width, 0.1,
                                  facecolor='lightgreen', edgecolor='green', alpha=0.7))
        ax.text(0.5, y_positions[1], "EfficientNet-B0 Backbone",
               ha='center', va='center', fontsize=12)

        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[2] - 0.05), box_width, 0.1,
                                  facecolor='salmon', edgecolor='red', alpha=0.7))
        ax.text(0.5, y_positions[2], "Reshape + Token CLS + Embedding",
               ha='center', va='center', fontsize=12)

        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[3] - 0.05), box_width, 0.1,
                                  facecolor='plum', edgecolor='purple', alpha=0.7))
        ax.text(0.5, y_positions[3], "Couches Transformer (Auto-Attention)",
               ha='center', va='center', fontsize=12)

        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[4] - 0.05), box_width, 0.1,
                                  facecolor='khaki', edgecolor='olive', alpha=0.7))
        ax.text(0.5, y_positions[4], "Extraction Token CLS + Couches Denses",
               ha='center', va='center', fontsize=12)

        ax.add_patch(plt.Rectangle((0.5 - box_width/2, y_positions[5] - 0.05), box_width, 0.1,
                                  facecolor='lightskyblue', edgecolor='navy', alpha=0.7))
        ax.text(0.5, y_positions[5], "Classification (7 classes)",
               ha='center', va='center', fontsize=12)

        # Dessiner des flèches entre les boîtes
        for i in range(len(y_positions) - 1):
            ax.arrow(0.5, y_positions[i] - 0.05, 0, -0.05, head_width=0.03,
                    head_length=0.02, fc='black', ec='black')

        plt.title("Architecture du Modèle EfficientNet-Transformer", fontsize=16)
        return fig

    # Si un modèle est fourni, essayer d'utiliser TF pour générer le diagramme
    try:
        from tensorflow.keras.utils import plot_model

        # Générer l'image du modèle en mémoire
        buffer = io.BytesIO()
        plot_model(model, to_file=buffer, show_shapes=True, show_layer_names=True,
                   expand_nested=True, dpi=96)
        buffer.seek(0)

        # Charger l'image en tant que figure matplotlib
        img = Image.open(buffer)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        ax.axis('off')
        return fig
    except Exception as e:
        print(f"Erreur lors de la génération du diagramme du modèle: {e}")
        # En cas d'échec, utiliser l'approche générique
        return plot_model_architecture(None)

def create_class_probabilities_pie(predictions, threshold=0.05):
    """
    Crée un graphique en camembert pour les probabilités de classes.
    
    Args:
        predictions: Liste de dictionnaires avec 'class_name' et 'probability'
        threshold: Seuil minimum pour afficher une classe séparément
        
    Returns:
        Figure matplotlib
    """
    # Filtrer les prédictions significatives
    significant_predictions = []
    other_prob = 0
    
    for pred in predictions:
        if pred['probability'] >= threshold:
            significant_predictions.append(pred)
        else:
            other_prob += pred['probability']
    
    # Ajouter "Autres" si nécessaire
    if other_prob > 0:
        significant_predictions.append({
            'class_name': 'Autres',
            'probability': other_prob
        })
    
    # Extraire les données pour le graphique
    labels = [pred['class_name'] for pred in significant_predictions]
    sizes = [pred['probability'] for pred in significant_predictions]
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Définir les couleurs
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Créer le camembert
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    
    # Améliorer l'apparence
    ax.set_title("Distribution des probabilités", fontsize=16)
    
    # Améliorer la lisibilité des pourcentages
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_weight('bold')
    
    return fig

def plot_top_n_predictions(predictions, n=5):
    """
    Crée un graphique montrant les N meilleures prédictions.
    
    Args:
        predictions: Liste de dictionnaires avec 'class_name' et 'probability'
        n: Nombre de prédictions à afficher
        
    Returns:
        Figure matplotlib
    """
    # Prendre les N meilleures prédictions
    top_predictions = predictions[:n]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Préparer les données
    classes = [pred['class_name'] for pred in top_predictions]
    probabilities = [pred['probability'] for pred in top_predictions]
    
    # Créer le graphique à barres
    bars = ax.bar(classes, probabilities, color='skyblue', edgecolor='navy')
    
    # Ajouter les valeurs au-dessus des barres
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Configurer le graphique
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probabilité', fontsize=12)
    ax.set_title(f'Top {n} Prédictions', fontsize=16)
    
    # Rotation des labels si nécessaire
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
