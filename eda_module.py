import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import Counter
import cv2

# Imports depuis votre structure existante
from models.model_loader import load_categories
from utils.preprocessing import preprocess_image_for_convnext

class EDAAnalyzer:
    """Classe pour l'analyse exploratoire des données d'images"""
    
    def __init__(self, examples_dir, categories):
        self.examples_dir = examples_dir
        self.categories = categories
        self.image_metadata_file = os.path.join(examples_dir, "image_metadata.json")
        self.image_mapping_file = os.path.join(examples_dir, "image_mapping.json")
    
    @st.cache_data
    def load_image_metadata(_self):
        """Charge les métadonnées des images depuis le fichier JSON"""
        if os.path.exists(_self.image_metadata_file):
            try:
                with open(_self.image_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Erreur lors du chargement des métadonnées: {e}")
                return {}
        return {}
    
    @st.cache_data
    def load_image_mapping(_self):
        """Charge le mapping image -> catégorie depuis le fichier JSON"""
        if os.path.exists(_self.image_mapping_file):
            try:
                with open(_self.image_mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Erreur lors du chargement du mapping: {e}")
                return {}
        return {}
    
    @st.cache_data
    def analyze_dataset(_self):
        """Analyse le dataset et retourne les statistiques"""
        image_files = [f for f in os.listdir(_self.examples_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Utiliser les métadonnées réelles
        metadata = _self.load_image_metadata()
        image_mapping = _self.load_image_mapping()
        
        # Calculer la distribution des catégories
        category_counts = {}
        if metadata and 'images' in metadata:
            for img_info in metadata['images']:
                category = img_info.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
        elif image_mapping:
            # Fallback sur le mapping
            category_counts = Counter([info.get('category', 'Unknown') 
                                     for info in image_mapping.values()])
        else:
            # Distribution par défaut
            category_counts = dict.fromkeys(_self.categories.values(), 2)
        
        return {
            'total_images': len(image_files),
            'category_distribution': category_counts,
            'image_files': image_files,
            'avg_file_size': _self._calculate_avg_file_size(image_files),
            'metadata': metadata,
            'mapping': image_mapping
        }
    
    def _calculate_avg_file_size(self, image_files):
        """Calcule la taille moyenne des fichiers d'images"""
        sizes = []
        for img_file in image_files[:10]:  # Limiter à 10 pour performance
            try:
                img_path = os.path.join(self.examples_dir, img_file)
                if os.path.exists(img_path):
                    size = os.path.getsize(img_path) / 1024  # en KB
                    sizes.append(size)
            except Exception:
                continue
        return np.mean(sizes) if sizes else 0

def get_sample_images_for_category(category_name, metadata, mapping, max_samples=3):
    """Récupère des images échantillons pour une catégorie donnée"""
    category_images = []
    
    # Utiliser les métadonnées d'abord
    if metadata and 'images' in metadata:
        for img_info in metadata['images']:
            if img_info.get('category') == category_name:
                category_images.append({
                    'filename': img_info['filename'],
                    'description': img_info.get('description', '')
                })
    
    # Fallback sur le mapping
    elif mapping:
        for filename, info in mapping.items():
            if info.get('category') == category_name:
                category_images.append({
                    'filename': filename,
                    'description': info.get('description', '')
                })
    
    return category_images[:max_samples]

def display_eda_mode():
    """Mode EDA principal - Analyse exploratoire des données"""
    # Ajouter ancre pour navigation accessible
    st.markdown('<div id="eda-content"></div>', unsafe_allow_html=True)
    st.header("📊 Analyse Exploratoire des Données")
    
    # Description introductive pour l'accessibilité
    st.markdown("""
    Cette section présente une analyse exploratoire des données d'images utilisées 
    pour l'entraînement du modèle de classification. Utilisez les onglets ci-dessous 
    pour explorer différents aspects du dataset.
    """)
    
    # Initialiser l'analyseur EDA
    examples_dir = os.path.join("assets", "examples")
    categories = load_categories()
    eda_analyzer = EDAAnalyzer(examples_dir, categories)
    
    # Vérifier que le dossier existe
    if not os.path.exists(examples_dir):
        st.error(f"❌ Le dossier d'exemples '{examples_dir}' n'existe pas.")
        return
    
    # Analyser le dataset
    with st.spinner("Analyse du dataset en cours..."):
        analysis_results = eda_analyzer.analyze_dataset()
    
    # Créer des onglets pour organiser l'EDA avec navigation accessible
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Vue d'ensemble",
        "🖼️ Exemples par Catégorie", 
        "📊 Distribution des Classes", 
        "🔍 Transformations"
    ])
    
    # Ajouter information sur navigation clavier
    st.markdown("""
    <div style="font-size: 14px; color: #666; margin-bottom: 1rem;">
    💡 <strong>Navigation</strong> : Utilisez les touches fléchées ou Tab pour naviguer entre les onglets
    </div>
    """, unsafe_allow_html=True)
    
    with tab1:
        display_overview_tab(analysis_results, categories)
    
    with tab2:
        display_category_examples_tab(analysis_results, examples_dir, eda_analyzer)
    
    with tab3:
        display_distribution_tab(analysis_results)
    
    with tab4:
        display_transformations_tab(analysis_results, examples_dir)

def display_overview_tab(analysis_results, categories):
    """Onglet Vue d'ensemble"""
    st.subheader("Statistiques Générales du Dataset")
    
    # Métriques principales avec colonnes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Images exemples",
            value=analysis_results['total_images'],
            help="Nombre d'images d'exemple disponibles dans l'application"
        )
    
    with col2:
        st.metric(
            label="Catégories",
            value=len(categories),
            help="Nombre total de catégories de classification"
        )
    
    with col3:
        st.metric(
            label="Taille moyenne",
            value=f"{analysis_results['avg_file_size']:.1f} KB",
            help="Taille moyenne des fichiers d'images"
        )
    
    with col4:
        # Données du dataset complet (vous pouvez ajuster cette valeur)
        total_dataset_size = 1050
        st.metric(
            label="Dataset complet",
            value=f"{total_dataset_size} images",
            help="Taille totale du dataset d'entraînement Flipkart"
        )
    
    # Informations détaillées
    st.markdown("### À propos du Dataset")
    
    # Vérifier s'il y a des métadonnées
    metadata = analysis_results.get('metadata', {})
    if metadata:
        st.success("✅ Métadonnées détaillées disponibles")
    else:
        st.info("ℹ️ Métadonnées basiques utilisées")
    
    # Description du dataset
    st.markdown("""
    **Dataset Flipkart** utilisé pour l'entraînement du modèle ConvNeXtTiny :
    
    - **Source** : Plateforme e-commerce Flipkart
    - **Format** : Images JPEG de produits variés
    - **Résolution** : Variable (redimensionnée à 224×224 pour le modèle)
    - **Classes** : 7 catégories de produits distinctes
    - **Traitement** : Augmentation de données appliquée durant l'entraînement
    """)
    
    # Informations techniques
    with st.expander("ℹ️ Détails techniques du préprocessing"):
        st.markdown("""
        **Pipeline de préprocessing appliqué** :
        1. **Redimensionnement** : 224×224 pixels (taille d'entrée ConvNeXtTiny)
        2. **Normalisation** : Valeurs pixel dans [-1, 1]
        3. **Augmentation** : Rotation, flip, variation luminosité (entraînement)
        4. **Format** : Conversion RGB, ajout dimension batch
        """)

def display_category_examples_tab(analysis_results, examples_dir, eda_analyzer):
    """Onglet Exemples par Catégorie"""
    st.subheader("Exemples d'Images par Catégorie")
    
    metadata = analysis_results.get('metadata', {})
    mapping = analysis_results.get('mapping', {})
    categories = load_categories()
    
    # Parcourir chaque catégorie
    for category_id, category_name in categories.items():
        st.markdown(f"### 📁 {category_name}")
        
        # Obtenir les images pour cette catégorie
        category_images = get_sample_images_for_category(
            category_name, metadata, mapping, max_samples=3
        )
        
        if category_images:
            # Afficher les images en colonnes
            cols = st.columns(min(3, len(category_images)))
            
            for idx, img_info in enumerate(category_images):
                with cols[idx]:
                    img_path = os.path.join(examples_dir, img_info['filename'])
                    
                    if os.path.exists(img_path):
                        try:
                            # Charger et afficher l'image avec alt text descriptif
                            img = Image.open(img_path)
                            st.image(
                                img,
                                caption=f"Exemple {idx+1} - {category_name}",
                                use_column_width=True
                            )
                            
                            # Description alternative pour lecteurs d'écran
                            st.markdown(f"**Description image** : {img_info.get('description', f'Produit de la catégorie {category_name}')}")
                            
                            # Informations sur l'image
                            st.markdown(f"**Fichier**: `{img_info['filename']}`")
                            
                            # Description si disponible
                            if img_info.get('description'):
                                st.markdown(f"**Description**: {img_info['description']}")
                            
                            # Dimensions
                            st.markdown(f"**Dimensions**: {img.size[0]}×{img.size[1]}px")
                            
                        except Exception as e:
                            st.error(f"Erreur chargement: {e}")
                    else:
                        st.warning(f"Image non trouvée: {img_info['filename']}")
        else:
            st.info(f"Aucun exemple trouvé pour la catégorie '{category_name}'")

def display_distribution_tab(analysis_results):
    """Onglet Distribution des Classes"""
    st.subheader("Distribution des Classes")
    
    category_counts = analysis_results['category_distribution']
    
    if not category_counts:
        st.warning("Aucune donnée de distribution disponible")
        return
    
    # Graphique en barres avec couleurs accessibles WCAG
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Couleurs avec bon contraste WCAG (4.5:1 minimum)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2']
    
    categories_list = list(category_counts.keys())
    counts_list = list(category_counts.values())
    
    # Créer le graphique
    bars = ax.bar(categories_list, counts_list, 
                  color=colors[:len(categories_list)], 
                  edgecolor='black', linewidth=0.5)
    
    # Améliorer l'accessibilité
    ax.set_title('Distribution des Exemples par Catégorie', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Catégories de Produits', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nombre d\'Exemples', fontsize=14, fontweight='bold')
    
    # Rotation des labels pour meilleure lisibilité
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts_list):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Grille pour faciliter la lecture
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    # Description alternative pour l'accessibilité
    st.markdown("""
    **Description du graphique** : Ce graphique en barres montre la répartition 
    des images d'exemple disponibles pour chaque catégorie de produits. Les valeurs 
    sont affichées au-dessus de chaque barre pour une lecture précise.
    """)
    
    # Tableau détaillé
    st.markdown("### 📋 Tableau Détaillé de Distribution")
    
    distribution_data = []
    total = sum(counts_list)
    
    for cat, count in category_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        distribution_data.append({
            "Catégorie": cat,
            "Nombre d'exemples": count,
            "Pourcentage": f"{percentage:.1f}%",
            "Proportion": f"{count}/{total}"
        })
    
    # Trier par nombre d'exemples décroissant
    distribution_data.sort(key=lambda x: x["Nombre d'exemples"], reverse=True)
    
    st.table(distribution_data)
    
    # Statistiques supplémentaires
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total exemples", total)
    
    with col2:
        avg_per_category = total / len(category_counts) if category_counts else 0
        st.metric("Moyenne par catégorie", f"{avg_per_category:.1f}")
    
    with col3:
        max_count = max(counts_list) if counts_list else 0
        min_count = min(counts_list) if counts_list else 0
        st.metric("Écart (max-min)", max_count - min_count)

def display_transformations_tab(analysis_results, examples_dir):
    """Onglet Transformations d'Images"""
    st.subheader("Transformations d'Images")
    
    # Sélectionner une image pour démontrer les transformations
    example_images = analysis_results['image_files']
    
    if not example_images:
        st.warning("Aucune image disponible pour démonstration")
        return
    
    selected_image = st.selectbox(
        "Choisir une image pour voir les transformations :",
        example_images,
        help="Sélectionnez une image pour voir les étapes de préprocessing appliquées. Navigation possible avec les flèches du clavier."
    )
    
    if selected_image:
        img_path = os.path.join(examples_dir, selected_image)
        
        if not os.path.exists(img_path):
            st.error(f"Image non trouvée : {img_path}")
            return
        
        try:
            # Charger l'image originale
            original_img = Image.open(img_path)
            
            # Créer les transformations
            st.markdown("### 🔄 Étapes de Préprocessing")
            
            # Organisation en colonnes
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Image Originale")
                st.image(original_img, 
                        caption="Image d'origine", 
                        use_column_width=True)
                
                # Description alternative pour accessibilité
                st.markdown(f"**Description de l'image** : Image d'origine du fichier {selected_image}")
                
                # Afficher les dimensions originales
                st.markdown(f"**Dimensions** : {original_img.size[0]} × {original_img.size[1]} pixels")
                
                # Taille du fichier
                file_size = os.path.getsize(img_path) / 1024
                st.markdown(f"**Taille du fichier** : {file_size:.1f} KB")
                
                # Mode couleur
                st.markdown(f"**Mode couleur** : {original_img.mode}")
            
            with col2:
                st.markdown("#### ⚙️ Après Préprocessing")
                
                # Appliquer les transformations existantes
                processed_img = preprocess_image_for_convnext(original_img)
                
                # Convertir pour affichage (dénormalisation)
                # Le préprocessing ConvNeXt normalise entre [-1, 1]
                if processed_img.ndim == 4:  # Batch dimension présente
                    display_img = processed_img[0]
                else:
                    display_img = processed_img
                
                # Dénormaliser de [-1, 1] vers [0, 255]
                display_img = (display_img + 1) * 127.5
                display_img = np.clip(display_img, 0, 255).astype(np.uint8)
                
                st.image(display_img, caption="Image préprocessée", use_column_width=True)
                st.markdown("**Nouvelles dimensions** : 224 × 224 pixels")
                st.markdown("**Format** : RGB normalisé [-1, 1]")
                st.markdown("**Prêt pour** : Inférence ConvNeXtTiny")
            
            # Détails des transformations
            st.markdown("### 📋 Détail des Transformations Appliquées")
            
            # Utiliser des colonnes pour présenter les étapes
            transform_col1, transform_col2 = st.columns(2)
            
            with transform_col1:
                st.markdown("""
                **1. Préparation de l'image** :
                - Conversion en mode RGB si nécessaire
                - Redimensionnement à 224×224 pixels
                - Préservation du ratio d'aspect (padding si nécessaire)
                """)
                
                st.markdown("""
                **2. Normalisation** :
                - Conversion en array NumPy
                - Normalisation des pixels : [0, 255] → [-1, 1]
                - Format attendu par ConvNeXtTiny
                """)
            
            with transform_col2:
                st.markdown("""
                **3. Formatage final** :
                - Ajout de la dimension batch (1, 224, 224, 3)
                - Type de données : float32
                - Ordre des canaux : RGB
                """)
                
                st.markdown("""
                **4. Optimisations** :
                - Preprocessing standard ConvNeXt
                - Compatible avec TensorFlow/Keras
                - Prêt pour prédiction
                """)
            
            # Comparaison optionnelle avec d'autres transformations
            with st.expander("🔍 Voir d'autres transformations possibles"):
                st.markdown("#### Transformations Supplémentaires (non utilisées dans le modèle)")
                
                st.info("""
                **Note** : Ces transformations sont présentées à titre éducatif. 
                Le modèle ConvNeXtTiny utilise uniquement le preprocessing standard décrit ci-dessus.
                """)
                
                try:
                    # Démonstration de transformations supplémentaires
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Égalisation d'histogramme
                        img_array = np.array(original_img.convert('RGB'))
                        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
                        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                        
                        st.image(img_eq, caption="Égalisation d'histogramme", use_column_width=True)
                        st.markdown("**Effet** : Améliore le contraste global")
                    
                    with col4:
                        # Flou gaussien
                        img_blur = cv2.GaussianBlur(np.array(original_img), (15, 15), 0)
                        st.image(img_blur, caption="Flou gaussien", use_column_width=True)
                        st.markdown("**Effet** : Réduit le bruit et les détails fins")
                
                except Exception as e:
                    st.warning(f"Erreur lors de la génération des transformations supplémentaires : {e}")
        
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des transformations : {e}")

# Fonction principale pour intégrer à app.py
def main_eda():
    """Fonction principale du module EDA"""
    display_eda_mode()

if __name__ == "__main__":
    # Test du module
    main_eda()
