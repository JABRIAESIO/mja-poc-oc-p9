# mja-poc-oc-p9
Projet OC-P9
# Application de Classification d'Images 
#  mja-poc-oc-p9 — Classification d'Images avec ConvNeXtTiny

Projet OC-P9 – Application Streamlit pour la classification d'images à partir d'un modèle **ConvNeXtTiny** entraîné sur le dataset **Flipkart** (7 catégories).

---

## Objectif

Déployer une application interactive permettant :
- L’**upload d’une image**
- La **prédiction de sa catégorie** parmi 7 classes Flipkart
- L'affichage du **score de confiance**
- Le chargement du modèle depuis **Hugging Face**

---

## Architecture du Modèle

- **Backbone** : ConvNeXtTiny (2022)
- **Tête de classification** : Dense + Softmax
- **Format du modèle** : `.keras` (sauvegarde Keras native)
- **Modèle final** : hébergé sur Hugging Face (via `model_loader.py`)

---

## Structure du projet

```
/
├── app.py                      # Application principale Streamlit
├── requirements.txt            # Dépendances Python
├── utils/                      # Utilitaires
│   ├── preprocessing.py        # Fonctions de prétraitement
│   └── visualization.py        # Fonctions de visualisation
├── models/                     # Dossier pour les modèles
│   ├── model_loader.py         # Chargement du modèle
│   └── inference.py            # Fonctions d'inférence
└── assets/                     # Images et autres ressources
    └── examples/               # Images d'exemple
```

## Installation locale

1. Cloner ce dépôt :
```bash
git clone https://github.com/votre-username/mja-poc-oc-p9.git
cd mja-poc-oc-p9
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancer l'application :
```bash
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

Cette application est configurée pour être facilement déployée sur [Streamlit Cloud](https://streamlit.io/cloud). 

Pour déployer  :
1. Forker ce dépôt
2. On se Connecte à Streamlit Cloud
3. On déploie depuis le fork GitHub

## Modèle et poids

Le modèle nécessite un fichier de poids pré-entraînés. Les chemins sont configurés pour rechercher :
Voici le chemin du modèle sur hugging Face ( car il dépasse 100MO limite imposée par streamlit Cloud )
HF_MODEL_URL = "https://huggingface.co/mourad42008/convnext-tiny-flipkart-classification/resolve/main/model_final.keras"
- `/data/OC/P9/convnext-output_07052025_1735/models/model_final.keras` (en local)

## Licence

Ce projet est disponible sous licence MIT. Voir le fichier LICENSE pour plus de détails.
