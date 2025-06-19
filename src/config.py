"""
Configuration centralisée pour le résumeur Wikipedia
"""

import os
from pathlib import Path
from typing import Dict, Any

# Chargement du fichier .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv non installé, utilisation des variables d'environnement système")


class Config:
    """Configuration principale du projet"""
    
    def __init__(self):
        # Chemins des dossiers
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
    
        # Configuration du dataset (utilise uniquement Wikipedia)
        self.USE_WIKIPEDIA_SUMMARIES = True  # Utilise les résumés Wikipedia natifs
        
        # Configuration Wikipedia
        self.WIKIPEDIA_LANGUAGES = {
            "fr": "français",
            "en": "english"
        }
        
        # Configuration du préprocessing (basée sur le TP)
        self.PREPROCESSING_CONFIG = {
            "remove_urls": True,
            "remove_phone_numbers": True,
            "remove_money_mentions": True,
            "convert_to_lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "min_word_length": 2,
            "max_word_length": 50
        }
        
        # Configuration du modèle de résumé
        self.MODEL_CONFIG = {
            "max_summary_length": 200,
            "min_summary_length": 50,
            "summary_ratio": 0.3,  # 30% de la longueur originale
            "sentence_count": 5,   # Nombre de phrases dans le résumé
        }
        
        # Configuration du dataset
        self.DATASET_CONFIG = {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": 42,
            "max_article_length": 10000,  # caractères
            "min_article_length": 500,
        }
        
        # Topics pour la création du dataset
        self.WIKIPEDIA_TOPICS = {
            "fr": [
                "Intelligence artificielle",
                "Machine learning",
                "Python (langage)", 
                "Traitement automatique du langage naturel",
                "Apprentissage profond",
                "Réseaux de neurones",
                "Science des données",
                "Algorithme",
                "Informatique",
                "Mathématiques",
                "Physique",
                "Histoire de France",
                "Géographie",
                "Biologie",
                "Chimie",
                "Philosophie",
                "Littérature française",
                "Art contemporain",
                "Musique classique",
                "Cinéma français"
            ],
            "en": [
                "Artificial intelligence",
                "Machine learning",
                "Python (programming language)",
                "Natural language processing",
                "Deep learning",
                "Neural network",
                "Data science",
                "Algorithm",
                "Computer science",
                "Mathematics",
                "Physics",
                "World history",
                "Geography",
                "Biology",
                "Chemistry",
                "Philosophy",
                "Literature",
                "Contemporary art",
                "Classical music",
                "Cinema"
            ]
        }
        
        # Créer les dossiers s'ils n'existent pas
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
    
    def get_dataset_path(self, language: str = "fr") -> Path:
        """Retourne le chemin du dataset pour une langue donnée"""
        return self.DATA_DIR / f"wikipedia_dataset_{language}.csv"
    
    def get_model_path(self, model_name: str = "summarizer") -> Path:
        """Retourne le chemin du modèle sauvegardé"""
        return self.MODELS_DIR / f"{model_name}.pkl"
    
    def get_summary_source(self) -> str:
        """Retourne la source utilisée pour les résumés"""
        return "Wikipedia native summaries" 