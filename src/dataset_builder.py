"""
Module de construction du dataset avec articles Wikipedia et résumés ChatGPT
"""

import json
import logging
import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import wikipediaapi
import pandas as pd

from .config import Config
from .text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Constructeur de dataset pour l'entraînement du résumeur
    Récupère des articles Wikipedia et génère des résumés avec ChatGPT
    """
    
    def __init__(self, config: Config):
        """
        Initialise le constructeur de dataset
        
        Args:
            config: Configuration du projet
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config.PREPROCESSING_CONFIG)
        
        # Initialiser l'API Wikipedia avec un user agent requis
        self.wiki = wikipediaapi.Wikipedia('NLP-Project (contact@example.com)', 'fr')
        
        logger.info("DatasetBuilder initialisé (mode Wikipedia uniquement)")
    
    def get_wikipedia_article(self, title: str, language: str = "fr") -> Optional[Dict[str, Any]]:
        """
        Récupère un article Wikipedia
        
        Args:
            title: Titre de l'article
            language: Langue ('fr' ou 'en')
            
        Returns:
            Dict avec les données de l'article ou None si erreur
        """
        try:
            # Créer une instance Wikipedia pour la langue demandée
            wiki = wikipediaapi.Wikipedia('NLP-Project (contact@example.com)', language)
            
            page = wiki.page(title)
            
            # Vérifier si la page existe
            if not page.exists():
                logger.error(f"Article '{title}' non trouvé")
                return None
            
            # Vérification de la longueur de l'article
            content = page.text
            content_length = len(content)
            min_length = self.config.DATASET_CONFIG['min_article_length']
            max_length = self.config.DATASET_CONFIG['max_article_length']
            
            if content_length < min_length:
                logger.warning(f"Article '{title}' trop court ({content_length} caractères)")
                return None
                
            if content_length > max_length:
                logger.info(f"Article '{title}' tronqué ({content_length} -> {max_length} caractères)")
                content = content[:max_length]
            
            # Générer un résumé simple (les 2 premières phrases)
            sentences = page.summary.split('. ')
            wikipedia_summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else page.summary
            
            article_data = {
                'title': page.title,
                'url': page.fullurl,
                'content': content,
                'wikipedia_summary': wikipedia_summary,
                'language': language,
                'content_length': len(content),
                'original_length': content_length
            }
            
            logger.info(f"Article récupéré: {page.title} ({len(content)} caractères)")
            return article_data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de '{title}': {e}")
            return None
    
    def generate_wikipedia_summary(self, content: str, wikipedia_summary: str, language: str = "fr") -> str:
        """
        Prépare le résumé basé sur Wikipedia (nettoyage et amélioration)
        
        Args:
            content: Contenu complet de l'article  
            wikipedia_summary: Résumé Wikipedia original
            language: Langue
            
        Returns:
            Résumé nettoyé et adapté
        """
        # Si le résumé Wikipedia est trop court, on prend les 2 premières phrases de l'article
        if len(wikipedia_summary) < 50:
            sentences = content.split('. ')
            clean_summary = '. '.join(sentences[:2]) + '.'
            logger.info(f"Résumé trop court, utilisation des premières phrases de l'article")
        else:
            # Utiliser le résumé Wikipedia comme base
            clean_summary = wikipedia_summary
            
        # Assurer une longueur raisonnable (100-500 caractères pour plus de flexibilité)
        if len(clean_summary) > 500:
            # Trouver le dernier point avant 500 caractères
            truncated = clean_summary[:500]
            last_period = truncated.rfind('.')
            if last_period > 300:  # Garde au moins 300 caractères
                clean_summary = truncated[:last_period + 1]
            else:
                clean_summary = truncated + "..."
                
        logger.info(f"Résumé Wikipedia préparé ({len(clean_summary)} caractères)")
        return clean_summary
    
    def create_dataset_entry(self, article_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Crée une entrée complète du dataset
        
        Args:
            article_data: Données de l'article Wikipedia
            
        Returns:
            Entrée du dataset avec toutes les métadonnées
        """
        try:
            # Génération du résumé basé sur Wikipedia
            logger.info(f"Préparation du résumé Wikipedia pour '{article_data['title']}'")
            processed_summary = self.generate_wikipedia_summary(
                article_data['content'],
                article_data['wikipedia_summary'], 
                article_data['language']
            )
            
            # Préprocessing du contenu
            processed_data = self.preprocessor.preprocess_for_summarization(
                article_data['content'],
                article_data['language']
            )
            
            # Construction de l'entrée finale - aplatie pour CSV
            entry = {
                'id': f"{article_data['language']}_{len(article_data['title'])}_{hash(article_data['title']) % 10000}",
                'title': article_data['title'],
                'url': article_data['url'],
                'language': article_data['language'],
                'original_content': article_data['content'],
                'processed_content': processed_data['clean_text'],
                'wikipedia_summary': article_data['wikipedia_summary'],
                'target_summary': processed_summary,
                
                # Features aplaties (correspondance avec TextPreprocessor)
                'sentence_count': processed_data['features']['sentence_count'],
                'avg_sentence_length': processed_data['features']['avg_sentence_length'],
                'lexical_diversity': processed_data['features']['lexical_diversity'],
                'caps_ratio': processed_data['features']['caps_ratio'],
                'word_count': processed_data['features']['word_count'],
                
                # Métadonnées aplaties
                'content_length': article_data['content_length'],
                'original_length': article_data['original_length'],
                'summary_length': len(processed_summary),
                'compression_ratio': len(processed_summary) / article_data['content_length'],
                'timestamp': time.time(),
                
                # Pour l'analyse détaillée, on garde les listes en format JSON string
                'sentences_json': json.dumps(processed_data['sentences'], ensure_ascii=False),
                'sentence_scores_json': json.dumps(processed_data['sentence_scores'], ensure_ascii=False)
            }
            
            logger.info(f"Entrée dataset créée pour: {article_data['title']}")
            return entry
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'entrée dataset: {e}")
            return None
    
    def build_dataset(self, num_articles: int = 100, language: str = "fr") -> Path:
        """
        Construit le dataset complet
        
        Args:
            num_articles: Nombre d'articles à traiter
            language: Langue des articles
            
        Returns:
            Chemin vers le fichier dataset créé
        """
        logger.info(f"Construction du dataset: {num_articles} articles en {language}")
        
        # Récupération des topics selon la langue
        topics = self.config.WIKIPEDIA_TOPICS.get(language, [])
        if not topics:
            raise ValueError(f"Aucun topic disponible pour la langue: {language}")
        
        dataset = []
        successful_articles = 0
        failed_articles = 0
        
        # Mélange des topics pour plus de diversité
        random.seed(self.config.DATASET_CONFIG['random_seed'])
        random.shuffle(topics)
        
        # Extension des topics si nécessaire
        extended_topics = (topics * ((num_articles // len(topics)) + 1))[:num_articles]
        
        for i, topic in enumerate(extended_topics):
            logger.info(f"Traitement {i+1}/{num_articles}: {topic}")
            
            try:
                # Récupération de l'article
                article_data = self.get_wikipedia_article(topic, language)
                if not article_data:
                    failed_articles += 1
                    continue
                
                # Création de l'entrée dataset
                entry = self.create_dataset_entry(article_data)
                if not entry:
                    failed_articles += 1
                    continue
                
                dataset.append(entry)
                successful_articles += 1
                
                # Pause pour éviter les limites de taux API
                time.sleep(1)
                
                # Sauvegarde intermédiaire tous les 10 articles
                if successful_articles % 10 == 0:
                    temp_path = self.config.get_dataset_path(language).with_suffix('.temp.csv')
                    self._save_dataset_csv(dataset, temp_path)
                    logger.info(f"Sauvegarde intermédiaire: {successful_articles} articles traités")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de '{topic}': {e}")
                failed_articles += 1
                continue
        
        # Sauvegarde finale
        output_path = self.config.get_dataset_path(language).with_suffix('.csv')
        self._save_dataset_csv(dataset, output_path)
        
        # Suppression du fichier temporaire s'il existe
        temp_path = output_path.with_suffix('.temp.csv')
        if temp_path.exists():
            temp_path.unlink()
        
        logger.info(f"Dataset construit: {successful_articles} articles réussis, {failed_articles} échecs")
        logger.info(f"Dataset sauvegardé: {output_path}")
        
        return output_path
    
    def _save_dataset_csv(self, dataset: List[Dict[str, Any]], path: Path) -> None:
        """
        Sauvegarde le dataset au format CSV
        
        Args:
            dataset: Liste des entrées du dataset
            path: Chemin de sauvegarde
        """
        try:
            # Conversion en DataFrame pandas
            df = pd.DataFrame(dataset)
            
            # Sauvegarde en CSV avec encodage UTF-8
            df.to_csv(path, index=False, encoding='utf-8')
                
            logger.info(f"Dataset CSV sauvegardé: {len(dataset)} entrées dans {path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde CSV: {e}")
            raise
    
    def _save_dataset(self, dataset: List[Dict[str, Any]], path: Path) -> None:
        """
        Sauvegarde le dataset au format CSV (ancien nom pour compatibilité)
        
        Args:
            dataset: Liste des entrées du dataset
            path: Chemin de sauvegarde
        """
        # Redirection vers la méthode CSV
        csv_path = path.with_suffix('.csv')
        self._save_dataset_csv(dataset, csv_path)
    
    def load_dataset(self, language: str = "fr") -> Optional[pd.DataFrame]:
        """
        Charge un dataset existant en CSV
        
        Args:
            language: Langue du dataset
            
        Returns:
            DataFrame pandas ou None si erreur
        """
        path = self.config.get_dataset_path(language).with_suffix('.csv')
        
        if not path.exists():
            logger.warning(f"Dataset CSV non trouvé: {path}")
            return None
        
        try:
            df = pd.read_csv(path, encoding='utf-8')
            
            logger.info(f"Dataset CSV chargé: {len(df)} entrées")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset CSV: {e}")
            return None 