"""
Module principal du résumeur Wikipedia
Implémente des techniques d'extractive summarization basées sur le preprocessing du TP
"""

import logging
import numpy as np
import pickle
import string
import heapq
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import wikipediaapi

from .config import Config
from .text_preprocessor import TextPreprocessor
from .dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


class WikipediaSummarizer:
    """
    Résumeur d'articles Wikipedia utilisant des techniques extractives
    Basé sur les techniques de preprocessing du TP Pipeline NLP
    """
    
    def __init__(self, config: Config):
        """
        Initialise le résumeur
        
        Args:
            config: Configuration du projet
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config.PREPROCESSING_CONFIG)
        self.dataset_builder = DatasetBuilder(config)
        
        # Modèles
        self.tfidf_vectorizer = None
        self.sentence_scorer = None
        self.is_trained = False
        
        # Configuration par défaut pour NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            logger.warning("Impossible de télécharger les ressources NLTK")
        
        # Initialisation de Wikipedia API
        self.wiki_api = {
            'fr': wikipediaapi.Wikipedia('fr'),
            'en': wikipediaapi.Wikipedia('en')
        }
        
        logger.info("WikipediaSummarizer initialisé")
    
    def get_wikipedia_article_and_summarize_nltk(self, article_title: str, language: str = "fr", sentences_count: int = 2) -> Dict[str, str]:
        """
        Récupère un article Wikipedia et le résume en utilisant NLTK
        
        Args:
            article_title: Titre de l'article Wikipedia
            language: Langue de l'article ('fr' ou 'en')
            sentences_count: Nombre de phrases dans le résumé
            
        Returns:
            Dictionnaire contenant l'article original et le résumé
        """
        logger.info(f"Récupération et résumé de l'article: {article_title} (langue: {language})")
        
        try:
            # Sélection de l'API Wikipedia selon la langue
            if language not in self.wiki_api:
                return {
                    "error": f"Langue non supportée: {language}. Langues supportées: fr, en",
                    "original_text": "",
                    "summary": ""
                }
            
            wiki = self.wiki_api[language]
            
            # Récupération de l'article
            page = wiki.page(article_title)
            
            if not page.exists():
                logger.error(f"Article non trouvé: {article_title}")
                return {
                    "error": f"Article '{article_title}' non trouvé",
                    "original_text": "",
                    "summary": ""
                }
            
            article_content = page.text
            article_url = page.fullurl
            logger.info(f"Article récupéré: {len(article_content)} caractères")
            
            # Génération du résumé avec NLTK
            summary = self._summarize_with_nltk(article_content, language, sentences_count)
            
            return {
                "title": page.title,
                "url": article_url,
                "original_text": article_content,
                "summary": summary,
                "original_length": len(article_content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(article_content) if article_content else 0
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération/résumé: {str(e)}")
            return {
                "error": str(e),
                "original_text": "",
                "summary": ""
            }
    
    def _summarize_with_nltk(self, text: str, language: str = "fr", sentences_count: int = 2) -> str:
        """
        Résume un texte en utilisant NLTK avec la méthode de fréquence pondérée
        Basé sur les techniques de la recherche web
        
        Args:
            text: Texte à résumer
            language: Langue du texte
            sentences_count: Nombre de phrases désirées dans le résumé
            
        Returns:
            Résumé du texte
        """
        logger.info(f"Résumé NLTK: {len(text)} caractères -> {sentences_count} phrases")
        
        # 1. Préprocessing du texte
        # Suppression des caractères spéciaux et conversion en minuscules
        formatted_text = re.sub(r'\[[0-9]*\]', ' ', text)  # Suppression des références
        formatted_text = re.sub(r'\s+', ' ', formatted_text)  # Suppression des espaces multiples
        
        # 2. Tokenisation des phrases
        sentence_list = sent_tokenize(text)
        
        if len(sentence_list) <= sentences_count:
            return text.strip()
        
        # 3. Suppression des stop words et préprocessing pour le calcul de fréquence
        stop_words = set()
        try:
            if language == "fr":
                stop_words = set(stopwords.words('french'))
            else:
                stop_words = set(stopwords.words('english'))
        except:
            logger.warning(f"Stop words non disponibles pour la langue: {language}")
        
        # 4. Calcul des fréquences pondérées
        word_frequencies = {}
        
        # Tokenisation des mots sans ponctuation
        words = word_tokenize(formatted_text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calcul des fréquences
        for word in words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        
        # Calcul des fréquences pondérées (division par la fréquence maximale)
        if word_frequencies:
            maximum_frequency = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] = word_frequencies[word] / maximum_frequency
        
        # 5. Calcul des scores des phrases
        sentence_scores = {}
        
        for i, sentence in enumerate(sentence_list):
            # Limitation aux phrases de longueur raisonnable
            words_in_sentence = word_tokenize(sentence.lower())
            words_in_sentence = [word for word in words_in_sentence if word.isalnum()]
            
            if len(words_in_sentence) < 30:  # Phrases pas trop longues
                sentence_score = 0
                word_count = 0
                
                for word in words_in_sentence:
                    if word in word_frequencies:
                        sentence_score += word_frequencies[word]
                        word_count += 1
                
                if word_count > 0:
                    sentence_scores[i] = sentence_score / word_count  # Score moyen
        
        # 6. Sélection des meilleures phrases
        if not sentence_scores:
            # Fallback: prendre les premières phrases
            return '. '.join(sentence_list[:sentences_count]) + '.'
        
        # Sélection des top phrases par score
        best_sentences_indices = heapq.nlargest(sentences_count, sentence_scores, key=sentence_scores.get)
        
        # Tri des indices pour conserver l'ordre original
        best_sentences_indices.sort()
        
        # Construction du résumé
        summary_sentences = [sentence_list[i] for i in best_sentences_indices]
        summary = ' '.join(summary_sentences)
        
        logger.info(f"Résumé généré: {len(summary)} caractères")
        return summary.strip()

    def summarize(self, content: str, language: str = "fr") -> str:
        """
        Génère un résumé d'un article (version simple pour commencer)
        
        Args:
            content: Contenu de l'article
            language: Langue de l'article
            
        Returns:
            Résumé généré
        """
        logger.info(f"Génération du résumé pour un texte de {len(content)} caractères")
        
        # Préprocessing du contenu
        processed_data = self.preprocessor.preprocess_for_summarization(content, language)
        
        sentences = processed_data['sentences']
        sentence_scores = processed_data['sentence_scores']
        
        if not sentences:
            return "Erreur: impossible d'extraire des phrases du contenu."
        
        # Sélection des meilleures phrases basée sur les scores
        sentence_count = min(
            self.config.MODEL_CONFIG['sentence_count'],
            len(sentences)
        )
        
        # Tri par score et sélection
        scored_sentences = list(zip(sentences, sentence_scores, range(len(sentences))))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Sélection des top phrases en préservant l'ordre original
        selected_indices = sorted([idx for _, _, idx in scored_sentences[:sentence_count]])
        selected_sentences = [sentences[i] for i in selected_indices]
        
        summary = ' '.join(selected_sentences)
        
        # Vérification de la longueur
        max_length = self.config.MODEL_CONFIG['max_summary_length']
        if len(summary) > max_length:
            # Troncature intelligente (au dernier point avant la limite)
            truncated = summary[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:  # Seulement si on ne perd pas trop
                summary = truncated[:last_period + 1]
            else:
                summary = truncated + "..."
        
        logger.info(f"Résumé généré: {len(summary)} caractères, {len(selected_sentences)} phrases")
        return summary
    
    def train(self) -> Dict[str, float]:
        """Version simplifiée pour commencer"""
        return {"message": "Entraînement non implémenté dans cette version de base"}
    
    def evaluate(self) -> Dict[str, float]:
        """Version simplifiée pour commencer"""
        return {"message": "Évaluation non implémentée dans cette version de base"} 