"""
Module de préprocessing de texte basé sur les techniques du TP Pipeline NLP
Adapté pour le résumé d'articles Wikipedia
"""

import re
import string
import logging
from typing import Dict, List, Tuple, Any
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Téléchargement des ressources NLTK si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Classe de préprocessing de texte inspirée du TP Pipeline NLP
    Adaptée pour les articles Wikipedia
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le préprocesseur
        
        Args:
            config: Configuration du préprocessing
        """
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        
        # Chargement des stop words selon la langue
        self.stop_words_fr = set(stopwords.words('french'))
        self.stop_words_en = set(stopwords.words('english'))
        
        # Mots importants à préserver (équivalent aux mots spam dans le TP)
        self.important_words_fr = {
            'donc', 'mais', 'car', 'parce', 'ainsi', 'alors', 'ensuite',
            'enfin', 'd\'abord', 'puis', 'cependant', 'néanmoins', 'toutefois'
        }
        
        self.important_words_en = {
            'therefore', 'however', 'moreover', 'furthermore', 'consequently',
            'thus', 'hence', 'although', 'nevertheless', 'nonetheless'
        }
        
        logger.info("TextPreprocessor initialisé")
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extrait les features numériques du texte (inspiré du TP)
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict avec les features extraites
        """
        features = {}
        
        if len(text) > 0:
            # Ratio de majuscules
            caps_count = sum(1 for char in text if char.isupper())
            features['caps_ratio'] = caps_count / len(text)
            
            # Nombre de phrases
            sentences = sent_tokenize(text)
            features['sentence_count'] = len(sentences)
            
            # Longueur moyenne des phrases
            if sentences:
                avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
                features['avg_sentence_length'] = avg_sentence_length
            else:
                features['avg_sentence_length'] = 0
                
            # Nombre de mots
            words = word_tokenize(text.lower())
            features['word_count'] = len(words)
            
            # Diversité lexicale (ratio mots uniques / total)
            unique_words = set(words)
            features['lexical_diversity'] = len(unique_words) / len(words) if words else 0
            
        else:
            features = {
                'caps_ratio': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'word_count': 0,
                'lexical_diversity': 0
            }
        
        return features
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        Nettoie le texte Wikipedia (adaptation du nettoyage spam du TP)
        
        Args:
            text: Texte brut Wikipedia
            
        Returns:
            Texte nettoyé
        """
        # Suppression des références Wikipedia [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Suppression des balises de formatage Wikipedia
        text = re.sub(r'{{[^}]*}}', '', text)
        text = re.sub(r'{[^}]*}', '', text)
        
        # Suppression des URLs (similaire au TP)
        if self.config.get('remove_urls', True):
            text = re.sub(r'(https?://\S+|www\.\S+|\S+\.(com|org|net|fr|en)\S*)', ' URL_TOKEN ', text)
        
        # Suppression des numéros de téléphone (adaptation du TP)
        if self.config.get('remove_phone_numbers', True):
            text = re.sub(r'\b\d{4,11}\b', ' PHONE_TOKEN ', text)
        
        # Suppression des montants d'argent (adaptation du TP) 
        if self.config.get('remove_money_mentions', True):
            text = re.sub(r'([$£€]\d+|\b\d+\s*(euros?|dollars?|pounds?)\b)', ' MONEY_TOKEN ', text, flags=re.IGNORECASE)
        
        # Conversion en minuscules
        if self.config.get('convert_to_lowercase', True):
            text = text.lower()
        
        # Suppression de la ponctuation excessive (adaptation du TP)
        text = re.sub(r'!{2,}', ' MULTI_EXCLAMATION ', text)
        text = re.sub(r'\?{2,}', ' MULTI_QUESTION ', text)
        
        # Suppression de la ponctuation
        if self.config.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Nettoyage des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str, language: str = "fr") -> str:
        """
        Supprime les mots vides en préservant les mots importants
        
        Args:
            text: Texte à traiter
            language: Langue du texte ('fr' ou 'en')
            
        Returns:
            Texte sans mots vides
        """
        if not self.config.get('remove_stopwords', True):
            return text
            
        # Sélection des stop words et mots importants selon la langue
        if language == "fr":
            stop_words = self.stop_words_fr - self.important_words_fr
        else:
            stop_words = self.stop_words_en - self.important_words_en
        
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatise le texte
        
        Args:
            text: Texte à lemmatiser
            
        Returns:
            Texte lemmalisé
        """
        if not self.config.get('lemmatize', True):
            return text
            
        try:
            blob = TextBlob(text)
            return ' '.join([word.lemmatize() for word in blob.words])
        except:
            # Version de secours avec NLTK
            words = word_tokenize(text.lower())
            lemmatized_words = []
            
            for word in words:
                if (word.isalpha() and 
                    len(word) >= self.config.get('min_word_length', 2) and 
                    len(word) <= self.config.get('max_word_length', 50)):
                    lemmatized_words.append(self.lemmatizer.lemmatize(word))
            
            return ' '.join(lemmatized_words)
    
    def preprocessing_pipeline(self, text: str, language: str = "fr") -> Tuple[str, Dict[str, Any]]:
        """
        Pipeline complet de préprocessing (inspiré du TP)
        
        Args:
            text: Texte brut à traiter
            language: Langue du texte
            
        Returns:
            Tuple (texte nettoyé, features extraites)
        """
        logger.debug(f"Début du preprocessing pour un texte de {len(text)} caractères")
        
        # 1. Extraction des features AVANT nettoyage (comme dans le TP)
        features = self.extract_text_features(text)
        
        # 2. Pipeline de nettoyage
        clean_text = self.clean_wikipedia_text(text)
        clean_text = self.remove_stopwords(clean_text, language)
        clean_text = self.lemmatize_text(clean_text)
        
        logger.debug(f"Fin du preprocessing, texte nettoyé: {len(clean_text)} caractères")
        
        return clean_text, features
    
    def preprocess_for_summarization(self, text: str, language: str = "fr") -> Dict[str, Any]:
        """
        Préprocessing spécialisé pour la summarisation
        
        Args:
            text: Texte brut de l'article
            language: Langue du texte
            
        Returns:
            Dict contenant le texte nettoyé et les métadonnées
        """
        # Pipeline complet
        clean_text, features = self.preprocessing_pipeline(text, language)
        
        # Extraction des phrases originales pour la summarisation
        original_sentences = sent_tokenize(text)
        
        # Calcul de scores de phrases basés sur les features
        sentence_scores = self._calculate_sentence_scores(original_sentences, features)
        
        result = {
            'clean_text': clean_text,
            'original_text': text,
            'features': features,
            'sentences': original_sentences,
            'sentence_scores': sentence_scores,
            'language': language
        }
        
        return result
    
    def _calculate_sentence_scores(self, sentences: List[str], global_features: Dict[str, Any]) -> List[float]:
        """
        Calcule un score pour chaque phrase basé sur différents critères
        
        Args:
            sentences: Liste des phrases
            global_features: Features globales du texte
            
        Returns:
            Liste des scores pour chaque phrase
        """
        scores = []
        
        for sentence in sentences:
            score = 0.0
            
            # Score basé sur la longueur (phrases ni trop courtes ni trop longues)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1.0
            elif 5 <= word_count < 10 or 30 < word_count <= 50:
                score += 0.5
            
            # Score basé sur la position (début et fin d'article plus importants)
            position_ratio = sentences.index(sentence) / len(sentences)
            if position_ratio <= 0.3 or position_ratio >= 0.7:
                score += 0.5
            
            # Score basé sur la présence de mots-clés numériques
            if re.search(r'\b\d+\b', sentence):
                score += 0.3
            
            scores.append(score)
        
        return scores 