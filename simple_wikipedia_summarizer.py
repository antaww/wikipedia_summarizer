#!/usr/bin/env python3
"""
Script simple pour récupérer un article Wikipedia et le résumer
Utilise wikipediaapi + pipeline de preprocessing + NLTK
"""

import sys
import os
import logging
import heapq
from typing import List, Dict

# Ajout du dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from config import Config
from text_preprocessor import TextPreprocessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_wikipedia_article(title: str, language: str = "fr") -> Dict[str, str]:
    """
    Récupère un article Wikipedia
    """
    logger.info(f"Récupération de l'article: {title} (langue: {language})")
    
    # Initialisation de l'API Wikipedia avec user agent approprié
    wiki = wikipediaapi.Wikipedia(
        user_agent='NLP-Pipeline-Summarizer/1.0 (Educational use)',
        language=language
    )
    page = wiki.page(title)
    
    if not page.exists():
        logger.error(f"Article '{title}' non trouvé")
        return {"error": f"Article '{title}' non trouvé"}
    
    logger.info(f"Article récupéré: {len(page.text)} caractères")
    return {
        "title": page.title,
        "text": page.text,
        "url": page.fullurl,
        "length": len(page.text)
    }

def summarize_with_nltk(sentences: List[str], word_frequencies: Dict[str, float], num_sentences: int = 2) -> str:
    """
    Résume en utilisant NLTK avec fréquences pondérées
    """
    logger.info(f"Résumé NLTK: {len(sentences)} phrases -> {num_sentences} phrases")
    
    if len(sentences) <= num_sentences:
        return '. '.join(sentences)
    
    # Calcul des scores des phrases
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum()]
        
        if len(words) < 30:  # Éviter les phrases trop longues
            sentence_score = 0
            word_count = 0
            
            for word in words:
                if word in word_frequencies:
                    sentence_score += word_frequencies[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[i] = sentence_score / word_count
    
    # Sélection des meilleures phrases
    if not sentence_scores:
        return '. '.join(sentences[:num_sentences])
    
    best_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    best_indices.sort()  # Conserver l'ordre original
    
    summary_sentences = [sentences[i] for i in best_indices]
    return ' '.join(summary_sentences)

def calculate_word_frequencies(text: str, language: str = "fr") -> Dict[str, float]:
    """
    Calcule les fréquences pondérées des mots
    """
    logger.info("Calcul des fréquences des mots")
    
    # Téléchargement des ressources NLTK si nécessaire
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        logger.warning("Impossible de télécharger les ressources NLTK")
    
    # Stop words
    try:
        if language == "fr":
            stop_words = set(stopwords.words('french'))
        else:
            stop_words = set(stopwords.words('english'))
    except:
        logger.warning(f"Stop words non disponibles pour {language}")
        stop_words = set()
    
    # Tokenisation et nettoyage
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calcul des fréquences
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Fréquences pondérées (division par le max)
    if word_freq:
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
    
    logger.info(f"Fréquences calculées pour {len(word_freq)} mots uniques")
    return word_freq

def main():
    """
    Fonction principale
    """
    print("=== Résumeur Wikipedia Simple ===\n")
    
    # Configuration
    config = Config()
    preprocessor = TextPreprocessor(config.PREPROCESSING_CONFIG)
    
    # Demander l'article à l'utilisateur
    article_title = input("📝 Nom de l'article Wikipedia: ").strip()
    if not article_title:
        print("❌ Nom d'article requis")
        return
    
    language = input("🌍 Langue (fr/en) [fr]: ").strip().lower()
    if language not in ['fr', 'en']:
        language = 'fr'
    
    try:
        num_sentences = int(input("📊 Nombre de phrases dans le résumé [2]: ") or "2")
    except ValueError:
        num_sentences = 2
    
    print(f"\n🔍 Récupération de '{article_title}'...")
    
    # 1. Récupération de l'article
    article = get_wikipedia_article(article_title, language)
    if "error" in article:
        print(f"❌ {article['error']}")
        return
    
    print(f"✅ Article récupéré: {article['length']:,} caractères")
    
    # 2. Preprocessing avec votre pipeline
    print("🔧 Preprocessing du texte...")
    processed_data = preprocessor.preprocess_for_summarization(article['text'], language)
    
    print(f"✅ Preprocessing terminé:")
    print(f"   - {len(processed_data['sentences'])} phrases détectées")
    print(f"   - Features extraites: {list(processed_data['features'].keys())}")
    
    # 3. Calcul des fréquences pondérées
    word_frequencies = calculate_word_frequencies(processed_data['clean_text'], language)
    
    # 4. Résumé avec NLTK
    print("📝 Génération du résumé...")
    summary = summarize_with_nltk(
        processed_data['sentences'], 
        word_frequencies, 
        num_sentences
    )
    
    # 5. Affichage des résultats
    print(f"\n{'='*60}")
    print(f"📰 Article: {article['title']}")
    print(f"🔗 URL: {article['url']}")
    print(f"📊 Taille: {article['length']:,} caractères")
    print(f"📉 Résumé: {len(summary):,} caractères ({len(summary)/article['length']:.1%})")
    print(f"\n📋 Résumé en {num_sentences} phrase(s):")
    print("-" * 60)
    print(summary)
    print("-" * 60)
    
    # 6. Affichage des features du preprocessing
    print(f"\n🔍 Features du preprocessing:")
    for feature, value in processed_data['features'].items():
        if isinstance(value, float):
            print(f"   - {feature}: {value:.3f}")
        else:
            print(f"   - {feature}: {value}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAu revoir! 👋")
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        print(f"❌ Erreur inattendue: {str(e)}") 