#!/usr/bin/env python3
"""
Script de visualisation de la pipeline de preprocessing
Récupère un article Wikipedia et montre chaque étape de preprocessing
"""

import sys
import logging
from pathlib import Path
import wikipediaapi
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import depuis le même dossier
from .config import Config
from .text_preprocessor import TextPreprocessor

def print_section(title: str, content: str, max_chars: int = 500):
    """Affiche une section avec un titre et du contenu limité"""
    print(f"\n{'=' * 60}")
    print(f"📋 {title}")
    print(f"{'=' * 60}")
    
    if len(content) > max_chars:
        print(f"{content[:max_chars]}...")
        print(f"\n[Tronqué - Longueur totale: {len(content)} caractères]")
    else:
        print(content)
    
    print(f"\n📊 Statistiques: {len(content)} caractères, {len(content.split())} mots")


def print_comparison(before: str, after: str, step_name: str, max_chars: int = 300):
    """Affiche une comparaison avant/après pour une étape"""
    print(f"\n{'🔄 ' + step_name:-^60}")
    
    print(f"\n🔴 AVANT ({len(before)} caractères):")
    if len(before) > max_chars:
        print(f"{before[:max_chars]}...")
    else:
        print(before)
    
    print(f"\n🟢 APRÈS ({len(after)} caractères):")
    if len(after) > max_chars:
        print(f"{after[:max_chars]}...")
    else:
        print(after)
    
    # Statistiques de changement
    removed_chars = len(before) - len(after)
    if removed_chars > 0:
        reduction_percent = (removed_chars / len(before)) * 100
        print(f"\n📉 Réduction: -{removed_chars} caractères (-{reduction_percent:.1f}%)")
    else:
        print(f"\n📈 Modification: {abs(removed_chars)} caractères")


def get_wikipedia_article(title: str, language: str = "fr") -> dict:
    """Récupère un article Wikipedia"""
    print(f"🌐 Récupération de l'article Wikipedia: '{title}' (langue: {language})")
    
    # Initialiser l'API Wikipedia
    wiki = wikipediaapi.Wikipedia('NLP-Preprocessing-Demo (demo@example.com)', language)
    
    page = wiki.page(title)
    
    if not page.exists():
        raise ValueError(f"❌ Article '{title}' non trouvé en {language}")
    
    print(f"✅ Article trouvé: {page.title}")
    print(f"🔗 URL: {page.fullurl}")
    
    return {
        'title': page.title,
        'url': page.fullurl,
        'content': page.text,
        'summary': page.summary,
        'language': language
    }


def demonstrate_step_by_step_preprocessing(text: str, language: str = "fr"):
    """Démontre chaque étape du preprocessing individuellement"""
    
    print(f"\n{'🔬 ANALYSE ÉTAPE PAR ÉTAPE':-^80}")
    
    # Configuration de preprocessing pour permettre chaque étape
    config = Config()
    
    # Étape 1: Nettoyage des références Wikipedia
    step1_text = re.sub(r'\[\d+\]', '', text)
    print_comparison(text[:500], step1_text[:500], "ÉTAPE 1: Suppression des références [1], [2], etc.")
    
    # Étape 2: Suppression des balises Wikipedia
    step2_text = re.sub(r'{{[^}]*}}', '', step1_text)
    step2_text = re.sub(r'{[^}]*}', '', step2_text)
    print_comparison(step1_text[:500], step2_text[:500], "ÉTAPE 2: Suppression des balises Wikipedia")
    
    # Étape 3: Suppression des URLs
    step3_text = re.sub(r'(https?://\S+|www\.\S+|\S+\.(com|org|net|fr|en)\S*)', ' URL_TOKEN ', step2_text)
    print_comparison(step2_text[:500], step3_text[:500], "ÉTAPE 3: Remplacement des URLs")
    
    # Étape 4: Conversion en minuscules
    step4_text = step3_text.lower()
    print_comparison(step3_text[:500], step4_text[:500], "ÉTAPE 4: Conversion en minuscules")
    
    # Étape 5: Suppression de la ponctuation
    step5_text = step4_text.translate(str.maketrans('', '', string.punctuation))
    print_comparison(step4_text[:500], step5_text[:500], "ÉTAPE 5: Suppression de la ponctuation")
    
    # Étape 6: Nettoyage des espaces multiples
    step6_text = re.sub(r'\s+', ' ', step5_text).strip()
    print_comparison(step5_text[:500], step6_text[:500], "ÉTAPE 6: Nettoyage des espaces")
    
    # Étape 7: Tokenisation
    tokens = word_tokenize(step6_text)
    print(f"\n{'🔄 ÉTAPE 7: Tokenisation':-^60}")
    print(f"🔴 AVANT: Texte continu")
    print(f"🟢 APRÈS: {len(tokens)} tokens")
    print(f"👀 Premiers 20 tokens: {tokens[:20]}")
    
    # Étape 8: Suppression des stopwords
    processor = TextPreprocessor(config.PREPROCESSING_CONFIG)
    if language == "fr":
        stop_words = processor.stop_words_fr - processor.important_words_fr
    else:
        stop_words = processor.stop_words_en - processor.important_words_en
    
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    print(f"\n{'🔄 ÉTAPE 8: Suppression des stopwords':-^60}")
    print(f"🔴 AVANT: {len(tokens)} tokens")
    print(f"🟢 APRÈS: {len(filtered_tokens)} tokens")
    print(f"📉 Supprimés: {len(tokens) - len(filtered_tokens)} stopwords")
    print(f"👀 Premiers 20 tokens filtrés: {filtered_tokens[:20]}")
    
    # Étape 9: Lemmatisation
    from textblob import TextBlob
    text_for_lemma = ' '.join(filtered_tokens)
    blob = TextBlob(text_for_lemma)
    lemmatized_words = [word.lemmatize() for word in blob.words]
    
    print(f"\n{'🔄 ÉTAPE 9: Lemmatisation':-^60}")
    print(f"🔴 AVANT: {filtered_tokens[:10]}")
    print(f"🟢 APRÈS: {lemmatized_words[:10]}")
    
    # Texte final
    final_text = ' '.join(lemmatized_words)
    
    return final_text


def demonstrate_full_pipeline(text: str, language: str = "fr"):
    """Démontre la pipeline complète via TextPreprocessor"""
    print(f"\n{'🚀 PIPELINE COMPLÈTE':-^80}")
    
    config = Config()
    processor = TextPreprocessor(config.PREPROCESSING_CONFIG)
    
    # Pipeline complète
    clean_text, features = processor.preprocessing_pipeline(text, language)
    
    print_section("TEXTE ORIGINAL", text)
    print_section("TEXTE APRÈS PIPELINE COMPLÈTE", clean_text)
    
    # Affichage des features extraites
    print(f"\n{'📊 FEATURES EXTRAITES':-^60}")
    for feature, value in features.items():
        if isinstance(value, float):
            print(f"📈 {feature}: {value:.3f}")
        else:
            print(f"📈 {feature}: {value}")


def visualize_preprocessing(article_title: str = None, language: str = "fr"):
    """Fonction principale de visualisation"""
    print("🎯 VISUALISATION DE LA PIPELINE DE PREPROCESSING")
    print("=" * 80)
    
    # Paramètres par défaut
    default_article = "Intelligence artificielle"
    default_language = "fr"
    
    if not article_title:
        # Demander à l'utilisateur de choisir un article
        print(f"\nArticle par défaut: '{default_article}' (langue: {default_language})")
        choice = input("📝 Appuyez sur Entrée pour continuer ou tapez un autre titre d'article: ").strip()
        
        if choice:
            article_title = choice
        else:
            article_title = default_article
        
        # Choix de la langue
        lang_choice = input("🌍 Langue (fr/en) [fr]: ").strip().lower()
        language = lang_choice if lang_choice in ['fr', 'en'] else default_language
    
    try:
        # Récupération de l'article
        article = get_wikipedia_article(article_title, language)
        
        # Limitation du texte pour la démonstration
        text = article['content'][:3000]  # Premiers 3000 caractères
        
        print(f"\n📄 Article sélectionné: {article['title']}")
        print(f"📏 Texte analysé: {len(text)} caractères (limité pour la démonstration)")
        
        # Démonstration étape par étape
        final_text = demonstrate_step_by_step_preprocessing(text, language)
        
        # Démonstration de la pipeline complète
        demonstrate_full_pipeline(text, language)
        
        # Résumé final
        print(f"\n{'✅ RÉSUMÉ FINAL':-^80}")
        print(f"📊 Texte original: {len(text)} caractères")
        print(f"📊 Texte final: {len(final_text)} caractères") 
        print(f"📉 Réduction totale: {((len(text) - len(final_text)) / len(text) * 100):.1f}%")
        
        print(f"\n🎉 Démonstration terminée avec succès !")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    visualize_preprocessing() 