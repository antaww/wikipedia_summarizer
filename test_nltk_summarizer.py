#!/usr/bin/env python3
"""
Script de test pour le résumeur Wikipedia avec NLTK
Démontre la récupération et résumé d'articles Wikipedia en 2 phrases
"""

import logging
import sys
import os

# Ajout du dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from wikipedia_summarizer import WikipediaSummarizer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_nltk_summarizer():
    """
    Teste le résumeur NLTK avec différents articles Wikipedia
    """
    print("=== Test du résumeur Wikipedia avec NLTK ===\n")
    
    # Initialisation
    config = Config()
    summarizer = WikipediaSummarizer(config)
    
    # Articles de test
    test_articles = [
        ("Intelligence artificielle", "fr"),
        ("Machine learning", "fr"), 
        ("Python (programming language)", "en"),
        ("Informatique", "fr")
    ]
    
    for article_title, language in test_articles:
        print(f"\n{'='*60}")
        print(f"Article: {article_title} (langue: {language})")
        print('='*60)
        
        try:
            # Récupération et résumé
            result = summarizer.get_wikipedia_article_and_summarize_nltk(
                article_title=article_title,
                language=language,
                sentences_count=2
            )
            
            if "error" in result:
                print(f"❌ Erreur: {result['error']}")
                continue
            
            # Affichage des résultats
            print(f"📰 Titre: {result['title']}")
            print(f"🔗 URL: {result['url']}")
            print(f"📊 Taille originale: {result['original_length']:,} caractères")
            print(f"📝 Taille résumé: {result['summary_length']:,} caractères")
            print(f"📉 Ratio de compression: {result['compression_ratio']:.1%}")
            print(f"\n📋 Résumé en 2 phrases:")
            print("-" * 40)
            print(result['summary'])
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Erreur inattendue: {str(e)}")

def test_interactive_mode():
    """
    Mode interactif pour tester avec des articles personnalisés
    """
    print("\n=== Mode interactif ===")
    print("Entrez le nom d'un article Wikipedia pour le résumer")
    print("Tapez 'quit' pour quitter\n")
    
    config = Config()
    summarizer = WikipediaSummarizer(config)
    
    while True:
        try:
            article_name = input("📝 Nom de l'article (ou 'quit'): ").strip()
            
            if article_name.lower() in ['quit', 'exit', 'q']:
                print("Au revoir! 👋")
                break
            
            if not article_name:
                continue
            
            # Demander la langue
            language = input("🌍 Langue (fr/en) [fr]: ").strip().lower()
            if language not in ['fr', 'en']:
                language = 'fr'
            
            # Demander le nombre de phrases
            try:
                sentences_count = int(input("📊 Nombre de phrases [2]: ") or "2")
            except ValueError:
                sentences_count = 2
            
            print(f"\n🔍 Recherche et résumé de '{article_name}'...")
            
            result = summarizer.get_wikipedia_article_and_summarize_nltk(
                article_title=article_name,
                language=language,
                sentences_count=sentences_count
            )
            
            if "error" in result:
                print(f"❌ {result['error']}")
                continue
            
            print(f"\n✅ Résumé de '{result['title']}':")
            print(f"📊 {result['original_length']:,} caractères → {result['summary_length']:,} caractères ({result['compression_ratio']:.1%})")
            print(f"\n{result['summary']}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nAu revoir! 👋")
            break
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    # Test automatique
    test_nltk_summarizer()
    
    # Mode interactif
    try:
        user_input = input("\n🤖 Voulez-vous essayer le mode interactif? (y/n): ").strip().lower()
        if user_input in ['y', 'yes', 'oui', 'o']:
            test_interactive_mode()
    except KeyboardInterrupt:
        print("\nAu revoir! 👋") 