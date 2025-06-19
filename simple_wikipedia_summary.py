#!/usr/bin/env python3
"""
Résumeur d'articles Wikipedia utilisant le modèle entraîné
Combine wikipediaapi pour la récupération et le modèle transformers entraîné pour le résumé
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wikipediaapi
import os
import re
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WikipediaSummarizer:
    """
    Classe pour résumer des articles Wikipedia avec le modèle entraîné
    """
    
    def __init__(self, model_path="./models/summarization_wikipedia"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wiki = None
        self.load_model()
        self.setup_wikipedia_api()
    
    def load_model(self):
        """
        Charge le modèle et le tokenizer entraînés
        """
        logger.info(f"Chargement du modèle depuis: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            logger.error(f"Modèle non trouvé: {self.model_path}")
            logger.info("Veuillez d'abord entraîner le modèle avec train_summarization_model.py")
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Modèle chargé sur: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return False
    
    def setup_wikipedia_api(self):
        """
        Configure l'API Wikipedia
        """
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='NLP-Trained-Summarizer/1.0 (Educational use)',
            language='fr'  # Par défaut en français
        )
        logger.info("✅ API Wikipedia configurée")
    
    def get_article(self, title, language='fr'):
        """
        Récupère un article Wikipedia
        """
        logger.info(f"Récupération de l'article: {title} (langue: {language})")
        
        # Reconfigurer si nécessaire pour la langue
        if language != 'fr':
            self.wiki = wikipediaapi.Wikipedia(
                user_agent='NLP-Trained-Summarizer/1.0 (Educational use)',
                language=language
            )
        
        page = self.wiki.page(title)
        
        if not page.exists():
            logger.error(f"Article '{title}' non trouvé")
            return None
        
        logger.info(f"✅ Article récupéré: {len(page.text)} caractères")
        return {
            'title': page.title,
            'text': page.text,
            'url': page.fullurl,
            'length': len(page.text)
        }
    
    def preprocess_text(self, text):
        """
        Prétraite le texte pour le modèle
        """
        # Suppression des références et nettoyage basique
        text = re.sub(r'\[[0-9]*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Limitation de la longueur pour éviter les problèmes de mémoire
        max_chars = 4000  # Environ 512 tokens après tokenisation
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Texte tronqué à {max_chars} caractères")
        
        return text
    
    def generate_summary(self, text, max_length=128, min_length=30, num_beams=4):
        """
        Génère un résumé avec le modèle entraîné
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Modèle non chargé")
            return None
        
        logger.info("Génération du résumé avec le modèle entraîné...")
        
        # Prétraitement
        text = self.preprocess_text(text)
        
        # Tokenisation
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Génération
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False,
                temperature=1.0,
                length_penalty=1.0
            )
        
        # Décodage
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary.strip()
    
    def summarize_article(self, title, language='fr', max_length=128, min_length=30):
        """
        Résume complètement un article Wikipedia
        """
        # 1. Récupération de l'article
        article = self.get_article(title, language)
        if not article:
            return None
        
        # 2. Génération du résumé
        summary = self.generate_summary(
            article['text'], 
            max_length=max_length, 
            min_length=min_length
        )
        
        if not summary:
            return None
        
        # 3. Compilation des résultats
        result = {
            'title': article['title'],
            'url': article['url'],
            'original_length': article['length'],
            'summary': summary,
            'summary_length': len(summary),
            'compression_ratio': len(summary) / article['length']
        }
        
        return result

def interactive_mode():
    """
    Mode interactif pour résumer des articles
    """
    summarizer = WikipediaSummarizer()
    
    if summarizer.model is None:
        print("❌ Impossible de charger le modèle")
        return
    
    print("🤖 Résumeur d'articles Wikipedia avec modèle entraîné")
    print("Tapez 'quit' pour quitter\n")
    
    while True:
        title = input("📝 Titre de l'article Wikipedia: ").strip()
        
        if title.lower() in ['quit', 'exit', 'q']:
            print("Au revoir! 👋")
            break
        
        if not title:
            print("⚠️ Veuillez entrer un titre d'article")
            continue
        
        # Paramètres optionnels
        language = input("🌍 Langue (fr/en) [fr]: ").strip().lower()
        if language not in ['fr', 'en']:
            language = 'fr'
        
        try:
            max_len = input("📏 Longueur max du résumé [128]: ").strip()
            max_len = int(max_len) if max_len else 128
            
            min_len = input("📏 Longueur min du résumé [30]: ").strip()
            min_len = int(min_len) if min_len else 30
            
        except ValueError:
            print("⚠️ Paramètres invalides, utilisation des valeurs par défaut")
            max_len, min_len = 128, 30
        
        print(f"\n{'='*80}")
        
        # Génération du résumé
        try:
            result = summarizer.summarize_article(
                title, 
                language=language, 
                max_length=max_len, 
                min_length=min_len
            )
            
            if result:
                print(f"📰 Article: {result['title']}")
                print(f"🔗 URL: {result['url']}")
                print(f"📏 Taille originale: {result['original_length']:,} caractères")
                print(f"📏 Taille résumé: {result['summary_length']:,} caractères")
                print(f"📉 Taux de compression: {result['compression_ratio']:.1%}")
                print(f"\n📋 Résumé généré par le modèle:")
                print("-" * 60)
                print(result['summary'])
                print("-" * 60)
            else:
                print("❌ Impossible de générer le résumé")
                
        except Exception as e:
            logger.error(f"Erreur lors du résumé: {e}")
            print(f"❌ Erreur: {e}")
        
        print(f"\n{'='*80}\n")

def demo_articles():
    """
    Démonstrateur avec quelques articles prédéfinis
    """
    demo_titles = [
        "Intelligence artificielle",
        "Réchauffement climatique", 
        "Python (langage)",
        "Machine learning",
        "Révolution française"
    ]
    
    summarizer = WikipediaSummarizer()
    
    if summarizer.model is None:
        print("❌ Impossible de charger le modèle")
        return
    
    print("🚀 Démonstration avec des articles prédéfinis\n")
    
    for title in demo_titles:
        print(f"{'='*80}")
        print(f"📰 Traitement: {title}")
        print(f"{'='*80}")
        
        try:
            result = summarizer.summarize_article(title, language='fr', max_length=100)
            
            if result:
                print(f"✅ Résumé généré pour: {result['title']}")
                print(f"📏 {result['original_length']:,} → {result['summary_length']:,} caractères ({result['compression_ratio']:.1%})")
                print(f"\n📋 Résumé:")
                print("-" * 60)
                print(result['summary'])
                print("-" * 60)
            else:
                print(f"❌ Échec pour: {title}")
                
        except Exception as e:
            print(f"❌ Erreur pour {title}: {e}")
        
        print("\n")

def main():
    """
    Fonction principale
    """
    print("=== Résumeur Wikipedia avec Modèle Entraîné ===\n")
    
    choice = input("""
Choisissez le mode:
1. Mode interactif
2. Démonstration avec articles prédéfinis
3. Les deux

Votre choix [1]: """).strip()
    
    if choice == "2":
        demo_articles()
    elif choice == "3":
        demo_articles()
        print("\n" + "="*80)
        print("Passage en mode interactif...")
        print("="*80 + "\n")
        interactive_mode()
    else:
        interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterruption utilisateur. Au revoir! 👋")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        print(f"❌ Erreur inattendue: {e}") 