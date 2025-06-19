#!/usr/bin/env python3
"""
Résumeur d'articles Wikipédia utilisant des techniques NLP
Basé sur les techniques de prétraitement du TP Pipeline NLP

Architecture modulaire:
1. DatasetBuilder: Création du dataset avec articles Wikipedia + résumés ChatGPT
2. TextPreprocessor: Nettoyage et prétraitement des textes
3. WikipediaSummarizer: Modèle de résumé automatique
4. Orchestrator: Pipeline principal
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajout du dossier src au path
sys.path.append('src')

from src.dataset_builder import DatasetBuilder
from src.text_preprocessor import TextPreprocessor
from src.wikipedia_summarizer import WikipediaSummarizer
from src.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Point d'entrée principal du programme"""
    parser = argparse.ArgumentParser(description="Résumeur d'articles Wikipédia")
    
    parser.add_argument(
        "--mode", 
        choices=["build_dataset", "train", "summarize", "evaluate"],
        default="summarize",
        help="Mode d'exécution du programme"
    )
    
    parser.add_argument(
        "--article",
        type=str,
        help="Titre de l'article Wikipedia à résumer (mode summarize)"
    )
    
    parser.add_argument(
        "--num_articles",
        type=int,
        default=100,
        help="Nombre d'articles pour construire le dataset (mode build_dataset)"
    )
    
    parser.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Langue des articles Wikipedia"
    )
    

    
    args = parser.parse_args()
    
    # Chargement de la configuration
    config = Config()
    
    # Le système utilise maintenant uniquement les résumés Wikipedia
    logger.info(f"Source des résumés: {config.get_summary_source()}")
    
    try:
        if args.mode == "build_dataset":
            logger.info(f"Construction du dataset avec {args.num_articles} articles en {args.lang}")
            builder = DatasetBuilder(config)
            dataset_path = builder.build_dataset(
                num_articles=args.num_articles,
                language=args.lang
            )
            logger.info(f"Dataset créé: {dataset_path}")
            
        elif args.mode == "train":
            logger.info("Entraînement du modèle de résumé")
            summarizer = WikipediaSummarizer(config)
            summarizer.train()
            logger.info("Entraînement terminé")
            
        elif args.mode == "summarize":
            if not args.article:
                logger.error("Veuillez spécifier un article avec --article")
                return
                
            logger.info(f"Résumé de l'article: {args.article}")
            summarizer = WikipediaSummarizer(config)
            
            # Récupération de l'article
            import wikipedia
            wikipedia.set_lang(args.lang)
            
            try:
                page = wikipedia.page(args.article)
                logger.info(f"Article récupéré: {page.title}")
                
                # Génération du résumé
                summary = summarizer.summarize(page.content)
                
                print(f"\n{'='*60}")
                print(f"RÉSUMÉ DE: {page.title}")
                print(f"{'='*60}")
                print(f"URL: {page.url}")
                print(f"\nRésumé original Wikipedia:")
                print(page.summary[:500] + "..." if len(page.summary) > 500 else page.summary)
                print(f"\nRésumé généré par notre modèle:")
                print(summary)
                print(f"{'='*60}")
                
            except wikipedia.exceptions.DisambiguationError as e:
                logger.error(f"Plusieurs articles trouvés. Suggestions: {e.options[:5]}")
            except wikipedia.exceptions.PageError:
                logger.error(f"Article '{args.article}' non trouvé")
                
        elif args.mode == "evaluate":
            logger.info("Évaluation du modèle")
            summarizer = WikipediaSummarizer(config)
            metrics = summarizer.evaluate()
            
            print(f"\n{'='*40}")
            print("MÉTRIQUES D'ÉVALUATION")
            print(f"{'='*40}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print(f"{'='*40}")
            
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise


if __name__ == "__main__":
    main() 