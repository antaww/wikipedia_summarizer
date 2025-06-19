#!/usr/bin/env python3
"""
Résumeur d'articles Wikipédia utilisant des techniques NLP
Pipeline complet : preprocessing → dataset → entraînement → résumé avec modèle entraîné

Modes disponibles:
- build_dataset: Construction du dataset d'entraînement
- train: Entraînement du modèle Transformer
- summarize: Résumé avec le modèle entraîné (différent du résumé Wikipedia pour dataset)
- interactive: Mode interactif de résumé
- demo: Démonstration avec articles prédéfinis
- visualize: Visualisation du preprocessing
- install_spacy: Installation des modèles spaCy
- evaluate: Évaluation du modèle
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajout du dossier src au path
sys.path.append('src')

from src.dataset_builder import DatasetBuilder
from src.text_preprocessor import TextPreprocessor
from src.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_dataset_mode(args):
    """Mode construction du dataset"""
    logger.info(f"Construction du dataset avec {args.num_articles} articles en {args.lang}")
    
    config = Config()
    builder = DatasetBuilder(config)
    dataset_path = builder.build_dataset(
        num_articles=args.num_articles,
        language=args.lang
    )
    logger.info(f"Dataset créé: {dataset_path}")


def train_mode(args):
    """Mode entraînement du modèle"""
    logger.info("Entraînement du modèle de résumé Transformer")
    
    from src.train_summarization_model import train_summarization_model
    train_summarization_model()
    logger.info("Entraînement terminé")


def summarize_mode(args):
    """Mode résumé avec le modèle entraîné"""
    if not args.article:
        logger.error("Veuillez spécifier un article avec --article")
        return
    
    logger.info(f"Résumé de l'article avec le modèle entraîné: {args.article}")
    
    from src.wikipedia_summarizer_trained import WikipediaSummarizerTrained
    
    # Initialisation du résumeur avec modèle entraîné
    summarizer = WikipediaSummarizerTrained()
    
    if summarizer.model is None:
        logger.error("Modèle entraîné non trouvé. Veuillez d'abord entraîner le modèle avec --mode train")
        return
    
    try:
        # Génération du résumé avec le modèle entraîné
        result = summarizer.summarize_article(
            args.article, 
            language=args.lang,
            max_length=args.max_length,
            min_length=args.min_length
        )
        
        if result:
            print(f"\n{'='*80}")
            print(f"📰 RÉSUMÉ GÉNÉRÉ PAR LE MODÈLE ENTRAÎNÉ")
            print(f"{'='*80}")
            print(f"🔗 Article: {result['title']}")
            print(f"🌐 URL: {result['url']}")
            print(f"📏 Longueur originale: {result['original_length']:,} caractères")
            print(f"📏 Longueur résumé: {result['summary_length']:,} caractères")
            print(f"📉 Taux de compression: {result['compression_ratio']:.1%}")
            print(f"\n📋 Résumé généré:")
            print("-" * 60)
            print(result['summary'])
            print("-" * 60)
            print(f"{'='*80}")
        else:
            print("❌ Impossible de générer le résumé")
            
    except Exception as e:
        logger.error(f"Erreur lors du résumé: {e}")


def interactive_mode(args):
    """Mode interactif"""
    logger.info("Lancement du mode interactif")
    
    from src.wikipedia_summarizer_trained import interactive_mode
    interactive_mode()


def demo_mode(args):
    """Mode démonstration"""
    logger.info("Lancement de la démonstration")
    
    from src.wikipedia_summarizer_trained import demo_articles
    demo_articles()


def visualize_mode(args):
    """Mode visualisation du preprocessing"""
    logger.info("Lancement de la visualisation du preprocessing")
    
    from src.visualize_preprocessing import visualize_preprocessing
    visualize_preprocessing(args.article, args.lang)


def install_spacy_mode(args):
    """Mode installation des modèles spaCy"""
    logger.info("Installation des modèles spaCy")
    
    from src.install_spacy_models import install_spacy_models
    install_spacy_models()


def evaluate_mode(args):
    """Mode évaluation du modèle"""
    logger.info("Évaluation du modèle")
    
    from src.wikipedia_summarizer_trained import WikipediaSummarizerTrained
    
    summarizer = WikipediaSummarizerTrained()
    
    if summarizer.model is None:
        logger.error("Modèle entraîné non trouvé. Veuillez d'abord entraîner le modèle.")
        return
    
    print(f"\n{'='*60}")
    print("📊 ÉVALUATION DU MODÈLE")
    print(f"{'='*60}")
    print("✅ Modèle chargé avec succès")
    print(f"🖥️  Device: {summarizer.device}")
    print(f"📁 Chemin du modèle: {summarizer.model_path}")
    print(f"{'='*60}")


def main():
    """Point d'entrée principal du programme"""
    parser = argparse.ArgumentParser(
        description="Résumeur d'articles Wikipédia avec modèle Transformer entraîné",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # 1. Installer les modèles spaCy
  python main.py --mode install_spacy
  
  # 2. Construire un dataset
  python main.py --mode build_dataset --num_articles 100 --lang fr
  
  # 3. Entraîner le modèle
  python main.py --mode train
  
  # 4. Résumer un article avec le modèle entraîné
  python main.py --mode summarize --article "Intelligence artificielle" --lang fr
  
  # 5. Mode interactif
  python main.py --mode interactive
  
  # 6. Démonstration
  python main.py --mode demo
  
  # 7. Visualiser le preprocessing
  python main.py --mode visualize --article "Python (langage)"
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=[
            "build_dataset", "train", "summarize", "interactive", 
            "demo", "visualize", "install_spacy", "evaluate"
        ],
        default="interactive",
        help="Mode d'exécution du programme"
    )
    
    parser.add_argument(
        "--article",
        type=str,
        help="Titre de l'article Wikipedia à résumer ou analyser"
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
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Longueur maximale du résumé généré"
    )
    
    parser.add_argument(
        "--min_length",
        type=int,
        default=30,
        help="Longueur minimale du résumé généré"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🤖 RÉSUMEUR D'ARTICLES WIKIPEDIA - PIPELINE NLP COMPLET")
    print("=" * 80)
    print(f"🔧 Mode sélectionné: {args.mode}")
    print(f"🌍 Langue: {args.lang}")
    if args.article:
        print(f"📄 Article: {args.article}")
    print("=" * 80)
    
    try:
        # Dispatch vers la fonction appropriée selon le mode
        if args.mode == "build_dataset":
            build_dataset_mode(args)
            
        elif args.mode == "train":
            train_mode(args)
            
        elif args.mode == "summarize":
            summarize_mode(args)
            
        elif args.mode == "interactive":
            interactive_mode(args)
            
        elif args.mode == "demo":
            demo_mode(args)
            
        elif args.mode == "visualize":
            visualize_mode(args)
            
        elif args.mode == "install_spacy":
            install_spacy_mode(args)
            
        elif args.mode == "evaluate":
            evaluate_mode(args)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interruption utilisateur. Au revoir! 👋")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"\n❌ Erreur: {e}")
        print("\n💡 Conseil: Vérifiez les prérequis et que les modèles sont bien installés.")
        raise


if __name__ == "__main__":
    main() 