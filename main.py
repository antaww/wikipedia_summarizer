#!/usr/bin/env python3
"""
🤖 Pipeline NLP Complet - Auto-encodeur + Clustering + Résumé Automatique
Architecture : TF-IDF → Auto-encodeur (32D) → K-means → Analyse intelligente

Modes disponibles:
- build_dataset: Construction du dataset d'entraînement
- train: Entraînement auto-encodeur + clustering (TensorFlow)
- summarize: Analyse et résumé d'un article avec le pipeline entraîné
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
    """Mode entraînement auto-encodeur + clustering (TensorFlow)"""
    logger.info("Entraînement du pipeline Auto-encodeur + Clustering")
    
    try:
        from src.autoencoder_clustering import run_autoencoder_clustering_pipeline
        
        # Configuration selon les arguments
        config = {
            'max_features': args.max_features,
            'encoding_dim': args.encoding_dim,
            'n_clusters': args.n_clusters,
            'epochs': args.epochs,
            'batch_size': getattr(args, 'batch_size', 32),
            'random_state': 42,
            'models_dir': 'models'
        }
        
        print(f"\n{'='*80}")
        print("🧠 ENTRAÎNEMENT AUTO-ENCODEUR + CLUSTERING")
        print(f"{'='*80}")
        print(f"🔤 Features TF-IDF max: {config['max_features']}")
        print(f"🧠 Dimensions encodage: {config['encoding_dim']}")
        print(f"🎯 Nombre de clusters: {config['n_clusters']}")
        print(f"⏰ Époques: {config['epochs']}")
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"{'='*80}")
        
        # Vérification du dataset
        dataset_path = "data/wikipedia_dataset_fr.csv"
        if not Path(dataset_path).exists():
            logger.error(f"❌ Dataset non trouvé: {dataset_path}")
            logger.info("💡 Construisez d'abord le dataset avec: python main.py --mode build_dataset")
            return
        
        # Suppression des anciens modèles si ils existent
        old_models = [
            'models/autoencoder.h5',
            'models/encoder.h5'
        ]
        
        for model_path in old_models:
            if Path(model_path).exists():
                Path(model_path).unlink()
                logger.info(f"🗑️  Suppression ancien modèle: {model_path}")
        
        # Lancement de l'entraînement
        logger.info("🚀 Démarrage de l'entraînement...")
        
        results = run_autoencoder_clustering_pipeline(
            dataset_path,
            max_features=config['max_features'],
            encoding_dim=config['encoding_dim'],
            n_clusters=config['n_clusters'],
            epochs=config['epochs'],
            batch_size=config['batch_size']
        )
        
        # Affichage des résultats
        print(f"\n{'='*80}")
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"{'='*80}")
        
        # Métriques
        metrics = results['clustering_metrics']
        print(f"📊 Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"📊 Inertie: {metrics['inertia']:.1f}")
        
        # Fichiers générés
        print(f"\n💾 Modèles sauvegardés:")
        print(f"   - models/autoencoder.weights.h5")
        print(f"   - models/encoder.weights.h5")
        print(f"   - models/autoencoder_config.pkl")
        print(f"   - models/encoder_config.pkl")
        print(f"   - models/tfidf_vectorizer.pkl")
        print(f"   - models/kmeans_model.pkl")
        print(f"   - models/pipeline_results.pkl")
        
        # Prochaines étapes
        print(f"\n💡 Prochaines étapes:")
        print(f"   - Tester: python main.py --mode analyze --article \"Titre Article\"")
        print(f"   - Analyser: python analyze_keywords_and_summarize.py \"Titre\"")
        
        logger.info("Entraînement auto-encodeur terminé")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        raise


def summarize_mode(args):
    """Mode analyse et résumé avec le pipeline auto-encodeur"""
    if not args.article:
        logger.error("Veuillez spécifier un article avec --article")
        return
    
    logger.info(f"Analyse et résumé de l'article: {args.article}")
    
    try:
        import sys
        import os
        
        # Exécution du script d'analyse
        print(f"\n{'='*80}")
        print("🔍 ANALYSE ET RÉSUMÉ AUTOMATIQUE")
        print(f"{'='*80}")
        
        # Import du module d'analyse
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        from analyze_keywords_and_summarize import analyze_article_keywords
        
        # Lancement de l'analyse
        analyze_article_keywords(args.article)
        
        print(f"\n{'='*80}")
        print("✅ ANALYSE TERMINÉE!")
        print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {e}")
        print("💡 Assurez-vous que les modèles sont entraînés avec --mode train")
        raise


def interactive_mode(args):
    """Mode interactif de résumé"""
    logger.info("Lancement du mode interactif")
    
    print("\n🎮 MODE INTERACTIF - RÉSUMÉ D'ARTICLES")
    print("="*60)
    
    try:
        from analyze_keywords_and_summarize import analyze_article_keywords
        
        while True:
            print("\n💬 Entrez le titre d'un article Wikipedia à analyser:")
            print("   (ou 'quit' pour quitter)")
            
            article_title = input("👉 Titre: ").strip()
            
            if article_title.lower() in ['quit', 'q', 'exit']:
                print("👋 Au revoir!")
                break
            
            if not article_title:
                print("❌ Titre vide, veuillez réessayer")
                continue
            
            try:
                print(f"\n🔍 Analyse de '{article_title}'...")
                analyze_article_keywords(article_title)
                print("\n" + "="*60)
                
            except KeyboardInterrupt:
                print("\n⏸️  Analyse interrompue")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                print("💡 Essayez un autre article")
        
    except Exception as e:
        logger.error(f"Erreur mode interactif: {e}")


def demo_mode(args):
    """Mode démonstration avec articles prédéfinis"""
    logger.info("Lancement de la démonstration")
    
    # Articles d'exemple
    demo_articles = [
        "Intelligence artificielle",
        "Python (langage)", 
        "Chimie",
        "Informatique",
        "Machine learning"
    ]
    
    print("\n🎪 DÉMONSTRATION - RÉSUMÉ D'ARTICLES")
    print("="*60)
    print(f"📚 {len(demo_articles)} articles d'exemple")
    
    try:
        from analyze_keywords_and_summarize import analyze_article_keywords
        
        for i, article in enumerate(demo_articles, 1):
            print(f"\n{'='*80}")
            print(f"📖 ARTICLE {i}/{len(demo_articles)}: {article}")
            print(f"{'='*80}")
            
            try:
                analyze_article_keywords(article)
                
                if i < len(demo_articles):
                    input("\n⏸️  Appuyez sur Entrée pour continuer...")
                    
            except KeyboardInterrupt:
                print("\n⏹️  Démonstration interrompue")
                break
            except Exception as e:
                print(f"❌ Erreur pour '{article}': {e}")
                continue
        
        print(f"\n🎉 Démonstration terminée!")
        
    except Exception as e:
        logger.error(f"Erreur démonstration: {e}")


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
    """Mode évaluation du pipeline"""
    logger.info("Évaluation du pipeline auto-encodeur")
    
    try:
        import pickle
        import numpy as np
        from pathlib import Path
        
        print(f"\n{'='*60}")
        print("📊 ÉVALUATION DU PIPELINE")
        print(f"{'='*60}")
        
        # Vérification des modèles
        model_files = [
            'models/autoencoder.weights.h5',
            'models/encoder.weights.h5',
            'models/tfidf_vectorizer.pkl',
            'models/kmeans_model.pkl',
            'models/pipeline_results.pkl'
        ]
        
        missing_files = [f for f in model_files if not Path(f).exists()]
        
        if missing_files:
            print("❌ Modèles manquants:")
            for f in missing_files:
                print(f"   - {f}")
            print("💡 Exécutez: python main.py --mode train")
            return
        
        # Chargement des résultats
        with open('models/pipeline_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print("✅ Tous les modèles sont présents")
        
        # Métriques
        metrics = results['clustering_metrics']
        tfidf_shape = results['tfidf_matrix'].shape
        encoded_shape = results['encoded_vectors'].shape
        
        print(f"\n📊 MÉTRIQUES DU PIPELINE:")
        print(f"   - Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"   - Inertie K-means: {metrics['inertia']:.1f}")
        print(f"   - Documents traités: {tfidf_shape[0]}")
        print(f"   - Compression: {tfidf_shape[1]} → {encoded_shape[1]} dims")
        print(f"   - Taux de compression: {(encoded_shape[1]/tfidf_shape[1])*100:.1f}%")
        
        # Distribution des clusters
        cluster_labels = results['cluster_labels']
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        print(f"\n🎯 DISTRIBUTION DES CLUSTERS:")
        total = len(cluster_labels)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"   - Cluster {cluster_id}: {count} docs ({percentage:.1f}%)")
        
        print(f"\n💾 Taille des modèles:")
        for model_file in model_files:
            if Path(model_file).exists():
                size_kb = Path(model_file).stat().st_size / 1024
                print(f"   - {model_file}: {size_kb:.1f} KB")
        
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Erreur évaluation: {e}")
        print("💡 Assurez-vous que les modèles sont entraînés")





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
  python main.py --mode build_dataset --num_articles 50 --lang fr
  
  # 3. Entraîner le pipeline auto-encodeur + clustering
  python main.py --mode train --encoding_dim 32 --n_clusters 5 --epochs 100
  
  # 4. Analyser et résumer un article
  python main.py --mode summarize --article "Chimie"
  
  # 5. Mode interactif
  python main.py --mode interactive
  
  # 6. Démonstration avec articles prédéfinis
  python main.py --mode demo
  
  # 7. Visualiser le preprocessing
  python main.py --mode visualize --article "Python (langage)"
  
  # 8. Évaluer les métriques du pipeline
  python main.py --mode evaluate
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
    
    # Nouveaux paramètres pour le mode autoencoder_clustering
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Nombre maximum de features TF-IDF (mode train)"
    )
    
    parser.add_argument(
        "--encoding_dim",
        type=int,
        default=32,
        help="Dimensions de la couche d'encodage de l'auto-encodeur (mode train)"
    )
    
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Nombre de clusters pour K-means (mode train)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'époques d'entraînement (mode train)"
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