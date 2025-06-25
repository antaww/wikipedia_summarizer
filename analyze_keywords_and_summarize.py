#!/usr/bin/env python3
"""
🔍 ANALYSE DES MOTS-CLÉS ET GÉNÉRATION DE RÉSUMÉ AUTOMATIQUE

Ce module analyse les termes TF-IDF les plus importants d'un article Wikipedia
et génère un résumé automatique basé sur les phrases contenant le plus de mots-clés.

Pipeline:
1. Récupération de l'article Wikipedia
2. Preprocessing avec le même pipeline que l'entraînement
3. Prédiction du cluster avec les modèles entraînés
4. Analyse TF-IDF des termes importants
5. Filtrage des références bibliographiques
6. Scoring des phrases par pertinence
7. Génération du résumé final

Usage:
    python analyze_keywords_and_summarize.py "Titre Article"
    
Ou depuis main.py:
    python main.py --mode analyze --article "Titre Article"
"""

import sys
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from collections import Counter
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout du path
sys.path.append('src')

def get_wikipedia_article_direct(title, language='fr'):
    """Récupère un article Wikipedia"""
    try:
        import wikipediaapi
        
        wiki = wikipediaapi.Wikipedia(
            user_agent='NLP-Pipeline-Analysis/1.0 (Educational use)',
            language=language
        )
        
        print(f"🔍 Recherche de l'article: '{title}'...")
        
        page = wiki.page(title)
        
        if not page.exists():
            print(f"❌ Article '{title}' non trouvé")
            
            # Suggestions d'articles populaires
            suggestions = [
                'Intelligence artificielle',
                'Python (langage)',
                'Machine learning',
                'France',
                'Informatique',
                'Chimie',
                'Physique',
                'Mathématiques'
            ]
            
            print(f"💡 Suggestions d'articles:")
            for suggestion in suggestions:
                print(f"   - {suggestion}")
            
            return None
        
        print(f"✅ Article trouvé: {page.title}")
        print(f"📄 Longueur: {len(page.text):,} caractères")
        print(f"🔗 URL: {page.fullurl}")
        
        return {
            'title': page.title,
            'text': page.text,
            'url': page.fullurl,
            'length': len(page.text)
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return None

def preprocess_article_direct(article_text, language='fr'):
    """Preprocess l'article avec le même pipeline que le dataset"""
    try:
        from src.text_preprocessor import TextPreprocessor
        from src.config import Config
        
        print("🧹 Preprocessing de l'article...")
        
        # Configuration identique au dataset
        config = Config()
        preprocessor = TextPreprocessor(config.PREPROCESSING_CONFIG)
        
        # Preprocessing complet
        clean_text, features = preprocessor.preprocessing_pipeline(article_text, language)
        
        print(f"✅ Preprocessing terminé")
        print(f"   📊 Texte original: {len(article_text):,} caractères")
        print(f"   📊 Texte nettoyé: {len(clean_text):,} caractères")
        print(f"   📊 Features extraites: {len(features)} éléments")
        
        return clean_text, features
        
    except Exception as e:
        print(f"❌ Erreur preprocessing: {e}")
        return None, None

def predict_with_pipeline_direct(clean_text):
    """Prédit le cluster avec le pipeline entraîné"""
    try:
        from src.autoencoder_clustering import AutoencoderClustering
        
        print("🤖 Chargement des modèles...")
        
        # Configuration (identique à l'entraînement)
        config = {
            'max_features': 5000,
            'encoding_dim': 32,
            'n_clusters': 5,
            'random_state': 42,
            'models_dir': 'models'
        }
        
        # Initialisation et chargement
        pipeline = AutoencoderClustering(config)
        
        if not pipeline.load_models():
            print("❌ Impossible de charger les modèles")
            print("💡 Assurez-vous d'avoir exécuté: python main.py --mode train2")
            return None
        
        print("✅ Modèles chargés avec succès")
        
        # Pipeline complet de prédiction
        print("🔤 Vectorisation TF-IDF...")
        tfidf_vector = pipeline.tfidf_vectorizer.transform([clean_text]).toarray()
        
        print("🧠 Encodage en 32 dimensions...")
        encoded_vector = pipeline.encoder.predict(tfidf_vector, verbose=0)
        
        print("🎯 Prédiction du cluster...")
        predicted_cluster = pipeline.kmeans.predict(encoded_vector)[0]
        
        # Distance aux centres des clusters
        cluster_distances = pipeline.kmeans.transform(encoded_vector)[0]
        
        # Probabilités relatives (plus la distance est petite, plus c'est probable)
        cluster_probs = 1 / (1 + cluster_distances)
        cluster_probs = cluster_probs / cluster_probs.sum()
        
        results = {
            'predicted_cluster': predicted_cluster,
            'tfidf_vector': tfidf_vector,
            'encoded_vector': encoded_vector,
            'cluster_distances': cluster_distances,
            'cluster_probabilities': cluster_probs
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        return None

def analyze_article_keywords(article_title="Chimie"):
    """Analyse les mots-clés les plus importants d'un article"""
    
    print("🔍 ANALYSE DES MOTS-CLÉS ET GÉNÉRATION DE RÉSUMÉ")
    print("="*70)
    
    try:
        print(f"📰 Analyse de l'article: {article_title}")
        
        # 1. Récupération et preprocessing de l'article
        article = get_wikipedia_article_direct(article_title, 'fr')
        if not article:
            return
        
        # 2. Preprocessing
        clean_text, features = preprocess_article_direct(article['text'], 'fr')
        if not clean_text:
            return
        
        # 3. Prédiction
        results = predict_with_pipeline_direct(clean_text)
        if not results:
            return
        
        predicted_cluster = results['predicted_cluster']
        print(f"🎯 Article classé dans le cluster: {predicted_cluster}")
        
        # 2. Analyse TF-IDF détaillée
        print(f"\n📊 ANALYSE TF-IDF:")
        analyze_tfidf_importance(clean_text, results)
        
        # 3. Analyse du cluster
        print(f"\n🧠 ANALYSE DU CLUSTER {predicted_cluster}:")
        analyze_cluster_characteristics(predicted_cluster)
        
        # 4. Extraction des phrases importantes
        print(f"\n📝 EXTRACTION DES PHRASES IMPORTANTES:")
        important_sentences = extract_important_sentences(article['text'], clean_text, results)
        
        # 5. Génération du résumé
        print(f"\n✨ GÉNÉRATION DU RÉSUMÉ:")
        generate_summary(article, important_sentences, results)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise

def analyze_tfidf_importance(clean_text, results):
    """Analyse l'importance des termes TF-IDF"""
    try:
        # Chargement du vectoriseur
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Vecteur TF-IDF de l'article
        tfidf_vector = results['tfidf_vector'][0]
        
        # Récupération du vocabulaire
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Tri des termes par importance TF-IDF
        term_scores = list(zip(feature_names, tfidf_vector))
        term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
        
        # Top 20 termes les plus importants
        print(f"🔤 TOP 20 TERMES TF-IDF:")
        for i, (term, score) in enumerate(term_scores[:20]):
            if score > 0:
                print(f"   {i+1:2d}. {term:<15} : {score:.4f}")
        
        # Statistiques
        non_zero_terms = sum(1 for _, score in term_scores if score > 0)
        print(f"\n📊 Statistiques TF-IDF:")
        print(f"   - Termes non-zéros: {non_zero_terms:,}")
        print(f"   - Score max: {tfidf_vector.max():.4f}")
        print(f"   - Score moyen: {tfidf_vector[tfidf_vector > 0].mean():.4f}")
        
        return term_scores[:50]  # Retourner top 50 pour l'analyse
        
    except Exception as e:
        print(f"❌ Erreur analyse TF-IDF: {e}")
        return []

def analyze_cluster_characteristics(cluster_id):
    """Analyse les caractéristiques du cluster"""
    try:
        # Chargement des résultats d'entraînement
        with open('models/pipeline_results.pkl', 'rb') as f:
            training_results = pickle.load(f)
        
        # Articles du même cluster dans le dataset d'entraînement
        cluster_labels = training_results['cluster_labels']
        cluster_mask = cluster_labels == cluster_id
        
        # Vecteurs encodés du cluster
        encoded_vectors = training_results['encoded_vectors']
        cluster_vectors = encoded_vectors[cluster_mask]
        
        print(f"📈 Caractéristiques du cluster {cluster_id}:")
        print(f"   - Nombre d'articles: {cluster_mask.sum()}")
        print(f"   - Vecteur moyen: [{cluster_vectors.mean(axis=0)[:5].round(3)}...]")
        print(f"   - Std moyen: {cluster_vectors.std(axis=0).mean():.3f}")
        
        # Centre du cluster K-means
        with open('models/kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        cluster_center = kmeans.cluster_centers_[cluster_id]
        print(f"   - Centre K-means: [{cluster_center[:5].round(3)}...]")
        
        return cluster_vectors.mean(axis=0)
        
    except Exception as e:
        print(f"❌ Erreur analyse cluster: {e}")
        return None

def extract_important_sentences(original_text, clean_text, results):
    """Extrait les phrases les plus importantes basées sur les mots-clés TF-IDF"""
    try:
        from nltk.tokenize import sent_tokenize
        
        # Chargement du vectoriseur
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Récupération des termes importants
        tfidf_vector = results['tfidf_vector'][0]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Top mots-clés (score > seuil)
        threshold = np.percentile(tfidf_vector[tfidf_vector > 0], 80)  # Top 20%
        important_terms = []
        
        for term, score in zip(feature_names, tfidf_vector):
            if score >= threshold:
                important_terms.append((term, score))
        
        important_terms = sorted(important_terms, key=lambda x: x[1], reverse=True)
        
        print(f"🎯 Mots-clés sélectionnés (score > {threshold:.4f}):")
        for term, score in important_terms[:10]:
            print(f"   - {term}: {score:.4f}")
        
        # Tokenisation en phrases
        sentences = sent_tokenize(original_text)
        
        # Patterns pour filtrer les références bibliographiques
        bibliographic_patterns = [
            r'\b(isbn|issn)\s*:?\s*\d',  # ISBN/ISSN
            r'\b(éd\.|édition|éditions)\b',  # Éditions
            r'\b(p\.|pp\.|page|pages)\s*\d+',  # Pages
            r'coll\.',  # Collection
            r'phonetoken',  # Tokens de téléphone
            r'et\s+al\.',  # Et alii
            r'^\s*[A-Z]\.\s*[A-Z]',  # Initiales d'auteur (début de phrase)
            r'\b\d{4}\b.*\b(paris|dunod|mcgraw|lavoisier|colin)\b',  # Année + éditeur
            r'^\s*[A-Z][a-z]+,\s*[A-Z]',  # Nom d'auteur au début
            r'^\s*\([^)]*\)',  # Parenthèses au début (souvent références)
            r'cours\s+de\s+[A-Z]',  # "cours de Paul"
        ]
        
        # Score de chaque phrase basé sur les mots-clés
        sentence_scores = []
        
        for sentence in sentences:
            # Filtrer les références bibliographiques
            is_bibliography = False
            for pattern in bibliographic_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_bibliography = True
                    break
            
            # Ignorer les références bibliographiques
            if is_bibliography:
                continue
            
            # Nettoyage basique de la phrase
            clean_sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
            clean_sentence = ' '.join(clean_sentence.split())
            
            # Score basé sur la présence des mots-clés
            score = 0
            word_count = 0
            
            for term, term_score in important_terms:
                if term in clean_sentence:
                    score += term_score
                    word_count += 1
            
            # Privilégier les phrases avec du contenu scientifique
            science_bonus = 0
            science_terms = ['science', 'étudie', 'matière', 'atomes', 'molécules', 'réactions', 'transformations', 'composés', 'éléments', 'propriétés']
            for science_term in science_terms:
                if science_term in clean_sentence:
                    science_bonus += 0.1
            
            # Normalisation par la longueur de la phrase + bonus scientifique
            if len(sentence.split()) > 8:  # Éviter les phrases trop courtes
                normalized_score = (score + science_bonus) / len(sentence.split()) * 100
                sentence_scores.append((sentence, score, normalized_score, word_count))
        
        # Tri par score normalisé
        sentence_scores = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
        
        print(f"\n📝 TOP 10 PHRASES IMPORTANTES:")
        for i, (sentence, score, norm_score, word_count) in enumerate(sentence_scores[:10]):
            print(f"\n{i+1:2d}. Score: {norm_score:.2f} ({word_count} mots-clés)")
            print(f"    {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
        
        return sentence_scores[:15]  # Retourner top 15 phrases
        
    except Exception as e:
        print(f"❌ Erreur extraction phrases: {e}")
        return []

def generate_summary(article, important_sentences, results):
    """Génère un résumé basé sur les phrases importantes"""
    try:
        predicted_cluster = results['predicted_cluster']
        
        print(f"📄 RÉSUMÉ AUTOMATIQUE - {article['title']}")
        print("="*60)
        
        # Sélection des meilleures phrases pour le résumé
        summary_sentences = []
        total_length = 0
        max_length = 800  # Longueur cible du résumé
        
        # Éviter les doublons et phrases trop similaires
        for sentence, score, norm_score, word_count in important_sentences:
            if total_length + len(sentence) <= max_length:
                # Vérifier qu'elle n'est pas trop similaire aux précédentes
                is_similar = False
                for existing in summary_sentences:
                    # Simple vérification de similarité
                    common_words = set(sentence.lower().split()) & set(existing.lower().split())
                    if len(common_words) > min(len(sentence.split()), len(existing.split())) * 0.5:
                        is_similar = True
                        break
                
                if not is_similar and len(sentence.split()) >= 6:
                    summary_sentences.append(sentence)
                    total_length += len(sentence)
                    
                    if len(summary_sentences) >= 5:  # Max 5 phrases
                        break
        
        # Affichage du résumé
        if summary_sentences:
            print("\n📋 Résumé généré:")
            for i, sentence in enumerate(summary_sentences):
                print(f"{i+1}. {sentence}")
            
            print(f"\n📊 Métadonnées du résumé:")
            print(f"   - Longueur: {total_length} caractères")
            print(f"   - Phrases: {len(summary_sentences)}")
            print(f"   - Article original: {len(article['text']):,} caractères")
            print(f"   - Taux de compression: {(total_length/len(article['text']))*100:.1f}%")
            print(f"   - Cluster assigné: {predicted_cluster}")
            
            # Sauvegarde du résumé
            summary_file = f"resume_{article['title'].replace(' ', '_')}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"RÉSUMÉ AUTOMATIQUE - {article['title']}\n")
                f.write(f"Cluster: {predicted_cluster}\n")
                f.write(f"Source: {article['url']}\n")
                f.write("="*60 + "\n\n")
                
                for i, sentence in enumerate(summary_sentences):
                    f.write(f"{i+1}. {sentence}\n\n")
            
            print(f"💾 Résumé sauvegardé: {summary_file}")
        else:
            print("❌ Impossible de générer un résumé")
        
    except Exception as e:
        print(f"❌ Erreur génération résumé: {e}")

def main():
    """Fonction principale"""
    
    # Récupération du titre d'article
    if len(sys.argv) > 1:
        article_title = " ".join(sys.argv[1:])
    else:
        article_title = "Chimie"  # Par défaut
    
    analyze_article_keywords(article_title)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analyse interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise 