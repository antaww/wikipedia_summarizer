#!/usr/bin/env python3
"""
Module d'auto-encodeur et clustering pour le pipeline NLP
Implémente les étapes demandées :
1. Vectorisation TF-IDF
2. Auto-encodeur (entrée -> 32 neurones -> reconstruction)  
3. Entraînement X -> X
4. Extraction vecteurs compressés
5. K-means clustering
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Imports pour TF-IDF et clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Imports pour l'auto-encodeur
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class AutoencoderClustering:
    """Classe pour l'auto-encodeur et le clustering"""
    
    def __init__(self, config):
        self.config = config
        self.tfidf_vectorizer = None
        self.autoencoder = None
        self.encoder = None
        self.kmeans = None
        
        self.max_features = config.get('max_features', 5000)
        self.encoding_dim = config.get('encoding_dim', 32)
        self.n_clusters = config.get('n_clusters', 5)
        self.random_state = config.get('random_state', 42)
        
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"AutoencoderClustering initialisé avec {self.encoding_dim} dimensions")
    
    def vectorize_texts_tfidf(self, texts):
        """Étape 1: Vectorisation TF-IDF"""
        logger.info(f"🔤 Vectorisation TF-IDF de {len(texts)} textes")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()
        
        logger.info(f"✅ Vectorisation terminée: {tfidf_dense.shape}")
        
        # Sauvegarde
        with open(self.models_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        return tfidf_dense
    
    def create_autoencoder(self, input_dim):
        """Étape 2: Créer l'auto-encodeur"""
        logger.info(f"🏗️  Auto-encodeur: {input_dim} -> {self.encoding_dim} -> {input_dim}")
        
        # Architecture
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        
        # Modèles
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        # Compilation avec loss explicite
        autoencoder.compile(
            optimizer='adam', 
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )
        
        logger.info("✅ Auto-encodeur créé")
        return autoencoder, encoder
    
    def train_autoencoder(self, X, epochs=100):
        """Étape 3: Entraîner X -> X"""
        logger.info(f"🎯 Entraînement auto-encodeur")
        
        self.autoencoder, self.encoder = self.create_autoencoder(X.shape[1])
        
        # Entraînement X -> X
        history = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Sauvegarde avec les noms corrects pour TensorFlow
        autoencoder_weights_path = self.models_dir / 'autoencoder.weights.h5'
        encoder_weights_path = self.models_dir / 'encoder.weights.h5'
        
        self.autoencoder.save_weights(autoencoder_weights_path)
        self.encoder.save_weights(encoder_weights_path)
        
        # Sauvegarde de l'architecture
        autoencoder_config = self.autoencoder.get_config()
        encoder_config = self.encoder.get_config()
        
        with open(self.models_dir / 'autoencoder_config.pkl', 'wb') as f:
            pickle.dump(autoencoder_config, f)
        
        with open(self.models_dir / 'encoder_config.pkl', 'wb') as f:
            pickle.dump(encoder_config, f)
        
        logger.info("✅ Entraînement terminé")
        logger.info("💾 Modèles sauvegardés (poids + architecture)")
        
        return history.history
    
    def extract_encoded_vectors(self, X):
        """Étape 4: Extraction vecteurs compressés"""
        logger.info("📊 Extraction vecteurs compressés")
        
        X_encoded = self.encoder.predict(X, verbose=0)
        
        logger.info(f"✅ Compression: {X.shape[1]} -> {X_encoded.shape[1]}")
        return X_encoded
    
    def apply_kmeans_clustering(self, X_encoded):
        """Étape 5: K-means"""
        logger.info(f"🎯 K-means avec {self.n_clusters} clusters")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = self.kmeans.fit_predict(X_encoded)
        
        silhouette_avg = silhouette_score(X_encoded, cluster_labels)
        
        metrics = {
            'silhouette_score': silhouette_avg,
            'inertia': self.kmeans.inertia_,
            'n_clusters': self.n_clusters
        }
        
        logger.info(f"✅ Silhouette Score: {silhouette_avg:.3f}")
        
        # Sauvegarde
        with open(self.models_dir / 'kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        return cluster_labels, metrics
    
    def load_models(self):
        """Charger les modèles sauvegardés"""
        try:
            # Chargement TF-IDF
            with open(self.models_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Reconstruction des modèles à partir de l'architecture et des poids
            with open(self.models_dir / 'autoencoder_config.pkl', 'rb') as f:
                autoencoder_config = pickle.load(f)
            
            with open(self.models_dir / 'encoder_config.pkl', 'rb') as f:
                encoder_config = pickle.load(f)
            
            # Reconstruction des modèles
            self.autoencoder = keras.Model.from_config(autoencoder_config)
            self.encoder = keras.Model.from_config(encoder_config)
            
            # Chargement des poids avec les noms corrects
            autoencoder_weights_path = self.models_dir / 'autoencoder.weights.h5'
            encoder_weights_path = self.models_dir / 'encoder.weights.h5'
            
            self.autoencoder.load_weights(autoencoder_weights_path)
            self.encoder.load_weights(encoder_weights_path)
            
            # Chargement K-means
            with open(self.models_dir / 'kmeans_model.pkl', 'rb') as f:
                self.kmeans = pickle.load(f)
            
            logger.info("✅ Modèles chargés avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            return False
    
    def run_complete_pipeline(self, texts, titles=None):
        """Pipeline complet des 5 étapes"""
        logger.info(f"🚀 Pipeline complet - {len(texts)} textes")
        
        results = {}
        
        # Étape 1: TF-IDF
        logger.info("\nÉTAPE 1: TF-IDF")
        X_tfidf = self.vectorize_texts_tfidf(texts)
        results['tfidf_matrix'] = X_tfidf
        
        # Étapes 2-3: Auto-encodeur
        logger.info("\nÉTAPES 2-3: AUTO-ENCODEUR")
        training_history = self.train_autoencoder(X_tfidf)
        results['training_history'] = training_history
        
        # Étape 4: Extraction
        logger.info("\nÉTAPE 4: VECTEURS COMPRESSÉS")
        X_encoded = self.extract_encoded_vectors(X_tfidf)
        results['encoded_vectors'] = X_encoded
        
        # Étape 5: Clustering
        logger.info("\nÉTAPE 5: K-MEANS")
        cluster_labels, metrics = self.apply_kmeans_clustering(X_encoded)
        results['cluster_labels'] = cluster_labels
        results['clustering_metrics'] = metrics
        
        logger.info("🎉 Pipeline terminé!")
        return results


def run_autoencoder_clustering_pipeline(dataset_path="data/wikipedia_dataset_fr.csv"):
    """Point d'entrée principal"""
    config = {
        'max_features': 5000,
        'encoding_dim': 32,
        'n_clusters': 5,
        'random_state': 42,
        'models_dir': 'models'
    }
    
    # Chargement dataset
    df = pd.read_csv(dataset_path)
    texts = df['processed_content'].dropna().tolist()
    titles = df['title'].dropna().tolist()
    
    # Pipeline
    pipeline = AutoencoderClustering(config)
    results = pipeline.run_complete_pipeline(texts, titles)
    
    # Sauvegarde
    with open('models/pipeline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results 