#!/usr/bin/env python3
"""
Script d'entraînement d'un modèle de résumé automatique
Utilise le dataset Wikipedia français pour entraîner un modèle Transformer
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaSummarizationDataset(Dataset):
    """
    Dataset personnalisé pour la tâche de résumé Wikipedia
    """
    
    def __init__(self, texts, summaries, tokenizer, max_input_length=512, max_target_length=128):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])
        
        # Tokenisation du texte d'entrée
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenisation du résumé cible
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def load_and_prepare_data(dataset_path, test_size=0.2, random_state=42):
    """
    Charge et prépare les données pour l'entraînement
    """
    logger.info(f"📊 Chargement du dataset: {dataset_path}")
    
    # Chargement du dataset
    df = pd.read_csv(dataset_path)
    logger.info(f"✅ Dataset chargé: {len(df)} échantillons")
    
    # Nettoyage des données manquantes
    df = df.dropna(subset=['original_content', 'wikipedia_summary'])
    logger.info(f"🧹 Après nettoyage: {len(df)} échantillons")
    
    # Extraction des colonnes nécessaires
    texts = df['original_content'].tolist()
    summaries = df['wikipedia_summary'].tolist()
    
    # Division train/test
    train_texts, test_texts, train_summaries, test_summaries = train_test_split(
        texts, summaries, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"📈 Division des données:")
    logger.info(f"  - Entraînement: {len(train_texts)} échantillons")
    logger.info(f"  - Test: {len(test_texts)} échantillons")
    
    return train_texts, test_texts, train_summaries, test_summaries

def setup_model_and_tokenizer(model_name="moussaKam/barthez"):
    """
    Configure le modèle et le tokenizer
    """
    logger.info(f"🤖 Chargement du modèle: {model_name}")
    
    # Chargement du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Si pas de token de fin de séquence, en ajouter un
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Chargement du modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Redimensionner les embeddings si nécessaire
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"✅ Modèle et tokenizer configurés")
    return model, tokenizer

def train_model(train_dataset, test_dataset, model, tokenizer, output_dir="./models/summarization"):
    """
    Entraîne le modèle de résumé
    """
    logger.info("🚀 Début de l'entraînement...")
    
    # Configuration des arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Changé de evaluation_strategy à eval_strategy
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Utilise FP16 si GPU disponible
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None  # Désactive wandb/tensorboard
    )
    
    # Collator pour les données
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Configuration du trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Entraînement
    logger.info("⏳ Entraînement en cours...")
    trainer.train()
    
    # Sauvegarde du modèle final
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✅ Modèle sauvegardé dans: {output_dir}")
    
    return trainer

def evaluate_model(trainer, test_dataset, tokenizer):
    """
    Évalue le modèle sur le dataset de test
    """
    logger.info("📊 Évaluation du modèle...")
    
    # Évaluation
    eval_results = trainer.evaluate()
    
    logger.info("📈 Résultats d'évaluation:")
    for key, value in eval_results.items():
        logger.info(f"  - {key}: {value:.4f}")
    
    return eval_results

def test_model_inference(model, tokenizer, test_text, max_length=128):
    """
    Teste l'inférence du modèle sur un exemple
    """
    logger.info("🧪 Test d'inférence...")
    
    # Tokenisation
    inputs = tokenizer(
        test_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Génération
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Décodage
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    logger.info("📝 Exemple de résumé généré:")
    logger.info(f"Texte original (extrait): {test_text[:200]}...")
    logger.info(f"Résumé généré: {summary}")
    
    return summary

def train_summarization_model():
    """
    Fonction principale d'entraînement
    """
    print("=== Entraînement Modèle de Résumé Wikipedia ===\n")
    
    # Configuration
    dataset_path = "data/wikipedia_dataset_fr.csv"
    output_dir = "./models/summarization_wikipedia"
    model_name = "moussaKam/barthez"  # Modèle BART français pour résumé
    
    # Vérification de l'existence du dataset
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        return
    
    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # 1. Chargement et préparation des données
        train_texts, test_texts, train_summaries, test_summaries = load_and_prepare_data(dataset_path)
        
        # 2. Configuration du modèle et tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_name)
        
        # 3. Création des datasets
        logger.info("📦 Création des datasets...")
        train_dataset = WikipediaSummarizationDataset(
            train_texts, train_summaries, tokenizer
        )
        test_dataset = WikipediaSummarizationDataset(
            test_texts, test_summaries, tokenizer
        )
        
        # 4. Entraînement
        trainer = train_model(train_dataset, test_dataset, model, tokenizer, output_dir)
        
        # 5. Évaluation
        eval_results = evaluate_model(trainer, test_dataset, tokenizer)
        
        # 6. Test d'inférence
        if test_texts:
            test_model_inference(model, tokenizer, test_texts[0])
        
        # 7. Sauvegarde des résultats
        results_file = f"{output_dir}/training_results.txt"
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Résultats d'entraînement - {datetime.now()}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Modèle utilisé: {model_name}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Échantillons d'entraînement: {len(train_texts)}\n")
            f.write(f"Échantillons de test: {len(test_texts)}\n\n")
            f.write("Résultats d'évaluation:\n")
            for key, value in eval_results.items():
                f.write(f"  - {key}: {value:.4f}\n")
        
        logger.info(f"✅ Entraînement terminé! Résultats sauvés dans: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Erreur durant l'entraînement: {e}")
        raise

if __name__ == "__main__":
    try:
        train_summarization_model()
    except KeyboardInterrupt:
        logger.info("\n\nEntraînement interrompu par l'utilisateur. Au revoir! 👋")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}") 