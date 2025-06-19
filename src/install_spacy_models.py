#!/usr/bin/env python3
"""
Script d'installation des modèles spaCy nécessaires pour la lemmatisation
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_spacy_model(model_name: str) -> bool:
    """
    Installe un modèle spaCy
    """
    try:
        logger.info(f"Installation du modèle spaCy: {model_name}")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", model_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Modèle {model_name} installé avec succès")
            return True
        else:
            logger.error(f"❌ Erreur installation {model_name}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Exception lors de l'installation de {model_name}: {e}")
        return False

def install_spacy_models():
    """
    Installe les modèles spaCy pour français et anglais
    """
    print("=== Installation des modèles spaCy ===\n")
    
    models = [
        ("fr_core_news_sm", "Français"),
        ("en_core_web_sm", "Anglais")
    ]
    
    success_count = 0
    
    for model_name, language in models:
        print(f"📦 Installation du modèle {language} ({model_name})...")
        if install_spacy_model(model_name):
            success_count += 1
        print()
    
    print(f"✅ Installation terminée: {success_count}/{len(models)} modèles installés")
    
    if success_count == len(models):
        print("🎉 Tous les modèles spaCy sont prêts!")
        print("Vous pouvez maintenant utiliser la lemmatisation dans le résumeur.")
    else:
        print("⚠️  Certains modèles n'ont pas pu être installés.")
        print("Le résumeur fonctionnera sans lemmatisation pour ces langues.")

if __name__ == "__main__":
    install_spacy_models() 