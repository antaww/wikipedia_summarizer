# 🤖 Résumeur d'Articles Wikipedia - Pipeline NLP

Ce projet implémente un système de résumé automatique d'articles Wikipedia utilisant des techniques avancées de traitement du langage naturel (NLP). Il combine un preprocessing sophistiqué avec un modèle Transformer fine-tuné pour générer des résumés de qualité.

## 📋 Table des matières

- [🚀 Installation](#-installation)
- [🔧 Preprocessing en détail](#-preprocessing-en-détail)
- [📊 Construction du dataset d'entraînement](#-construction-du-dataset-dentraînement)
- [🎯 Entraînement du modèle](#-entraînement-du-modèle)
- [📝 Résumé d'articles avec le modèle](#-résumé-darticles-avec-le-modèle)
- [📁 Structure du projet](#-structure-du-projet)
- [🛠️ Utilisation avancée](#️-utilisation-avancée)

## 🚀 Installation

### Prérequis
- Python 3.8+
- GPU recommandé pour l'entraînement (optionnel)

### 1. Cloner le projet
```bash
git clone <url-du-repo>
cd NLP
```

### 2. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

### 3. Installer les modèles spaCy
```bash
python main.py --mode install_spacy
```

### 4. Télécharger les ressources NLTK
Les ressources NLTK se téléchargent automatiquement au premier lancement.

## 🔧 Preprocessing en détail

Le module de preprocessing (`src/text_preprocessor.py`) implémente un pipeline sophistiqué inspiré des techniques du TP Pipeline NLP.

### Visualisation du preprocessing

Pour voir le preprocessing en action étape par étape :

```bash
# Visualisation interactive du preprocessing
python main.py --mode visualize

# Ou pour un article spécifique
python main.py --mode visualize --article "Intelligence artificielle" --lang fr
```

### Étapes du preprocessing

#### 1. **Nettoyage du texte Wikipedia**
- ✅ Suppression des références `[1]`, `[2]`, etc.
- ✅ Suppression des balises de formatage `{{...}}`
- ✅ Suppression des URLs → remplacées par `URL_TOKEN`
- ✅ Suppression des numéros de téléphone → remplacés par `PHONE_TOKEN`
- ✅ Suppression des montants d'argent → remplacés par `MONEY_TOKEN`
- ✅ Normalisation de la ponctuation excessive

#### 2. **Tokenisation**
- Séparation en phrases avec `sent_tokenize()`
- Séparation en mots avec `word_tokenize()`

#### 3. **Suppression des mots vides intelligente**
- Supprime les stop words classiques
- **Préserve les mots de liaison importants** : `donc`, `mais`, `car`, `ainsi`, etc.
- Adaptée pour français et anglais

#### 4. **Lemmatisation**
- Utilise TextBlob pour la lemmatisation
- Réduit les mots à leur forme canonique
- Améliore la qualité du preprocessing

#### 5. **Extraction de features**
```python
features = {
    'caps_ratio': ratio_majuscules,
    'sentence_count': nombre_phrases,
    'avg_sentence_length': longueur_moyenne_phrases,
    'word_count': nombre_mots,
    'lexical_diversity': ratio_mots_uniques,
    'sentence_scores': scores_qualité_phrases
}
```

### Configuration du preprocessing

Le preprocessing est configurable via `src/config.py` :

```python
PREPROCESSING_CONFIG = {
    'remove_urls': True,
    'remove_phone_numbers': True,  
    'remove_money_mentions': True,
    'convert_to_lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True
}
```

## 📊 Construction du dataset d'entraînement

### Processus automatisé

Le dataset est construit automatiquement via le mode `build_dataset` :

```bash
# Construction d'un dataset de 100 articles français
python main.py --mode build_dataset --num_articles 100 --lang fr

# Pour l'anglais
python main.py --mode build_dataset --num_articles 50 --lang en
```

### Ce qui se passe lors de la construction

1. **Récupération d'articles Wikipedia aléatoires**
   - Utilise l'API Wikipedia
   - Filtre par longueur (min: 1000, max: 10000 caractères)
   - Gère les erreurs et articles manquants

2. **Préprocessing complet**
   - Application du pipeline de nettoyage
   - Extraction de toutes les features numériques
   - Scoring des phrases pour le résumé

3. **Génération des résumés de référence**
   - Utilise les résumés Wikipedia originaux
   - Nettoyage et normalisation
   - Limitation à 100-500 caractères

4. **Sauvegarde structurée**
   ```
   data/wikipedia_dataset_fr.csv
   ├── original_content      # Texte brut
   ├── processed_content     # Texte après preprocessing
   ├── wikipedia_summary     # Résumé Wikipedia original
   ├── target_summary        # Résumé cible nettoyé
   ├── sentence_count        # Nombre de phrases
   ├── lexical_diversity     # Diversité lexicale
   ├── compression_ratio     # Ratio de compression
   └── ... autres features
   ```

### Structure du dataset final

Chaque ligne contient :
- **Texte original ET texte preprocessé**
- **Toutes les features numériques extraites**
- **Métadonnées complètes** (URL, langue, longueurs)
- **Scores de qualité** de chaque phrase
- **Résumé de référence** nettoyé

## 🎯 Entraînement du modèle

### Lancement de l'entraînement

```bash
# Entraînement complet automatique
python main.py --mode train
```

### Processus d'entraînement

#### 1. **Chargement et préparation des données**
- Lecture du dataset CSV
- Nettoyage des données manquantes
- Division train/test (80%/20%)

#### 2. **Configuration du modèle**
- **Modèle de base** : `moussaKam/barthez` (BART optimisé pour le français)
- **Tokenizer** : SentencePiece avec tokens spéciaux
- **Architecture** : Seq2Seq Transformer

#### 3. **Paramètres d'entraînement**
```python
training_args = {
    'learning_rate': 3e-5,
    'batch_size': 4,
    'num_epochs': 3,
    'max_input_length': 512,
    'max_output_length': 128,
    'fp16': True,  # Si GPU disponible
    'eval_strategy': 'epoch'
}
```

#### 4. **Entraînement avec validation**
- Sauvegarde automatique des checkpoints
- Monitoring des métriques (loss, BLEU, ROUGE)
- Early stopping basé sur la validation

#### 5. **Sauvegarde du modèle**
```
models/summarization_wikipedia/
├── config.json
├── model.safetensors
├── tokenizer.json
├── training_args.bin
└── checkpoint-*/
```

### Monitoring de l'entraînement

- **Logs** : Affichage temps réel des métriques
- **Tensorboard** : Graphiques de progression
- **Checkpoints** : Sauvegarde toutes les epochs

## 📝 Résumé d'articles avec le modèle

### Modes de résumé disponibles

#### 1. **Mode ligne de commande**
```bash
# Résumé d'un article spécifique avec le modèle entraîné
python main.py --mode summarize --article "Intelligence artificielle" --lang fr

# Avec paramètres personnalisés
python main.py --mode summarize --article "Machine learning" --lang fr --max_length 150 --min_length 50
```

#### 2. **Mode interactif**
```bash
# Interface interactive complète
python main.py --mode interactive
```

#### 3. **Mode démonstration**
```bash
# Démonstration avec articles prédéfinis
python main.py --mode demo
```

### Fonctionnalités du résumé

- **Modèle Transformer entraîné** (pas de résumé extractif basique)
- **Paramètres configurables** : longueur min/max, beam search
- **Support multi-langue** : français et anglais
- **Métriques de qualité** : compression ratio, longueurs
- **Interface intuitive** avec affichage formaté

### Utilisation programmatique

```python
import sys
sys.path.append('src')

from src.wikipedia_summarizer_trained import WikipediaSummarizerTrained

# Initialisation
summarizer = WikipediaSummarizerTrained("./models/summarization_wikipedia")

# Résumé d'un article
result = summarizer.summarize_article("Machine learning", language='fr')

print(f"Titre: {result['title']}")
print(f"Résumé: {result['summary']}")
print(f"Compression: {result['compression_ratio']:.2%}")
```

### Paramètres de génération

Le modèle supporte différents paramètres :

```python
summary = summarizer.generate_summary(
    text,
    max_length=128,    # Longueur max du résumé
    min_length=30,     # Longueur min du résumé  
    num_beams=4,       # Beam search
    temperature=1.0,   # Créativité
    length_penalty=1.0 # Pénalité longueur
)
```

## 📁 Structure du projet

```
NLP/
├── 📄 main.py                    # Point d'entrée unifié pour tous les modes
├── 📄 requirements.txt           # Dépendances Python
├── 📄 build_steps.txt           # Documentation technique
│
├── 📁 src/                      # Modules principaux
│   ├── 📄 config.py             # Configuration globale
│   ├── 📄 dataset_builder.py    # Construction dataset
│   ├── 📄 text_preprocessor.py  # Pipeline preprocessing
│   ├── 📄 wikipedia_summarizer_trained.py # Résumeur avec modèle entraîné
│   ├── 📄 train_summarization_model.py    # Script d'entraînement
│   ├── 📄 visualize_preprocessing.py      # Visualisation preprocessing
│   └── 📄 install_spacy_models.py         # Installation modèles spaCy
│
├── 📁 data/                     # Datasets
│   └── 📄 wikipedia_dataset_fr.csv
│
├── 📁 models/                   # Modèles entraînés
│   └── 📁 summarization_wikipedia/
│       ├── 📄 config.json
│       ├── 📄 model.safetensors
│       └── 📁 checkpoint-*/
│
└── 📁 wandb/                    # Logs d'entraînement
```

## 🛠️ Utilisation avancée

### Pipeline complet d'utilisation

```bash
# 1. Installation des prérequis
python main.py --mode install_spacy

# 2. Construction du dataset
python main.py --mode build_dataset --num_articles 200 --lang fr

# 3. Entraînement du modèle
python main.py --mode train

# 4. Évaluation du modèle
python main.py --mode evaluate

# 5. Utilisation du modèle
python main.py --mode interactive
```

### Modes disponibles

| Mode | Description | Exemple |
|------|-------------|---------|
| `install_spacy` | Installation modèles spaCy | `python main.py --mode install_spacy` |
| `build_dataset` | Construction dataset | `python main.py --mode build_dataset --num_articles 100` |
| `train` | Entraînement modèle | `python main.py --mode train` |
| `summarize` | Résumé article | `python main.py --mode summarize --article "IA"` |
| `interactive` | Mode interactif | `python main.py --mode interactive` |
| `demo` | Démonstration | `python main.py --mode demo` |
| `visualize` | Visualisation preprocessing | `python main.py --mode visualize` |
| `evaluate` | Évaluation modèle | `python main.py --mode evaluate` |

### Personnalisation du preprocessing

Modifiez `src/config.py` pour adapter le preprocessing :

```python
PREPROCESSING_CONFIG = {
    'remove_urls': False,         # Garder les URLs
    'lemmatize': False,          # Désactiver lemmatisation
    'min_sentence_length': 10    # Filtrer phrases courtes
}
```

### Entraînement avec plus de données

```bash
# Dataset plus large pour de meilleures performances
python main.py --mode build_dataset --num_articles 1000 --lang fr
```

### Utilisation multi-langue

Le système supporte français et anglais :

```bash
# Dataset anglais
python main.py --mode build_dataset --lang en --num_articles 200

# Résumé en anglais
python main.py --mode summarize --article "Artificial intelligence" --lang en
```

---

## 🎯 Résultats attendus

- **Résumés cohérents** de 30-128 mots
- **Compression intelligente** (ratio 5-15%)
- **Préservation du sens** principal de l'article
- **Qualité linguistique** grâce au preprocessing avancé

## 🔧 Différences importantes

### ⚠️ **Attention : Deux types de résumé**

1. **Résumé pour dataset** (construction) : Utilise les résumés Wikipedia originaux nettoyés
2. **Résumé final** (utilisation) : Utilise le modèle Transformer entraîné

Le mode `summarize` utilise le **modèle entraîné**, pas les résumés Wikipedia !

## 🔧 Dépannage

### Problèmes courants

1. **Modèle non trouvé** : Vérifiez que l'entraînement s'est bien terminé avec `python main.py --mode train`
2. **Erreur CUDA** : Le système détecte automatiquement GPU/CPU
3. **Article non trouvé** : Vérifiez l'orthographe et la langue
4. **Dataset manquant** : Construisez d'abord le dataset avec `--mode build_dataset`

### Support

Pour plus d'informations, consultez les logs détaillés ou les fichiers de configuration dans `src/`. 