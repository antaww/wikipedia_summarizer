import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Poids pour l'entrée 
        self.W_xh = nn.Linear(input_size, hidden_size)

        # Poids pour l'état caché (récurence)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

        # Biais
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Couche de sortie 
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len): 
            x_t = x[:, t, :]  # (batch_size, input_size)
            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t) + self.b_h)  # Mise à jour de l'état caché Ht

        output = self.fc(h_t)  # (batch_size, output_size)
        return output


def preprocess_text(text):
    """Nettoie et préprocesse le texte"""
    if isinstance(text, str):
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""


def load_model_and_vectorizer():
    """Charge le modèle et recréé le vectorizer"""
    print("Chargement du modèle et préparation du vectorizer...")
    
    # Recharger les données pour recréer le vectorizer identique
    df = pd.read_csv('data/sentiment_data.csv')
    df = df.dropna()
    df['Comment'] = df['Comment'].apply(preprocess_text)
    df = df[df['Comment'].str.len() > 0]
    
    # Recréer le vectorizer avec les mêmes paramètres
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer.fit(df['Comment'])
    
    # Charger le modèle
    input_size = 1000
    hidden_size = 128
    output_size = 3
    
    model = SimpleRNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('rnn_sentiment_model.pth'))
    model.eval()
    
    return model, vectorizer


def predict_sentiment(model, vectorizer, text):
    """Prédit le sentiment d'un texte"""
    # Préprocesser le texte
    cleaned_text = preprocess_text(text)
    
    # Vectoriser
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    text_tensor = torch.FloatTensor(text_vector).unsqueeze(1)
    
    # Prédiction
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Mapping des labels
    sentiment_labels = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    
    return {
        'sentiment': sentiment_labels[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'Négatif': probabilities[0][0].item(),
            'Neutre': probabilities[0][1].item(),
            'Positif': probabilities[0][2].item()
        }
    }


def test_examples():
    """Teste le modèle sur quelques exemples"""
    print("=== CHARGEMENT DU MODÈLE ===")
    model, vectorizer = load_model_and_vectorizer()
    
    print("\n=== TESTS SUR DES EXEMPLES ===")
    
    # Exemples de test
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible, I hate it so much. Worst purchase ever.",
        "It's okay, nothing special but not bad either.",
        "Apple Pay is so convenient and easy to use!",
        "The payment failed again, this is so frustrating.",
        "The app works fine, no complaints.",
        "Absolutely fantastic! Best app ever created!",
        "Doesn't work properly, very disappointed.",
        "It's average, could be better but could be worse"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Texte: '{text}'")
        
        result = predict_sentiment(model, vectorizer, text)
        
        print(f"Prédiction: {result['sentiment']} (confiance: {result['confidence']:.3f})")
        print("Probabilités:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.3f}")


def interactive_test():
    """Mode interactif pour tester vos propres textes"""
    print("\n=== MODE INTERACTIF ===")
    print("Entrez vos textes pour tester le modèle (tapez 'quit' pour quitter)")
    
    model, vectorizer = load_model_and_vectorizer()
    
    while True:
        text = input("\nEntrez votre texte: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Au revoir!")
            break
            
        if not text:
            print("Veuillez entrer un texte valide.")
            continue
            
        result = predict_sentiment(model, vectorizer, text)
        
        print(f"\n📊 Résultat:")
        print(f"Sentiment: {result['sentiment']} (confiance: {result['confidence']:.3f})")
        print("Détail des probabilités:")
        for sentiment, prob in result['probabilities'].items():
            bar = "█" * int(prob * 20)  # Barre de progression visuelle
            print(f"  {sentiment:8}: {prob:.3f} {bar}")


if __name__ == "__main__":
    try:
        # Tests automatiques
        test_examples()
        
        # Mode interactif
        interactive_test()
        
    except FileNotFoundError:
        print("❌ Erreur: Le modèle 'rnn_sentiment_model.pth' n'a pas été trouvé.")
        print("Veuillez d'abord entraîner le modèle avec 'python test_rnn_torch.py'")
    except Exception as e:
        print(f"❌ Erreur: {e}") 