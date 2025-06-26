import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re


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
    

# Insérer ci-dessous votre chaine de pré-traitement de la donnée (y compris l'import)

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

def load_and_preprocess_data():
    """Charge et préprocesse les données de sentiment"""
    # Charger les données
    df = pd.read_csv('data/sentiment_data.csv')
    
    # Nettoyer les données
    df = df.dropna()
    df['Comment'] = df['Comment'].apply(preprocess_text)
    
    # Filtrer les commentaires vides
    df = df[df['Comment'].str.len() > 0]
    
    return df

# Charger et préparer les données
df = load_and_preprocess_data()
print(f"Nombre d'échantillons chargés: {len(df)}")
print(f"Distribution des sentiments:\n{df['Sentiment'].value_counts()}")

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['Comment']).toarray()
y = df['Sentiment'].values

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Conversion en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Ajouter dimension séquence
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

print(f"Forme X_train: {X_train_tensor.shape}")
print(f"Forme y_train: {y_train_tensor.shape}")

input_size = X_train.shape[1]  # Nombre de features TF-IDF
output_size = len(np.unique(y))  # Nombre de classes de sentiment
hidden_size = 128
num_epochs = 100

print(f"Input size: {input_size}")
print(f"Output size: {output_size}")
print(f"Hidden size: {hidden_size}")

model = SimpleRNN(input_size, hidden_size, output_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entraînement
print("\nDébut de l'entraînement...")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # Évaluation sur l'ensemble de test
        model.eval()
        with torch.no_grad():
            test_output = model(X_test_tensor)
            test_predictions = torch.argmax(test_output, dim=1)
            test_accuracy = accuracy_score(y_test, test_predictions.numpy())
        
        print(f"Epoch {epoch}/{num_epochs}, Perte = {loss.item():.4f}, Précision test = {test_accuracy:.4f}")

# Évaluation finale
model.eval()
with torch.no_grad():
    # Prédictions sur l'ensemble de test
    test_output = model(X_test_tensor)
    test_predictions = torch.argmax(test_output, dim=1).numpy()
    
    # Métriques finales
    final_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Précision finale: {final_accuracy:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, test_predictions, target_names=['Négatif', 'Neutre', 'Positif']))

# Sauvegarder le modèle
torch.save(model.state_dict(), 'rnn_sentiment_model.pth')
print(f"\nModèle sauvegardé sous 'rnn_sentiment_model.pth'")