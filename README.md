![Python](https://img.shields.io/badge/Python-3.10-blue)

# Analyse de sentiments avec BERT et TF-IDF

## Description

Projet de NLP comparant deux approches de classification de sentiments sur le dataset IMDb (50 000 avis) :
une méthode classique TF-IDF + Régression Logistique, et un fine-tuning de BERT via Hugging Face.

---

## Résultats

| Méthode                        | Accuracy | F1-score |
|-------------------------------|----------|----------|
| BERT (fine-tuning)            | 94 %     | 0.94     |
| TF-IDF + Logistic Regression  | 88 %     | 0.88     |

BERT surpasse TF-IDF sur les négations, les formulations ambiguës et les nuances contextuelles.

---

## Stack technique

| Catégorie        | Outils                                      |
|-----------------|---------------------------------------------|
| Modélisation    | PyTorch, Hugging Face Transformers, BERT    |
| Classique NLP   | Scikit-learn, TF-IDF, Logistic Regression  |
| Data            | Pandas, NumPy                               |
| Visualisation   | Matplotlib, Seaborn                         |
| Déploiement     | Streamlit                                   |
| Environnement   | Python, Jupyter Notebook                    |

---

## Pipeline

1. **Chargement** — dataset IMDb via Hugging Face (`datasets`)
2. **Prétraitement** — suppression des balises HTML, nettoyage, normalisation
3. **TF-IDF** — vectorisation + GridSearchCV sur Logistic Regression
4. **BERT** — tokenisation, dataset PyTorch custom, fine-tuning `bert-base-uncased`
5. **Évaluation** — accuracy, F1, matrices de confusion, analyse des erreurs
6. **Déploiement** — application web Streamlit, prédiction de sentiment en temps réel

---

## Déploiement

Application web développée avec **Streamlit** permettant :
- de saisir un avis utilisateur,
- d'obtenir une prédiction de sentiment en temps réel via le modèle BERT fine-tuné.

> Lancer l'application : `streamlit run app.py`

---

## Améliorations envisagées

- Dataset plus volumineux et classification multi-classes
- Modèles Transformers plus récents (RoBERTa, DistilBERT)
- Optimisation avancée des hyperparamètres

---

## Auteur

**Chazline Baghdadi**  
EFREI Paris · Ingé3 BDML1 · 2025/2026