import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

class TestTrain(unittest.TestCase):

    def setUp(self):
        # Charger les données pour les tests
        self.data_path = 'data/customer_churn.csv'
        self.model_path = 'data/churn_model_clean.pkl'
        # Entraîner et sauvegarder le modèle si nécessaire
        if not os.path.exists(self.model_path):
            data = pd.read_csv(self.data_path)
            X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
            y = data['Churn']
            model = LogisticRegression()
            model.fit(X, y)
            joblib.dump(model, self.model_path)

    def test_data_loading(self):
        # Tester le chargement des données
        data = pd.read_csv(self.data_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        # Vérifier que les colonnes nécessaires existent
        required_columns = ['Age', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']
        for col in required_columns:
            self.assertIn(col, data.columns)

    def test_feature_selection(self):
        # Tester la sélection des features
        data = pd.read_csv(self.data_path)
        X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = data['Churn']
        self.assertEqual(X.shape[1], 4)  # 4 features
        self.assertEqual(len(y), len(X))

    def test_model_training(self):
        # Tester l'entraînement du modèle
        data = pd.read_csv(self.data_path)
        X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = data['Churn']
        model = LogisticRegression()
        model.fit(X, y)
        # Vérifier que le modèle est entraîné (a des coefficients)
        self.assertIsNotNone(model.coef_)
        self.assertEqual(model.coef_.shape[1], 4)

    def test_model_saving(self):
        # Tester la sauvegarde du modèle
        data = pd.read_csv(self.data_path)
        X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = data['Churn']
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        # Vérifier que le fichier est créé
        self.assertTrue(os.path.exists(self.model_path))
        # Vérifier qu'on peut le recharger
        loaded_model = joblib.load(self.model_path)
        self.assertIsInstance(loaded_model, LogisticRegression)

    def test_prediction(self):
        # Tester la prédiction avec le modèle chargé
        model = joblib.load(self.model_path)
        # Données d'exemple pour la prédiction (utiliser DataFrame avec noms de colonnes)
        input_data = pd.DataFrame([[35.0, 1, 5.0, 10]], 
                                  columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[0]
        # Vérifier que la prédiction est 0 ou 1
        self.assertIn(prediction[0], [0, 1])
        # Vérifier que les probabilités somment à 1
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=5)
        # Vérifier que chaque probabilité est entre 0 et 1
        for prob in probabilities:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()