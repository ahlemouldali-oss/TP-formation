import unittest
import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Ajouter le répertoire parent au path pour importer data.app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.app import app  # Importer l'app Flask

class TestApp(unittest.TestCase):

    def setUp(self):
        # Assurer que le modèle existe pour les tests
        model_path = 'data/churn_model_clean.pkl'
        if not os.path.exists(model_path):
            # Entraîner le modèle si nécessaire
            data = pd.read_csv('data/customer_churn.csv')
            X = data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
            y = data['Churn']
            model = LogisticRegression()
            model.fit(X, y)
            joblib.dump(model, model_path)

        self.app = app.test_client()
        self.app.testing = True

    def test_app_creation(self):
        # Tester que l'app Flask est créée
        self.assertIsNotNone(app)
        self.assertEqual(app.name, 'data.app')

    def test_get_request(self):
        # Tester la requête GET sur la route principale
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        # Vérifier que le formulaire est présent
        self.assertIn(b'form method="POST"', response.data)
        self.assertIn(b'name="age"', response.data)
        self.assertIn(b'name="account_manager"', response.data)

    def test_post_request_valid_data(self):
        # Tester la requête POST avec des données valides
        data = {
            'age': '35',
            'account_manager': '1',
            'years': '5.0',
            'num_sites': '10'
        }
        response = self.app.post('/', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Vérifier que le résultat de prédiction est affiché
        self.assertIn(b'R\xc3\xa9sultat de la Pr\xc3\xa9diction', response.data)
        self.assertIn(b'Probabilit\xc3\xa9 de churn', response.data)
        self.assertIn(b'Probabilit\xc3\xa9 de rester', response.data)

    def test_post_request_prediction_values(self):
        # Tester que les valeurs de prédiction sont affichées
        data = {
            'age': '35',
            'account_manager': '1',
            'years': '5.0',
            'num_sites': '10'
        }
        response = self.app.post('/', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Vérifier que les probabilités sont présentes (format XX.XX%)
        response_text = response.data.decode('utf-8')
        # Chercher un pattern comme "19.02%" pour la probabilité de churn
        self.assertRegex(response_text, r'\d+\.\d+%')
        # Vérifier que les informations saisies sont affichées
        self.assertIn('35.0', response_text)
        self.assertIn('Oui', response_text)  # Account Manager 1
        self.assertIn('5.0', response_text)
        self.assertIn('10', response_text)

    def test_model_loading(self):
        # Tester que le modèle est chargé dans l'app
        from data.app import model
        self.assertIsNotNone(model)
        # Vérifier que c'est un modèle LogisticRegression
        self.assertIsInstance(model, LogisticRegression)

if __name__ == '__main__':
    unittest.main()