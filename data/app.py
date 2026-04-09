from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle entraîné
model = joblib.load('data/churn_model_clean.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    age = None
    account_manager = None
    years = None
    num_sites = None
    prob_0 = None
    prob_1 = None
    if request.method == 'POST':
        # Récupérer les données du formulaire
        age = float(request.form['age'])
        account_manager = int(request.form['account_manager'])
        years = float(request.form['years'])
        num_sites = int(request.form['num_sites'])
        
        # Transformer en DataFrame avec noms de colonnes
        input_data = pd.DataFrame([[age, account_manager, years, num_sites]], 
                                  columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        
        # Faire la prédiction
        pred = model.predict(input_data)
        probas = model.predict_proba(input_data)[0]
        prob_0 = f"{probas[0] * 100:.2f}"  # Probabilité de rester
        prob_1 = f"{probas[1] * 100:.2f}"  # Probabilité de churn
        
        prediction = str(int(pred[0]))
        probability = prob_1  # Garder pour compatibilité, mais on utilisera prob_0 et prob_1
    
    return render_template('index.html', prediction=prediction, probability=probability, age=age, account_manager=account_manager, years=years, num_sites=num_sites, prob_0=prob_0, prob_1=prob_1)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')