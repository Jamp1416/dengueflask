import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import os

def load_model():
    try:
        dataset_path = os.path.join(os.path.dirname(__file__), 'Dengue diseases dataset.csv')
        if not os.path.exists(dataset_path):
            print(f"Advertencia: No se encuentra el dataset en {dataset_path}. Usando modelo simulado.")
            model = SimulatedModel()
            expected_features = ['Age', 'IsChild', 'Platelet Count', 'Haemoglobin', 'WBC Count', 'Differential Count', 'RBC PANEL', 'PDW']
            return model, expected_features

        df = pd.read_csv(dataset_path)
        df['IsChild'] = df['Age'].apply(lambda x: 1 if x < 18 else 0) if 'Age' in df.columns else 0
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'Male' else 0)
        if 'Final Output' not in df.columns:
            raise ValueError("Falta 'Final Output' en datos.")
        df.dropna(subset=['Final Output'], inplace=True)

        X = df.drop('Final Output', axis=1)
        y = df['Final Output']
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        scores = cross_val_score(model, X_imputed, y, cv=5)
        print(f"Precisión media: {scores.mean():.2f}, std: {scores.std():.2f}")
        print(classification_report(y_test, model.predict(X_test)))

        return model, X.columns.tolist()
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = SimulatedModel()
        expected_features = ['Age', 'IsChild', 'Platelet Count', 'Haemoglobin', 'WBC Count', 'Differential Count', 'RBC PANEL', 'PDW']
        return model, expected_features

def predict_dengue(model, expected_features, age, platelets, hemoglobin, wbc, diff_count, rbc_panel, pdw):
    df_in = pd.DataFrame([{col: 0 for col in expected_features}])
    df_in['Age'], df_in['IsChild'] = age, int(age < 18)
    df_in['Platelet Count'], df_in['Haemoglobin'] = platelets, hemoglobin
    df_in['WBC Count'], df_in['Differential Count'] = wbc, diff_count
    df_in['RBC PANEL'], df_in['PDW'] = rbc_panel, pdw

    prediction = model.predict(df_in)[0]
    probability = model.predict_proba(df_in)[0][1]

    return prediction, probability

class SimulatedModel:
    def predict(self, X):
        platelets = X['Platelet Count'].values[0]
        age = X['Age'].values[0]

        if platelets < 150000:
            return [1]
        elif platelets < 100000 or age > 65:
            return [1]
        else:
            return [0]

    def predict_proba(self, X):
        platelets = X['Platelet Count'].values[0]

        if platelets < 100000:
            prob = 0.9 + (100000 - platelets) / 100000 * 0.1
            prob = min(prob, 1.0)
        elif platelets < 150000:
            prob = 0.7 + (150000 - platelets) / 50000 * 0.2
        else:
            prob = max(0.1, 0.7 - (platelets - 150000) / 350000)

        return [[1 - prob, prob]]
