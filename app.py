from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import load_model, predict_dengue
from utils import MedicalResponseGenerator
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Referencias médicas
REFERENCES = {
    "[1]": "OMS. (2023). Guía de diagnóstico y manejo clínico del dengue.",
    "[2]": "CDC. (2024). Dengue: Clinical Guidance.",
    "[3]": "Ministerio de Salud. Protocolo de atención para dengue (2024)."
}

# Cargar el modelo una sola vez al inicio
model, expected_features = load_model()
response_generator = MedicalResponseGenerator(REFERENCES)
ai_available = response_generator.use_ai

@app.route('/')
def home():
    return render_template('index.html', 
                         ai_available=ai_available,
                         gemini_key=response_generator.gemini_api_key[:5] + "..." if response_generator.gemini_api_key else "No configurada")

@app.route('/api/analyze', methods=['POST'])
def analyze():
    start_time = datetime.now()
    try:
        data = request.json
        
        # Validar datos de entrada
        if not data:
            return jsonify({'error': 'Datos no proporcionados'}), 400

        # Extraer y validar datos del formulario
        try:
            name = data.get('name', 'Paciente')
            age = int(data.get('age', 30))
            sex = data.get('sex', 'No especificado')
            platelets = int(data.get('platelets', 150000))
            hemoglobin = float(data.get('hemoglobin', 14.0))
            wbc = int(data.get('wbc', 5000))
            diff_count = float(data.get('diff_count', 60.0))
            rbc_panel = float(data.get('rbc_panel', 4.5))
            pdw = float(data.get('pdw', 12.0))
            symptoms = data.get('symptoms', 'No especificados')
        except (TypeError, ValueError) as e:
            return jsonify({'error': f'Datos inválidos: {str(e)}'}), 400

        # Validar rangos
        if not (0 <= age <= 120):
            return jsonify({'error': 'La edad debe estar entre 0 y 120 años'}), 400
        if not (10000 <= platelets <= 500000):
            return jsonify({'error': 'El recuento de plaquetas debe estar entre 10,000 y 500,000'}), 400
        if not (5 <= hemoglobin <= 20):
            return jsonify({'error': 'La hemoglobina debe estar entre 5 y 20 g/dL'}), 400
        
        # Hacer predicción
        prediction, probability = predict_dengue(
            model, 
            expected_features, 
            age, 
            platelets, 
            hemoglobin, 
            wbc, 
            diff_count, 
            rbc_panel, 
            pdw
        )
        
        # Generar explicación médica
        patient_data = {
            'name': name,
            'age': age,
            'sex': sex,
            'platelets': platelets,
            'hemoglobin': hemoglobin,
            'wbc': wbc,
            'diff_count': diff_count,
            'rbc_panel': rbc_panel,
            'pdw': pdw,
            'symptoms': symptoms
        }
        
        risk_level = "ALTA probabilidad de dengue" if prediction == 1 else "BAJA probabilidad de dengue"
        
        # Medir tiempo de generación de explicación
        explanation_start = datetime.now()
        explanation = response_generator.generate_medical_explanation(risk_level, patient_data)
        explanation_time = (datetime.now() - explanation_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'explanation': explanation,
            'risk_level': risk_level,
            'patient_data': patient_data,
            'metadata': {
                'ai_generated': ai_available,
                'explanation_time_seconds': explanation_time,
                'total_processing_time_seconds': total_time
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

@app.route('/api/status')
def status():
    return jsonify({
        'ai_available': ai_available,
        'gemini_key': response_generator.gemini_api_key[:5] + "..." if response_generator.gemini_api_key else "No configurada",
        'status': 'active',
        'last_checked': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)