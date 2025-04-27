import os
import requests
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Definir mensajes médicos pregenerados como respaldo
PREGENERATED_MESSAGES = {
    "high_risk": (
        "El dengue es una enfermedad viral transmitida por mosquitos que puede ser grave si no se trata adecuadamente. "
        "Con los datos que ha proporcionado (plaquetas bajas y otros indicadores), nuestro sistema detecta una alta "
        "probabilidad de dengue.\n\n"
        "Es fundamental que acuda a un centro médico inmediatamente para recibir atención especializada. Mientras tanto:\n\n"
        "• Mantenga una hidratación constante con agua y sueros de rehidratación oral\n"
        "• Evite medicamentos como aspirina o ibuprofeno, ya que pueden aumentar el riesgo de sangrado\n"
        "• Esté atento a signos de alarma como dolor abdominal intenso, vómitos persistentes, sangrado en encías o nariz, "
        "y sensación de mareo o desmayo\n\n"
        "El dengue puede evolucionar rápidamente, por lo que el monitoreo médico es esencial. El tratamiento oportuno "
        "reduce significativamente el riesgo de complicaciones graves."
    ),
    "low_risk": (
        "Según los datos analizados, presenta un riesgo bajo de dengue. Sin embargo, es importante mantenerse vigilante "
        "y tomar medidas preventivas para evitar la infección por el virus del dengue.\n\n"
        "Recomendaciones preventivas importantes:\n\n"
        "• Use repelente de insectos regularmente, especialmente en horas de mayor actividad del mosquito (amanecer y atardecer)\n"
        "• Elimine posibles criaderos de mosquitos en su entorno: recipientes con agua estancada, floreros, llantas, etc.\n"
        "• Utilice mosquiteros en ventanas y al dormir\n"
        "• Vista ropa que cubra la mayor parte de su piel\n\n"
        "Si desarrolla síntomas como fiebre alta repentina, dolor de cabeza intenso, dolor detrás de los ojos, dolores "
        "musculares y articulares, náuseas, vómitos o erupciones en la piel, consulte inmediatamente a un médico."
    )
}

class MedicalResponseGenerator:
    def __init__(self, references):
        self.references = references
        # Configuración directa de la API Key de Gemini
        self.gemini_api_key = "AIzaSyCwgpLuzd-JL-qbCicV8aaGqAgTfDFEUP4"
        self.use_ai = True  # Siempre activo ya que tenemos la clave

    def generate_medical_explanation(self, recommendation, patient_data=None):
        """Genera una explicación médica usando Gemini AI"""
        is_high_risk = "ALTA" in recommendation or "alta" in recommendation.lower()

        try:
            platelets = patient_data.get('platelets', 'N/A') if patient_data else 'N/A'
            age = patient_data.get('age', 'N/A') if patient_data else 'N/A'
            hemoglobin = patient_data.get('hemoglobin', 'N/A') if patient_data else 'N/A'
            symptoms = patient_data.get('symptoms', 'N/A') if patient_data else 'N/A'
            risk_level = "alto" if is_high_risk else "bajo"

            prompt = f"""
            Como especialista médico en enfermedades transmitidas por vectores, proporciona una explicación detallada
            y recomendaciones para un paciente con los siguientes datos:

            - Edad: {age} años
            - Nivel de plaquetas: {platelets}
            - Nivel de hemoglobina: {hemoglobin}
            - Síntomas reportados: {symptoms}
            - Nivel de riesgo evaluado: {risk_level}

            El paciente ha sido evaluado con un riesgo {risk_level} de dengue.

            Proporciona:
            1. Una explicación clara de lo que indican estos valores
            2. Recomendaciones médicas específicas basadas en estos valores
            3. Signos de alarma que el paciente debe vigilar
            4. Medidas preventivas adecuadas
            5. Pasos a seguir según el nivel de riesgo

            Usa un lenguaje claro y comprensible para el paciente.
            Incluye referencias médicas confiables.
            Sé preciso pero empático en tus recomendaciones.

            Formato de respuesta:
            [Explicación médica]
            [Recomendaciones específicas]
            [Signos de alarma]
            [Medidas preventivas]
            [Pasos a seguir]
            [Referencias]
            """

            response = self.call_gemini_api(prompt)

            if response:
                # Formatear mejor la respuesta
                formatted_response = response.replace("[Explicación médica]", "<strong>Explicación médica:</strong>")
                formatted_response = formatted_response.replace("[Recomendaciones específicas]", "<br><strong>Recomendaciones específicas:</strong>")
                formatted_response = formatted_response.replace("[Signos de alarma]", "<br><strong>Signos de alarma:</strong>")
                formatted_response = formatted_response.replace("[Medidas preventivas]", "<br><strong>Medidas preventivas:</strong>")
                formatted_response = formatted_response.replace("[Pasos a seguir]", "<br><strong>Pasos a seguir:</strong>")
                formatted_response = formatted_response.replace("[Referencias]", "<br><strong>Referencias:</strong>")
                
                full_response = (
                    formatted_response +
                    "\n\n" +
                    "\n".join([f"{k} {v}" for k, v in self.references.items()])
                )
                return full_response
        except Exception as e:
            print(f"Error al generar respuesta con Gemini AI: {e}")

        # Fallback a mensajes pregenerados si hay error con Gemini
        explanation = PREGENERATED_MESSAGES["high_risk"] if is_high_risk else PREGENERATED_MESSAGES["low_risk"]
        
        if patient_data:
            age = patient_data.get('age', '')
            platelets = patient_data.get('platelets', '')
            hemoglobin = patient_data.get('hemoglobin', '')

            if is_high_risk:
                if platelets < 100000:
                    explanation += "\n\nSus niveles de plaquetas están por debajo del rango normal (150,000-450,000/μL), " \
                                  "lo que es un indicador importante de la posible gravedad del dengue."
                if age < 15 or age > 65:
                    explanation += "\n\nDebe prestar especial atención ya que su edad representa un factor de riesgo " \
                                  "adicional para complicaciones relacionadas con el dengue."

        return explanation + "\n\nReferencias:\n" + "\n".join([f"{k} {v}" for k, v in self.references.items()])

    def call_gemini_api(self, prompt):
        """Llama a la API de Google Gemini para generar una respuesta médica"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.95,
                "maxOutputTokens": 1000
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_MEDICAL",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30  # 30 segundos de timeout
            )

            response.raise_for_status()
            response_data = response.json()

            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Respuesta inesperada de la API: {response_data}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error al llamar a la API de Gemini: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None