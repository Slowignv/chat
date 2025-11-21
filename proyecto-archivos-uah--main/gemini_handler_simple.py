import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiHandler:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("No hay API Key")
        
        genai.configure(api_key=self.api_key)
        self.model = self.find_working_model()
    
    def find_working_model(self):
        """Encuentra cualquier modelo que funcione"""
        try:
            # Primero lista todos los modelos
            models = genai.list_models()
            print("🔍 Buscando modelo funcional...")
            
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    print(f"🔄 Probando: {model.name}")
                    try:
                        test_model = genai.GenerativeModel(model.name)
                        response = test_model.generate_content("Hola")
                        print(f"✅ ✅ ✅ MODELO FUNCIONAL: {model.name}")
                        return test_model
                    except:
                        continue
            
            print("❌ Ningún modelo funcionó")
            return None
            
        except Exception as e:
            print(f"Error buscando modelos: {e}")
            return None
    
    def generate_response(self, prompt, context=""):
        if not self.model:
            return "Servicio de IA no disponible"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# Test inmediato
if __name__ == "__main__":
    print("🚀 Test rápido de Gemini")
    try:
        handler = GeminiHandler()
        if handler.model:
            response = handler.generate_response("Di 'Hola Mundo'")
            print(f"Respuesta: {response}")
        else:
            print("No hay modelos disponibles")
    except Exception as e:
        print(f"Error: {e}")