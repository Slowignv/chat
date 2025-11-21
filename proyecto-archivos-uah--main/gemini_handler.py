import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiHandler:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en .env")
        
        genai.configure(api_key=self.api_key)
        
        # Usar el modelo que SÍ funciona
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        print("✅ Gemini 2.5 Flash configurado correctamente")
    
    def generate_response(self, prompt, context=""):
        """Genera respuesta usando Gemini con contexto de documentos archivísticos"""
        
        full_prompt = f"""
        Eres un asistente especializado en documentos archivísticos de la Universidad Alberto Hurtado.
        Tu función es ayudar a investigadores, estudiantes y personal a encontrar información en archivos históricos.

        CONTEXTO DE DOCUMENTOS DISPONIBLES:
        {context}

        PREGUNTA DEL USUARIO: {prompt}

        INSTRUCCIONES IMPORTANTES:
        - Responde de manera precisa y útil basándote ÚNICAMENTE en la información del contexto proporcionado
        - Si la información no está en el contexto, indica amablemente: "No tengo información específica sobre esto en los documentos disponibles"
        - Sé claro, conciso y profesional
        - Si el contexto menciona múltiples documentos relevantes, puedes referenciarlos
        - Mantén un tono amigable pero formal apropiado para un archivo universitario
        """
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error temporal del servicio: {str(e)}"
    
    def summarize_document(self, text, max_length=300):
        """Genera un resumen conciso de documentos archivísticos"""
        prompt = f"""
        Resume el siguiente documento archivístico en aproximadamente {max_length} palabras.
        Conserva la información más importante: fechas clave, personas, eventos, lugares y temas principales.
        
        DOCUMENTO:
        {text[:6000]}  # Limitar tamaño
        
        Proporciona un resumen bien estructurado que capture la esencia del documento.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al resumir: {str(e)}"