import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def diagnostic():
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"🔑 API Key: {api_key[:10]}...")
    
    if not api_key:
        print("❌ No hay API Key")
        return
    
    genai.configure(api_key=api_key)
    
    try:
        print("📋 Obteniendo lista de modelos...")
        models = genai.list_models()
        
        print("\n🎯 MODELOS GEMINI DISPONIBLES:")
        print("=" * 60)
        
        gemini_count = 0
        for model in models:
            if 'gemini' in model.name.lower():
                gemini_count += 1
                print(f"\n🔹 {model.name}")
                print(f"   Métodos: {model.supported_generation_methods}")
                print(f"   Versión: {getattr(model, 'version', 'N/A')}")
        
        print(f"\n📊 Total modelos Gemini: {gemini_count}")
        
        if gemini_count == 0:
            print("\n🚨 NO HAY MODELOS GEMINI DISPONIBLES")
            print("Posibles causas:")
            print("1. Tu API Key no está activada para Gemini")
            print("2. Problemas de región/ubicación")
            print("3. Necesitas activar Gemini en Google AI Studio")
            
    except Exception as e:
        print(f"❌ Error grave: {e}")

if __name__ == "__main__":
    diagnostic()