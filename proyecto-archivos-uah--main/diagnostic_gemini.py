import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def diagnostic():
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"API Key: {api_key[:10]}...")
    
    if not api_key:
        print("❌ No API key found")
        return
    
    genai.configure(api_key=api_key)
    
    try:
        # List available models
        models = genai.list_models()
        print("✅ Modelos disponibles:")
        for model in models:
            if 'gemini' in model.name.lower():
                print(f"  - {model.name}")
                print(f"    Métodos: {model.supported_generation_methods}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    diagnostic()