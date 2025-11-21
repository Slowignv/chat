from gemini_handler import GeminiHandler

def test_gemini():
    print("🧪 Probando conexión con Gemini 1.5 Flash...")
    
    try:
        gemini = GeminiHandler()
        
        # Test simple
        test_question = "Hola, ¿puedes ayudarme a encontrar documentos sobre la historia universitaria?"
        response = gemini.generate_response(test_question)
        
        print("✅ Conexión exitosa!")
        print(f"📝 Respuesta de prueba: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        return False

if __name__ == "__main__":
    test_gemini()