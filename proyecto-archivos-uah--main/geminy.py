from gemini_handler import GeminiHandler

def test_gemini():
    try:
        gemini = GeminiHandler()
        response = gemini.generate_response("Hola, ¿puedes ayudarme con documentos archivísticos?")
        print("✅ Respuesta de Gemini:", response)
        return True
    except Exception as e:
        print("❌ Error:", e)
        return False

if __name__ == "__main__":
    test_gemini()