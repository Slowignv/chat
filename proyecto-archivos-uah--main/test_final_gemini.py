from gemini_handler import GeminiHandler

def test_integration():
    print("🧪 Test Final de Integración Gemini 2.5 Flash")
    
    try:
        gemini = GeminiHandler()
        
        # Simular contexto de documentos archivísticos
        document_context = """
        - Documento 1: "Reglamento estudiantil de 1985 - Universidad Alberto Hurtado"
        - Documento 2: "Acta de fundación de la universidad - 1958" 
        - Documento 3: "Correspondencia sobre reformas académicas - 1992"
        - Documento 4: "Archivo fotográfico de eventos culturales - 1970-1980"
        """
        
        # Pregunta de prueba
        test_question = "¿Qué documentos tienen información sobre la historia fundacional de la universidad?"
        
        response = gemini.generate_response(test_question, document_context)
        
        print("✅ ✅ ✅ INTEGRACIÓN EXITOSA!")
        print(f"📋 Contexto: {document_context}")
        print(f"❓ Pregunta: {test_question}")
        print(f"🤖 Respuesta: {response}")
        
        # Probar resumen también
        sample_document = """
        Acta de fundación de la Universidad Alberto Hurtado, fechada el 15 de marzo de 1958.
        El documento establece los principios educativos basados en la formación humanista cristiana.
        Firma el rector fundador Dr. Carlos Alberto González y los miembros del primer consejo académico.
        Se especifican las primeras facultades: Humanidades, Derecho y Educación.
        """
        
        summary = gemini.summarize_document(sample_document)
        print(f"📄 Resumen de prueba: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return False

if __name__ == "__main__":
    test_integration()