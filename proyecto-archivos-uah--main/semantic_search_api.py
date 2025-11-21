from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import random
from pathlib import Path

# Import ML libraries with graceful fallback
HAS_TORCH = False
HAS_SENTTRANS = False
HAS_SKLEARN = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTTRANS = True
except Exception:
    SentenceTransformer = None
    util = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import numpy as np
    HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    linear_kernel = None
    np = None

# --- Inicializar Flask ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# --- Configuración ---
EMBEDDINGS_FILE = 'models/document_embeddings.pkl'
SEARCH_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
search_model = None
DOC_DATA = []
DOC_EMBEDDINGS = None
DEVICE = 'cpu'
if HAS_TORCH:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fallback vectorizer for when sentence-transformers / torch aren't available
VECTORIZER = None
DOC_MATRIX = None

CONVERSATIONAL_MODEL_FILE = 'models/conversational_model.pkl'
conversational_model = None
INTENTS_RESPONSES = {}

# --- 1. Cargar modelos ---
def load_all_models():
    global search_model, DOC_DATA, DOC_EMBEDDINGS, conversational_model, INTENTS_RESPONSES, DEVICE
    
    # Cargar Cerebro 1 (Recepcionista)
    print(f"Cargando 'Cerebro 1 (Recepcionista)' desde {CONVERSATIONAL_MODEL_FILE}...")
    try:
        with open(CONVERSATIONAL_MODEL_FILE, 'rb') as f:
            conversational_model = pickle.load(f)
        with open('intents_conversacionales.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
            for intent in intents_data['intents']:
                INTENTS_RESPONSES[intent['tag']] = intent['responses']
        print("✅ Cerebro 1 cargado.")
    except Exception as e:
        print(f"❌ Error cargando Cerebro 1: {e}")
        return False

    # Cargar Cerebro 2 (Archivista)
    print(f"Cargando 'Cerebro 2 (Archivista)' ({SEARCH_MODEL_NAME})...")
    if HAS_SENTTRANS:
        try:
            search_model = SentenceTransformer(SEARCH_MODEL_NAME)
            if HAS_TORCH and DEVICE == 'cuda':
                search_model.to(DEVICE)
            print(f"✅ Modelo de búsqueda cargado en {DEVICE}.")
        except Exception as e:
            print(f"❌ Error cargando modelo de búsqueda: {e}")
            search_model = None
    else:
        print("⚠️ sentence-transformers no disponible; usando fallback TF-IDF si está sklearn.")

    # Cargar embeddings (o generar desde JSON si no existen)
    print(f"Cargando 'Memoria del Archivista' desde {EMBEDDINGS_FILE}...")
    try:
        if Path(EMBEDDINGS_FILE).exists():
            with open(EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                DOC_DATA = data.get('data', [])
                DOC_EMBEDDINGS = data.get('embeddings')
                if DOC_EMBEDDINGS is not None:
                    DOC_EMBEDDINGS = DOC_EMBEDDINGS.to(DEVICE)
            print(f"✅ {len(DOC_DATA)} documentos listos desde embeddings guardados.")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    except Exception as e:
        print(f"⚠️ No se pudieron cargar embeddings precomputados: {e}")
        # Intentar cargar JSONs y generar embeddings en caliente
        try:
            from time import sleep
            def load_documents_from_json():
                candidates = [
                    'scraped_data/archivo_uah_data.json',
                    'scraped_data/archivo_uah_consolidado_completo.json',
                    'scraped_data/archivo_uah_completo_data.json',
                ]
                docs = []
                for c in candidates:
                    p = Path(c)
                    if not p.exists():
                        continue
                    try:
                        with open(p, 'r', encoding='utf-8') as fh:
                            data = json.load(fh)
                            # soportar formatos: {'data': [...] } o lista directa
                            if isinstance(data, dict) and 'data' in data:
                                items = data['data']
                            elif isinstance(data, list):
                                items = data
                            else:
                                # intentar detectar estructuras comunes
                                items = data.get('documents') if isinstance(data, dict) else []
                            for it in items:
                                # cada item: intentar extraer title, url, text/content
                                title = it.get('title') if isinstance(it, dict) else None
                                url = it.get('url') if isinstance(it, dict) else None
                                text = None
                                if isinstance(it, dict):
                                    text = it.get('text') or it.get('content') or it.get('body') or it.get('excerpt')
                                else:
                                    text = str(it)
                                docs.append({'title': title or 'Sin Título', 'url': url or '#', 'text': text or ''})
                    except Exception as ex:
                        print(f"Error leyendo {p}: {ex}")
                return docs

            DOC_DATA = load_documents_from_json()
            if not DOC_DATA:
                print("❌ No se encontraron documentos JSON en 'scraped_data/'.")
                return False

            print(f"🔁 Generando embeddings para {len(DOC_DATA)} documentos (esto puede tardar)...")
            texts = [d.get('text', '') for d in DOC_DATA]
            # dividir en batch si es necesario
            try:
                if HAS_SENTTRANS and search_model is not None:
                    embeddings = search_model.encode(texts, convert_to_tensor=True, device=DEVICE)
                    DOC_EMBEDDINGS = embeddings
                    print("✅ Embeddings generados en memoria con sentence-transformers.")
                elif HAS_SKLEARN:
                    # Usar TF-IDF como fallback (no requiere torch)
                    global VECTORIZER, DOC_MATRIX
                    VECTORIZER = TfidfVectorizer(max_features=20000)
                    DOC_MATRIX = VECTORIZER.fit_transform(texts)
                    DOC_EMBEDDINGS = DOC_MATRIX
                    print("✅ Matriz TF-IDF generada en memoria (fallback).")
                else:
                    print("❌ No hay método disponible para generar embeddings (instala torch o scikit-learn).")
                    return False
            except Exception as enc_err:
                print(f"❌ Error generando embeddings: {enc_err}")
                return False
        except Exception as e2:
            print(f"❌ Fallback JSON/embeddings falló: {e2}")
            return False

    return True

# --- 2. Funciones ---
def get_intent(query):
    if conversational_model is None:
        return "buscar"
    intent = conversational_model.predict([query])[0]
    confidence = conversational_model.decision_function([query]).max()
    if confidence < 0.3:
        return "buscar"
    return intent

def get_conversational_response(intent_tag):
    return random.choice(INTENTS_RESPONSES.get(intent_tag, ["No entendí eso."]))

def semantic_search(query, top_k=5):
    if DOC_EMBEDDINGS is None and DOC_MATRIX is None:
        return []

    resultados = []
    try:
        # Preferred path: sentence-transformers + torch/numpy
        if HAS_SENTTRANS and search_model is not None and DOC_EMBEDDINGS is not None:
            if HAS_TORCH:
                query_embedding = search_model.encode(query, convert_to_tensor=True, device=DEVICE)
                cos_scores = util.pytorch_cos_sim(query_embedding, DOC_EMBEDDINGS)[0]
                top_results = torch.topk(cos_scores, k=min(top_k, len(DOC_DATA)))
                for score, idx in zip(top_results[0], top_results[1]):
                    if score.item() > 0.4:
                        resultados.append({
                            'title': DOC_DATA[int(idx)].get('title', 'Sin Título'),
                            'url': DOC_DATA[int(idx)].get('url', '#'),
                            'relevance_score': float(score.item())
                        })
                return resultados
            else:
                # sentence-transformers available but torch not: use numpy arrays
                query_embedding = search_model.encode(query, convert_to_tensor=False)
                # DOC_EMBEDDINGS expected to be numpy array
                if hasattr(DOC_EMBEDDINGS, 'shape'):
                    from sklearn.metrics.pairwise import cosine_similarity
                    cos_scores = cosine_similarity([query_embedding], DOC_EMBEDDINGS)[0]
                    idxs = cos_scores.argsort()[::-1][:top_k]
                    for idx in idxs:
                        score = float(cos_scores[idx])
                        if score > 0.4:
                            resultados.append({
                                'title': DOC_DATA[int(idx)].get('title', 'Sin Título'),
                                'url': DOC_DATA[int(idx)].get('url', '#'),
                                'relevance_score': score
                            })
                    return resultados

        # Fallback path: TF-IDF matrix via scikit-learn
        if DOC_MATRIX is not None and VECTORIZER is not None:
            query_vec = VECTORIZER.transform([query])
            # cosine similarity via linear_kernel (faster for sparse)
            sims = linear_kernel(query_vec, DOC_MATRIX)[0]
            idxs = sims.argsort()[::-1][:top_k]
            for idx in idxs:
                score = float(sims[idx])
                if score > 0.05:  # lower threshold for TF-IDF
                    resultados.append({
                        'title': DOC_DATA[int(idx)].get('title', 'Sin Título'),
                        'url': DOC_DATA[int(idx)].get('url', '#'),
                        'relevance_score': score
                    })
            return resultados
    except Exception as e:
        print(f"Error en semantic_search fallback: {e}")

    return resultados

# --- 3. Endpoint principal ---
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chatbot_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    if request.method == 'POST':
        try:
            data = request.get_json()
            query = data.get('message', '').strip()
            if not query:
                return jsonify({'error': 'Mensaje vacío'}), 400

            print(f"--- Query recibida: '{query}' ---")
            intent = get_intent(query.lower())

            if intent == 'buscar':
                resultados = semantic_search(query)
                if resultados:
                    response_text = f"📄 Resultados para '{query}':<br><br>"
                    for i, result in enumerate(resultados):
                        response_text += f"{i+1}. {result['title']}<br>"
                        response_text += f"   🔗 <a href='{result['url']}' target='_blank'>Acceder al documento</a><br><br>"
                else:
                    response_text = f"Lo siento, no encontré resultados para '{query}'."
            else:
                response_text = get_conversational_response(intent)

            return jsonify({'response': response_text})

        except Exception as e:
            print(f"Error procesando la solicitud: {e}")
            return jsonify({'error': 'Error interno del servidor'}), 500

    return jsonify({'error': 'Método no permitido'}), 405

# --- 4. Main ---
if __name__ == '__main__':
    if load_all_models():
        print("🚀 Servidor Flask iniciado en http://0.0.0.0:5000")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("--- Error: No se pudo iniciar el servidor. ---")
