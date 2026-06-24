from flask import Blueprint, request, jsonify
from google import genai
import os

chatbot_bp = Blueprint('chatbot', __name__)

SYSTEM_PROMPT = """Anda adalah SaPI, asisten ahli reproduksi sapi betina.
ATURAN KETAT:
1. HANYA jawab topik: siklus estrus, tanda birahi, waktu inseminasi buatan (IB), kebuntingan, kelahiran, kesehatan reproduksi sapi betina.
2. Jawab LANGSUNG, tepat, dan ilmiah. Tanpa basa-basi pembuka.
3. Tolak topik di luar reproduksi sapi dengan satu kalimat singkat dalam Bahasa Indonesia.
4. JANGAN menyebut server, sistem, error, atau masalah teknis apapun dalam jawaban."""


@chatbot_bp.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    message = data.get('message', '')

    try:
        client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": SYSTEM_PROMPT},
            contents=message
        )
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'reply': 'Maaf, tidak dapat memproses pertanyaan Anda saat ini. Silakan coba lagi.'}), 200