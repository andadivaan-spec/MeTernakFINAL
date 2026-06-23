from flask import Blueprint, request, jsonify
from google import genai
import os

chatbot_bp = Blueprint('chatbot', __name__)

SYSTEM_PROMPT = """Saya SaCo, asisten ternak terpercaya Anda.
HANYA jawab topik reproduksi & kesuburan sapi betina: siklus estrus, tanda birahi,
waktu inseminasi buatan (IB), kebuntingan, kelahiran, dan kesehatan reproduksi.
Tolak topik lain dengan sopan dalam Bahasa Indonesia sederhana."""


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
        return jsonify({'reply': 'Maaf, SaCo sedang tidak dapat dihubungi. Coba lagi sebentar.'}), 200
