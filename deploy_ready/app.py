from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, torch, torch.nn as nn
from datetime import datetime, timezone
from supabase import create_client
from chatbot import chatbot_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(chatbot_bp)

# ─── SUPABASE ─────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')  # service_role key
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── LOAD MODELS (YOLO sudah pindah ke browser, jadi cuma LSTM + RF) ─────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, 2, batch_first=True, dropout=.4)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(64, 4), nn.Softmax(dim=-1))

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])


lstm_model = LSTMModel()
try:
    lstm_model.load_state_dict(torch.load('LSTM.pth', map_location='cpu'))
    lstm_model.eval()
    print("LSTM loaded")
except Exception as e:
    lstm_model = None
    print(f"LSTM: {e}")

try:
    import joblib
    rf_model = joblib.load('rf_model.pkl')
    print("RF loaded")
except Exception as e:
    rf_model = None
    print(f"RF: {e}")

NAMES = ['Day1', 'Day2', 'Day3', 'Kuning']
MAX_LEN = 5
MUCUS_TYPE_MAP = {'transparant': 0, 'darah': 1, 'putih': 2, 'kuning': 3}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def predict_lstm(seq):
    empty = {'predicted': 'Unknown', 'window_remaining': None,
             'p_day1': 0, 'p_day2': 0, 'p_day3': 0, 'p_kuning': 0}
    if lstm_model is None:
        return empty
    try:
        s = [[x[0] / 3., x[1] / 72., x[2]] for x in seq]
        s = [[0, 0, 0]] * (MAX_LEN - len(s)) + s
        x = torch.tensor([s[-MAX_LEN:]], dtype=torch.float32)
        with torch.no_grad():
            p = lstm_model(x).cpu().squeeze().numpy()
        idx = int(p.argmax())
        return {'predicted': NAMES[idx],
                'window_remaining': max(0, 3 - idx) if idx < 3 else None,
                'p_day1': round(float(p[0]), 3), 'p_day2': round(float(p[1]), 3),
                'p_day3': round(float(p[2]), 3), 'p_kuning': round(float(p[3]), 3)}
    except Exception as e:
        print(f"LSTM error: {e}")
        return empty


def predict_rf(lstm_out, temperature, resistance, mucus_type, confidence):
    if rf_model:
        try:
            features = [[mucus_type or 0, confidence or 0,
                         lstm_out['p_day1'], lstm_out['p_day2'],
                         lstm_out['p_day3'], lstm_out['p_kuning'],
                         temperature or 0, resistance or 0]]
            return rf_model.predict(features)[0]
        except Exception as e:
            print(f"RF error: {e}")
    if lstm_out['predicted'] == 'Kuning':
        return 'JANGAN_IB'
    if lstm_out['predicted'] in ['Day2', 'Day3'] and temperature and 38.2 <= temperature <= 39.5:
        return 'IB_SEKARANG'
    return 'STANDBY'


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'lstm': lstm_model is not None, 'rf': rf_model is not None})


@app.route('/api/esp32', methods=['POST'])
def esp32():
    """ESP32 kirim resistansi mentah -> simpan ke Supabase."""
    data = request.get_json(force=True)
    cattle_id = data.get('cattle_id')
    resistance = data.get('resistance')
    if resistance is None:
        return jsonify({'error': 'resistance wajib'}), 400

    supabase.table('esp32_readings').insert({
        'cattle_id': cattle_id,
        'resistance_ohm': resistance,
        'created_at': now_iso()
    }).execute()
    return jsonify({'status': 'ok', 'resistance': resistance}), 201


@app.route('/api/esp32/latest', methods=['GET'])
def esp32_latest():
    """Frontend polling nilai resistansi terbaru per sapi."""
    cattle_id = request.args.get('cattle_id')
    q = supabase.table('esp32_readings').select('*').order('created_at', desc=True).limit(1)
    if cattle_id:
        q = q.eq('cattle_id', cattle_id)
    res = q.execute()
    if res.data:
        return jsonify(res.data[0])
    return jsonify({'resistance_ohm': None})


@app.route('/api/tracking', methods=['POST'])
def tracking():
    """
    Browser sudah menjalankan YOLO sendiri (onnxruntime-web) dan kirim hasilnya
    di sini sebagai JSON. Server tinggal jalankan LSTM + RF, lalu simpan ke Supabase.
    Body: { cattle_id, farmer_name, mucus_color, confidence, temperature, dt_hours }
    """
    data = request.get_json(force=True)
    cattle_id = data.get('cattle_id')
    farmer_name = data.get('farmer_name', '')
    mucus_color = data.get('mucus_color')
    confidence = data.get('confidence', 1.0)
    temperature = data.get('temperature')
    dt_hours = data.get('dt_hours', 0)

    if not cattle_id or not mucus_color:
        return jsonify({'error': 'cattle_id dan mucus_color wajib'}), 400

    mucus_type = MUCUS_TYPE_MAP.get(mucus_color, 0)

    # Ambil 4 riwayat terakhir sapi ini + resistansi ESP32 terbaru, semua dari Supabase
    hist = (supabase.table('tracking_logs')
            .select('mucus_type, confidence')
            .eq('cattle_id', cattle_id)
            .order('created_at', desc=True)
            .limit(4)
            .execute())
    esp = (supabase.table('esp32_readings')
           .select('resistance_ohm')
           .eq('cattle_id', cattle_id)
           .order('created_at', desc=True)
           .limit(1)
           .execute())

    resistance = esp.data[0]['resistance_ohm'] if esp.data else None
    rows = list(reversed(hist.data)) if hist.data else []
    seq = [[r['mucus_type'], dt_hours, r['confidence']] for r in rows]
    seq.append([mucus_type, dt_hours, confidence])

    lstm_out = predict_lstm(seq)
    decision = predict_rf(lstm_out, temperature, resistance, mucus_type, confidence)

    supabase.table('tracking_logs').insert({
        'cattle_id': cattle_id,
        'farmer_name': farmer_name,
        'mucus_type': mucus_type,
        'mucus_color': mucus_color,
        'confidence': confidence,
        'temperature': temperature,
        'resistance_ohm': resistance,
        'p_day1': lstm_out['p_day1'], 'p_day2': lstm_out['p_day2'],
        'p_day3': lstm_out['p_day3'], 'p_kuning': lstm_out['p_kuning'],
        'predicted': decision,
        'created_at': now_iso()
    }).execute()

    return jsonify({'cattle_id': cattle_id, 'mucus_color': mucus_color,
                     'confidence': confidence, 'temperature': temperature,
                     'resistance': resistance, 'lstm': lstm_out, 'decision': decision})


@app.route('/api/cows', methods=['POST'])
def register_cow():
    """Simpan/update identitas sapi & peternak (dipanggil dari form Pengaturan Data)."""
    data = request.get_json(force=True)
    cattle_id = data.get('cattle_id')
    if not cattle_id:
        return jsonify({'error': 'cattle_id wajib'}), 400

    supabase.table('cows').upsert({
        'cattle_id': cattle_id,
        'farmer_name': data.get('farmer_name', ''),
        'farm_address': data.get('farm_address', ''),
        'cattle_age': data.get('cattle_age', '')
    }, on_conflict='cattle_id').execute()
    return jsonify({'status': 'ok'})


@app.route('/api/cows/history', methods=['GET'])
def cows_history():
    """Daftar semua sapi + catatan/prediksi terakhirnya, untuk tab Histori."""
    cows = supabase.table('cows').select('*').execute().data or []
    result = []
    for cow in cows:
        last = (supabase.table('tracking_logs')
                .select('predicted, created_at')
                .eq('cattle_id', cow['cattle_id'])
                .order('created_at', desc=True)
                .limit(1)
                .execute())
        last_row = last.data[0] if last.data else {}
        result.append({
            'cattle_id': cow['cattle_id'],
            'cattle_age': cow.get('cattle_age'),
            'last_record': last_row.get('created_at'),
            'last_ib': last_row.get('predicted')
        })
    return jsonify(result)


@app.route('/api/tracking/<cattle_id>', methods=['GET'])
def tracking_detail(cattle_id):
    """Histori lengkap parameter untuk satu sapi (dipakai di detail Histori)."""
    rows = (supabase.table('tracking_logs')
            .select('*')
            .eq('cattle_id', cattle_id)
            .order('created_at', desc=True)
            .limit(50)
            .execute())
    return jsonify(rows.data or [])


@app.route('/')
def index():
    return send_from_directory('.', 'MeTernak (yolo).html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
