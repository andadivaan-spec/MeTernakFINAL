from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, torch, torch.nn as nn
from datetime import datetime, timezone
from supabase import create_client

app = Flask(__name__)
CORS(app)

try:
    from chatbot import chatbot_bp
    app.register_blueprint(chatbot_bp)
except:
    pass

# ─── SUPABASE ─────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── LSTM MODEL ───────────────────────────────────────────────────────────────
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


# ─── ESP32 MOCK: Klasifikasi resistansi & suhu berdasarkan tabel ──────────────
def mock_esp32_from_phase(phase):
    """
    Generate nilai resistansi & suhu mock berdasarkan fase estrus LSTM.
    Berdasarkan Tabel 3.8: Klasifikasi Fase Estrus Berdasarkan Resistansi dan Suhu
    """
    import random
    if phase == 'Day2' or phase == 'Day3':
        # Peak Estrus: resistansi < 140, suhu > 39
        resistance = random.randint(100, 139)
        temperature = round(random.uniform(39.1, 39.8), 1)
        status = 'Peak Estrus'
    elif phase == 'Day1':
        # Estrus: resistansi 140-220, suhu 38-39
        resistance = random.randint(140, 220)
        temperature = round(random.uniform(38.0, 39.0), 1)
        status = 'Estrus'
    elif phase == 'Kuning':
        # Non Estrus / lewat peak
        resistance = random.randint(601, 900)
        temperature = round(random.uniform(37.5, 38.2), 1)
        status = 'Non Estrus'
    else:
        # Menuju Estrus / default
        resistance = random.randint(220, 600)
        temperature = round(random.uniform(37.0, 38.0), 1)
        status = 'Menuju Estrus'
    return resistance, temperature, status


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def predict_lstm(seq, current_dt_hours=0):
    empty = {'predicted': 'Unknown', 'window_remaining': None,
             'p_day1': 0, 'p_day2': 0, 'p_day3': 0, 'p_kuning': 0,
             'peak_probability': 0}
    if lstm_model is None:
        return empty
    try:
        s = [[x[0] / 3., x[1] / 72., x[2]] for x in seq]
        s = [[0, 0, 0]] * (MAX_LEN - len(s)) + s
        x = torch.tensor([s[-MAX_LEN:]], dtype=torch.float32)
        with torch.no_grad():
            p = lstm_model(x).cpu().squeeze().numpy()

        idx = int(p.argmax())

        # Constraint: hari-1 (< 24 jam) tidak bisa peak
        if current_dt_hours < 24 and idx in [1, 2]:
            idx = 0

        # Window remaining dinamis berdasarkan dt_hours
        if current_dt_hours < 24:
            window = 2
        elif current_dt_hours < 48:
            window = 1
        elif current_dt_hours < 72:
            window = 0
        else:
            window = None  # window terlewat

        peak_prob = round(float(p[1]) + float(p[2]), 3)

        return {
            'predicted': NAMES[idx],
            'window_remaining': window,
            'p_day1': round(float(p[0]), 3),
            'p_day2': round(float(p[1]), 3),
            'p_day3': round(float(p[2]), 3),
            'p_kuning': round(float(p[3]), 3),
            'peak_probability': peak_prob
        }
    except Exception as e:
        print(f"LSTM error: {e}")
        return empty


def predict_rf(lstm_out, temperature, resistance, mucus_type, confidence):
    # Rule-based dari tabel klasifikasi jika RF tidak ada
    def rule_based(temperature, resistance, lstm_out):
        if lstm_out['predicted'] == 'Kuning' or lstm_out.get('window_remaining') is None:
            return 'JANGAN_IB'
        if resistance is not None and resistance < 140 and temperature and temperature > 39:
            return 'IB_SEKARANG'
        if resistance is not None and 140 <= resistance <= 220 and temperature and 38 <= temperature <= 39:
            if lstm_out['predicted'] in ['Day2', 'Day3']:
                return 'IB_SEKARANG'
        if lstm_out.get('window_remaining', 0) == 0:
            return 'JANGAN_IB'
        return 'STANDBY'

    if rf_model:
        try:
            features = [[mucus_type or 0, confidence or 0,
                         lstm_out['p_day1'], lstm_out['p_day2'],
                         lstm_out['p_day3'], lstm_out['p_kuning'],
                         temperature or 0, resistance or 0]]
            return rf_model.predict(features)[0]
        except Exception as e:
            print(f"RF error: {e}")

    return rule_based(temperature, resistance, lstm_out)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'lstm': lstm_model is not None, 'rf': rf_model is not None})


@app.route('/api/esp32', methods=['POST'])
def esp32():
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
    data = request.get_json(force=True)
    cattle_id   = data.get('cattle_id')
    farmer_name = data.get('farmer_name', '')
    mucus_color = data.get('mucus_color')
    confidence  = data.get('confidence', 1.0)
    temperature = data.get('temperature')
    dt_hours    = data.get('dt_hours', 0)

    if not cattle_id or not mucus_color:
        return jsonify({'error': 'cattle_id dan mucus_color wajib'}), 400

    mucus_type = MUCUS_TYPE_MAP.get(mucus_color, 0)

    # Ambil histori terakhir DENGAN dt_hours
    hist = (supabase.table('tracking_logs')
            .select('mucus_type, confidence, dt_hours')
            .eq('cattle_id', cattle_id)
            .order('created_at', desc=True)
            .limit(4)
            .execute())

    rows = list(reversed(hist.data)) if hist.data else []
    seq  = [[r['mucus_type'], r.get('dt_hours', 0), r['confidence']] for r in rows]
    seq.append([mucus_type, dt_hours, confidence])

    lstm_out = predict_lstm(seq, current_dt_hours=dt_hours)

    # ── ESP32 MOCK jika tidak ada data ESP32 fisik ────────────────────────────
    esp = (supabase.table('esp32_readings')
           .select('resistance_ohm')
           .eq('cattle_id', cattle_id)
           .order('created_at', desc=True)
           .limit(1)
           .execute())

    if esp.data:
        resistance = esp.data[0]['resistance_ohm']
        esp_mocked = False
    else:
        # Generate mock berdasarkan fase LSTM → tabel klasifikasi
        resistance, temperature_mock, esp_status = mock_esp32_from_phase(lstm_out['predicted'])
        if not temperature:
            temperature = temperature_mock
        esp_mocked = True

    decision = predict_rf(lstm_out, temperature, resistance, mucus_type, confidence)

    # Flag window terlewat
    if dt_hours >= 72:
        decision = 'WINDOW_TERLEWAT'
        lstm_out['predicted'] = 'Kuning'

    supabase.table('tracking_logs').insert({
        'cattle_id':    cattle_id,
        'farmer_name':  farmer_name,
        'mucus_type':   mucus_type,
        'mucus_color':  mucus_color,
        'confidence':   confidence,
        'temperature':  temperature,
        'resistance_ohm': resistance,
        'dt_hours':     dt_hours,
        'p_day1':       lstm_out['p_day1'],
        'p_day2':       lstm_out['p_day2'],
        'p_day3':       lstm_out['p_day3'],
        'p_kuning':     lstm_out['p_kuning'],
        'predicted':    decision,
        'created_at':   now_iso()
    }).execute()

    return jsonify({
        'cattle_id':        cattle_id,
        'mucus_color':      mucus_color,
        'confidence':       confidence,
        'temperature':      temperature,
        'resistance':       resistance,
        'esp_mocked':       esp_mocked,
        'lstm':             lstm_out,
        'decision':         decision
    })


@app.route('/api/cows', methods=['POST'])
def register_cow():
    data = request.get_json(force=True)
    cattle_id = data.get('cattle_id')
    if not cattle_id:
        return jsonify({'error': 'cattle_id wajib'}), 400
    supabase.table('cows').upsert({
        'cattle_id':    cattle_id,
        'farmer_name':  data.get('farmer_name', ''),
        'farm_address': data.get('farm_address', ''),
        'cattle_age':   data.get('cattle_age', '')
    }, on_conflict='cattle_id').execute()
    return jsonify({'status': 'ok'})


@app.route('/api/cows/history', methods=['GET'])
def cows_history():
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
            'cattle_id':   cow['cattle_id'],
            'cattle_age':  cow.get('cattle_age'),
            'last_record': last_row.get('created_at'),
            'last_ib':     last_row.get('predicted')
        })
    return jsonify(result)


@app.route('/api/tracking/<cattle_id>', methods=['GET'])
def tracking_detail(cattle_id):
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
