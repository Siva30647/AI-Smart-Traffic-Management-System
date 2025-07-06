from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import threading
import logging

app = Flask(__name__)
CSV_FILE = 'sensor_data.csv'
LOCK = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_csv():
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=['timestamp', 'ir1', 'ir2', 'ultrasonic'])
        df.to_csv(CSV_FILE, index=False)
        logging.info(f"Created new CSV file: {CSV_FILE}")

initialize_csv()

@app.route('/data', methods=['POST'])
def receive_data():
    client_ip = request.remote_addr

    try:
        data = request.get_json(force=True)
        required_keys = ['ir1', 'ir2', 'ultrasonic']
        if not all(key in data for key in required_keys):
            logging.warning(f"Missing keys in request from {client_ip}: {data}")
            return jsonify({
                'status': 'error',
                'message': 'Missing one or more required fields: ir1, ir2, ultrasonic'
            }), 400

        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ir1': data['ir1'],
            'ir2': data['ir2'],
            'ultrasonic': data['ultrasonic']
        }

        with LOCK:
            df = pd.DataFrame([row])
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)

        logging.info(f"Data saved from {client_ip}: {row}")
        return jsonify({'status': 'success', 'message': 'Data recorded'}), 200

    except Exception as e:
        logging.error(f"Error processing request from {client_ip}: {e}")
        return jsonify({'status': 'error', 'message': 'Internal Server Error'}), 500
