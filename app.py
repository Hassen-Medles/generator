"""
app.py
Flask API — exposes the simulator as a REST endpoint.

Run:
    pip install flask
    python app.py

Endpoints:
    POST /simulate   { resident_id, conditions, simulation_days }
    GET  /           serves index.html
"""

from flask import Flask, request, jsonify, send_from_directory
from simulator import run_simulation
import os

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()

    resident_id     = data.get('resident_id', 'R001')
    conditions      = data.get('conditions', {})
    simulation_days = int(data.get('simulation_days', 180))

    if not conditions:
        return jsonify({'error': 'No conditions provided'}), 400

    result = run_simulation(resident_id, conditions, simulation_days)
    return jsonify(result)

if __name__ == '__main__':
    print("Simulator running at http://localhost:5000")
    app.run(debug=True, port=5000)
