# Radar Behaviour Simulator

This project is a small web-based simulator for generating synthetic resident behavior data based on selected health conditions and severity levels.

## Overview

The application combines a Flask backend and a simple frontend interface to simulate daily resident patterns such as:

- activity duration
- sleep window
- bathroom visits
- breathing rate
- sleep fragmentation
- other behavioral and physiological indicators

It is designed to help visualize how different conditions may influence daily patterns over time.

## Features

- Interactive simulator UI in the browser
- Multiple condition selection (for example reduced mobility, heart failure, COPD, sleep-disordered breathing)
- Adjustable severity levels
- Simulation over a configurable number of days
- Daily summary table and charts
- REST API endpoint for running simulations programmatically

## Requirements

Make sure Python is installed, then install the required packages:

```bash
pip install flask numpy pandas
```

## Run the app locally

Start the Flask server:

```bash
python app.py
```

Then open your browser at:

```text
http://localhost:5000
```

## API usage

The backend exposes a POST endpoint at /simulate.

Example request:

```json
{
  "resident_id": "R001",
  "conditions": {
    "heart_failure": "mild",
    "COPD": "severe"
  },
  "simulation_days": 180
}
```

## Project structure

- app.py: Flask application and routes
- simulator.py: simulation logic
- index.html: frontend layout
- script.js: browser-side logic and chart rendering
- style.css: styling for the UI
