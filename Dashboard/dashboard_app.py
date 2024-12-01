from flask import Flask, render_template, jsonify
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Path to the JSON file that contains dashboard data
DASHBOARD_JSON = os.path.join(os.path.dirname(__file__), "dashboard_data.json")

def read_dashboard_data():
    """Read metrics from the JSON file."""
    if os.path.exists(DASHBOARD_JSON):
        with open(DASHBOARD_JSON, "r") as file:
            data = json.load(file)
    else:
        data = {}
    return data

@app.route("/")
def index():
    """Render the dashboard with the metrics."""
    data = read_dashboard_data()
    return render_template("index.html", data=data)

@app.route("/api/data")
def get_data():
    """API endpoint to fetch JSON data."""
    data = read_dashboard_data()
    return jsonify(data)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
