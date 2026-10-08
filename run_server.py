"""
Launcher for Fair Loan Predictor web app.
Run: python run_server.py
"""
import sys
import os

# Verify model exists
if not os.path.exists("model/model.pkl"):
    print("[ERROR] Model not found. Run: python train_model.py")
    sys.exit(1)

# Import and start app
from app import app, load_artifacts
load_artifacts()
print("\n  Fair Loan Predictor running at http://127.0.0.1:5000\n")
app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)
