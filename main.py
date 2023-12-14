from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port)
