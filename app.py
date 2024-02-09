import sys
sys.path.append("./home/experiment")
from flask import Flask, Blueprint, render_template, jsonify,  request
from home import home_bp


app = Flask(__name__)
app.register_blueprint(home_bp)
app.secret_key = "secretkey123"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)
