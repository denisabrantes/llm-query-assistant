import sys
from flask import Flask
from cache_common import cache
from pathlib import Path

sys.path.append("./home/experiment")

from home import home_bp

app = Flask(__name__)
cache.init_app(app=app, config={"CACHE_TYPE": "filesystem",'CACHE_DIR': Path('/tmp')})

app.register_blueprint(home_bp)
app.secret_key = "secretkey123"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)
