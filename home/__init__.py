from flask import Blueprint
import sys
sys.path.append("./experiment")

home_bp = Blueprint('home_bp', __name__, template_folder='templates', static_folder='../static', static_url_path='assets', url_prefix='/')

from . import views
