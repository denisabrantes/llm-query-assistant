import os
import sys
import json
import socket
import requests

import glob
import torch
import transformers 
from transformers import AutoTokenizer

sys.path.append("./experiment")

from chat_format import ASSISTANT_PROMPT as ASSISTANT_PROMPT
from chat_format import EOS_TOKEN as EOS_TOKEN
from chat_format import get_chat_format as get_chat_format
from dataset_utils import load_or_create_dataset as load_or_create_dataset
from finetune import get_model_and_tokenizer as get_model_and_tokenizer

# Import MLDE packages
from determined.experimental import client as det
from determined import pytorch
from flask import render_template, session, request, redirect, Response

# Google Cloud packages
import google.oauth2.id_token
import google.auth.transport.requests

from . import home_bp

pipeline, tokenizer, models, datasets = None, None, None, None

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

@home_bp.route('/')
def index():
    print("--> Loading Home Page")
    return render_template('/home.html', hostname = hostname, port = 8080)


@home_bp.route('/execute_query', methods=['POST'])
def execute_query():
    query = request.form.get('query')
    google_auth_request = google.auth.transport.requests.Request()
    cloud_functions_url = 'https://us-central1-dai-dev-554.cloudfunctions.net/postgres_api'

    id_token = google.oauth2.id_token.fetch_id_token(google_auth_request, cloud_functions_url)

    headers = {'Authorization': f'Bearer {id_token}', 'Content-Type': 'application/json'}
    payload = {"query": query}

    response = requests.request("post", cloud_functions_url, json=payload, headers=headers)

    response_msg = {'message': response.text }
    return Response(str(response_msg), status=200, mimetype='application/json')


def load_model(model):
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, pipeline



@home_bp.route('/connect', methods=['POST'])
def connect():
    global pipeline
    global tokenizer
    
    model_name = ""

    try:
        tokenizer, pipeline = load_model(model_name)
        response_msg = {'message': 'Connection Successful' }
        return Response(str(response_msg), status=200, mimetype='application/json')
    except Exception as ex:
        response_msg = {'message': f"FAILED TO LOAD MODEL: {ex}" }
        return Response(str(response_msg), status=500, mimetype='application/json')



@home_bp.route('/list_models', methods=['GET'])
def list_models():
    global models
    
    try:
        f = open('/app/client/home/models.json')
        models = json.load(f)
        return Response(str(models), status=200, mimetype='application/json')
    except Exception as ex:
        response_msg = {'message': f"FAILED TO LOAD MODEL LIST: {ex}" }
        return Response(str(response_msg), status=500, mimetype='application/json')


@home_bp.route('/list_datasets', methods=['GET'])
def list_datasets():
    global datasets
    
    try:
        f = open('/app/client/home/datasets.json')
        datasets = json.load(f)
        return Response(str(datasets), status=200, mimetype='application/json')
    except Exception as ex:
        response_msg = {'message': f"FAILED TO LOAD MODEL LIST: {ex}" }
        return Response(str(response_msg), status=500, mimetype='application/json')

