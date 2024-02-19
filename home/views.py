import os
import sys
import json
import glob
import torch
import requests
import transformers
from cache_common import cache
from determined import pytorch
from transformers import AutoTokenizer
from determined.experimental import client as det
from flask import render_template, request, Response

sys.path.append("./experiment")

from chat_format import ASSISTANT_PROMPT as ASSISTANT_PROMPT
from chat_format import EOS_TOKEN as EOS_TOKEN
from chat_format import get_chat_format as get_chat_format
from dataset_utils import load_or_create_dataset as load_or_create_dataset
from finetune import get_model_and_tokenizer as get_model_and_tokenizer

# Google Cloud packages
import google.oauth2.id_token
import google.auth.transport.requests

from . import home_bp

ip_address = "34.72.65.1"

def printout(message):
    print(message, flush=True)

def format_failure_message(ex, message):
    cl_ex = str(ex).replace("\"", "'")
    return {"message": f"FAILED TO {message} : {cl_ex}"}

def get_model_list():
    try:
        f = open('/app/client/home/models.json')
        return json.load(f)
    except Exception as ex:
        print(f"--> FAILED TO LIST MODELS: {ex}")

def get_model_type(model_name):
    model_type = None
    model_list = get_model_list()
    for model in model_list:
        if model_name == model['name']:
            model_type = model['type']    
    return model_type

@home_bp.route('/')
def index():
    printout("--> Loading Home Page")
    return render_template('/home.html', hostname = ip_address, port = 8080)

@home_bp.route('/connect', methods=['POST'])
def connect():
    response_msg = {'message': "Connected"}
    return Response(str(response_msg), status=200, mimetype='application/json')

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

@home_bp.route('/load_model', methods=['POST'])
def load_model():
    model_name = request.values.get('model_id')
    model_type = get_model_type(model_name)

    if "hf" in model_type:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            cache.set("tokenizer", tokenizer)
            cache.set("pipeline", pipeline)
            cache.set("active_model", model_name)

            response_msg = {'message': f"Model Loaded Successfully: {model_name}"}
            return Response(str(response_msg), status=200, mimetype='application/json')
        except Exception as ex:
            response_msg = format_failure_message(ex, "LOAD MODEL")
            return Response(str(response_msg), status=500, mimetype='application/json')

@home_bp.route('/list_models', methods=['GET'])
def list_models():
    try:
        models = get_model_list()
        response_msg = {'models': models}
        return Response(str(response_msg), status=200, mimetype='application/json')
    except Exception as ex:
        response_msg = format_failure_message(ex, "LOAD MODEL LIST")
        return Response(str(response_msg), status=500, mimetype='application/json')

@home_bp.route('/list_datasets', methods=['GET'])
def list_datasets():
    try:
        f = open('/app/client/home/datasets.json')
        datasets = json.load(f)
        return Response(str(datasets), status=200, mimetype='application/json')
    except Exception as ex:
        response_msg = format_failure_message(ex, "LOAD MODEL LIST")
        return Response(str(response_msg), status=500, mimetype='application/json')

@home_bp.route('/question', methods=['POST'])
def answer_question():
    try:
        tokenizer = cache.get("tokenizer")
        pipeline = cache.get("pipeline")
        eos_token_id = tokenizer.get_vocab()[EOS_TOKEN]

        form_data = request.json
        model_name = form_data["model"]
        prompt = form_data["prompt"]
        # model_name = request.values.get('model')
        # prompt = request.values.get('prompt')
        
        model_type = get_model_type(model_name)
        
        printout(f"--> Model Name: {model_name} | Model Type: {model_type}")# | prompt: {prompt}")

        if "hf" in model_type:
            try:
                sequences = pipeline(
                    prompt,
                    do_sample=True,
                    top_k=50,
                    top_p = 0.9,
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                    max_new_tokens=1024,
                    eos_token_id=eos_token_id,
                )
                seq = sequences[0]
                response_text = seq['generated_text'].replace("\"", "'").replace("''","'")
                response_msg = {'response': response_text.replace("\"", "'") }
                printout(f"--> MODEL RESPONSE: {seq['generated_text']}")
                return Response(str(response_msg), status=200, mimetype='application/json')
            except Exception as ex:
                printout(ex)
                response_msg = format_failure_message(ex, "GENERATE PREDICTION")
                return Response(str(response_msg), status=500, mimetype='application/json')

    except Exception as ex:
        printout(ex)
        response_msg = format_failure_message(ex, "LOAD MODEL FOR PREDICTION")
        return Response(str(response_msg), status=500, mimetype='application/json')
