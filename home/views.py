import os
import sys
import json
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

ip_address = "34.66.149.119"

class FlaskBackend(object):
    def __init__(self, app, **configs):
        self.app = app
        self.configs(**configs)
        self.pipeline = None
        self.tokenizer = None
        self.models = None
        self.datasets = None

    @home_bp.route('/')
    def index(self):
        #print("--> Loading Home Page", flush=True)
        if self.pipeline is None:
            self.model_status = "not_loaded"
        else:
            model_status = "loaded"
        return render_template('/home.html', hostname = ip_address, port = 8080, model_status = model_status)


    @home_bp.route('/connect', methods=['POST'])
    def connect(self):
        if self.pipeline is None:
            model_status = "not_loaded"
        else:
            model_status = "loaded"
        response_msg = {'message': "Connected", "model_status" : model_status }
        return Response(str(response_msg), status=200, mimetype='application/json')


    @home_bp.route('/execute_query', methods=['POST'])
    def execute_query(self):
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
    def load_model(self):
        isLocalModel = False

        model_name = request.values.get('model_id')
        if "local" in model_name:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4" 
            isLocalModel = True

        if isLocalModel:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                response_msg = {'message': f"Model Loaded Successfully: {model_name}", "model_status" : "loaded" }
                return Response(str(response_msg), status=200, mimetype='application/json')
            except Exception as ex:
                response_msg = {'message': f"FAILED TO LOAD MODEL: {ex}", "model_status" : "not_loaded" }
                return Response(str(response_msg), status=500, mimetype='application/json')


    @home_bp.route('/list_models', methods=['GET'])
    def list_models(self):
        if self.pipeline is None:
            model_status = "not_loaded"
        else:
            model_status = "loaded"
        try:
            f = open('/app/client/home/models.json')
            self.models = json.load(f)
            response_msg = {'models': self.models, "model_status" : model_status }
            return Response(str(response_msg), status=200, mimetype='application/json')
        except Exception as ex:
            response_msg = {'message': f"FAILED TO LOAD MODEL LIST: {ex}", "model_status" : model_status }
            return Response(str(response_msg), status=500, mimetype='application/json')


    @home_bp.route('/list_datasets', methods=['GET'])
    def list_datasets(self):
        try:
            f = open('/app/client/home/datasets.json')
            self.datasets = json.load(f)
            return Response(str(self.datasets), status=200, mimetype='application/json')
        except Exception as ex:
            response_msg = {'message': f"FAILED TO LOAD MODEL LIST: {ex}" }
            return Response(str(response_msg), status=500, mimetype='application/json')


    @home_bp.route('/question', methods=['POST'])
    def answer_question(self):
            try:
                eos_token_id = self.tokenizer.get_vocab()[EOS_TOKEN]
                prompt = request.values.get('prompt')
                model_id = request.values.get('model')

                if "local" in model_id:
                    try:
                        sequences = self.pipeline(
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
                        response_msg = {'response': seq['generated_text'] }
                        return Response(str(response_msg), status=200, mimetype='application/json')
                    except Exception as ex:
                        response_msg = {'message': f"FAILED TO GENERATE PREDICTION: {ex}"}
                        return Response(str(response_msg), status=500, mimetype='application/json')

            except Exception as ex:
                response_msg = {'message': f"FAILED TO LOAD MODEL FOR PREDICTION: {ex}"}
                return Response(str(response_msg), status=500, mimetype='application/json')
