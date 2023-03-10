import os
import sys
import logging
from datetime import datetime
import json
from uuid import uuid4, UUID
from typing import Type
from functools import wraps
import requests
from flask import Flask, request, Blueprint
from flask import jsonify
from flask_cors import CORS
from search import MPNetEmbedder
logging.basicConfig(format='%(levelname)s :: %(asctime)s :: %(message)s', level=logging.DEBUG)

app = Flask(__name__, static_folder=None)
app.config['JSON_SORT_KEYS'] = False
app.url_map.strict_slashes = False
bp = Blueprint('xDD-askem-api', __name__, static_folder="static")

VERSION = 1

model = MPNetEmbedder()

@bp.route('/', methods=["GET"])
def index():
    return {
                "description" : "API for reserving or registering JSON objects for storage within the xDD system.",
                "routes": {
                    f"/reserve" : "Reserve a block of ASKEM-IDs for later registration.",
                    f"/register" : "Register a location for a reserved ASKEM-ID.",
                    f"/create" : "Create and register ASKEM-IDs for existing resources.",
                    f"/object" : "Retrieve or search for an object."
                    }
        }

@bp.route('/embed', methods=['GET'])
def embed():
    """Data service operation: execute a vector query on the specified word."""

    usage = {
            "v" : VERSION,
            "description" : "...",
            "options":  {
                }
            }
    # TODO: add parameter for the number of similar documents to return

    text = request.values.get("text")
    logging.info(f"embedding {text}")
    embedding = model.embed([text])
    return jsonify(embedding.tolist())

if 'PREFIX' in os.environ:
    logging.info(f"Stripping {os.environ['PREFIX']}")
    app.register_blueprint(bp, url_prefix=os.environ['PREFIX'])
else:
    logging.info("No prefix stripped.")
    app.register_blueprint(bp)
CORS(app)

#if __name__ == '__main__':
#    app.run(debug=True,host='0.0.0.0', port=80)

