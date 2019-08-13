import json
import os
import requests
import yaml


def raster_prediction(tif_path):
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model_call_config.yml'), 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)
    model_api_cfg = cfg["model-api"]
    API_ENDPOINT = model_api_cfg["url"] + ':' + str(model_api_cfg["port"]) + model_api_cfg["endpoint"]
    data = {"image_path": tif_path}
    r = requests.post(url=API_ENDPOINT, json=data)
    result = r.text
    datastore = json.loads(result)
    return datastore
