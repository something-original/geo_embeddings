import base64
import numpy as np
import requests

GEO_EMBEDDINGS_URL = "http://url_to_service"
request_polygon = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [30.128137, 59.989371],
                [30.182044, 59.985937],
                [30.172773, 59.994522],
                [30.131914, 59.997098],
                [30.128137, 59.989371],
            ]
        ],
    },
    "properties": None,
}


geo_embeddings = requests.post(GEO_EMBEDDINGS_URL, json=request_polygon)

json_answer = geo_embeddings.json()
raw_bytes = base64.b64decode(json_answer['vectors_b64'])
all_numbers = np.frombuffer(raw_bytes, dtype='<f4')
vectors_matrix = all_numbers.reshape(-1, json_answer['dim'])