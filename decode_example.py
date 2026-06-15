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

geo_embeddings = requests.post(GEO_EMBEDDINGS_URL, data=request_polygon)
json_answer = geo_embeddings.json()

b64 = json_answer["vectors_b64"]
d_type = json_answer["data_type"]
shape = json_answer["shape"]

raw_bytes = base64.b64decode(b64)
arr = np.frombuffer(raw_bytes, dtype=d_type).reshape(shape)
