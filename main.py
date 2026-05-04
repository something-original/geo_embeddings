from parsers.osm_municipal import parse_mun_data, form_mun_geometry
from pathlib import Path

root_dir = Path(__file__).resolve().parent

#parse_mun_data(root_dir)
form_mun_geometry()
