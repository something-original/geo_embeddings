import shapely
import shapely.ops
import math
import pyproj

def _split_bounds(minx, miny, maxx, maxy, maxstep):
    height = maxy - miny
    width = maxx - minx

    stepy = height / math.ceil(height / maxstep)
    stepx = width / math.ceil(width / maxstep)

    bboxes = []

    y1 = maxy
    x1 = minx

    while y1 > miny:
        # Add step to lng
        y2 = y1 - stepy

        # Check if lng smaller than needed
        if y2 < miny:
            y2 = miny

        while x1 < maxx:
            # Add step to lng
            x2 = x1 + stepx

            # Check if lng bigger than needed
            if x2 > maxx:
                x2 = maxx

            bboxes.append((x1, y2, x2, y1))

            x1 = x2

        y1 = y2
        x1 = minx

    return bboxes


def _convert_bounds_to_cian_bbox(minx, miny, maxx, maxy):
    return {
        "topLeft": {"lat": maxy, "lng": minx},
        "bottomRight": {"lat": miny, "lng": maxx},
    }


def get_cian_bboxes_for_geojson(geojson, maxstep):
    geometry4326 = shapely.from_geojson(geojson)

    epsg4326 = pyproj.CRS('EPSG:4326')
    epsg3857 = pyproj.CRS('EPSG:3857')

    transform4326to3857 = pyproj.Transformer.from_crs(epsg4326, epsg3857, always_xy=True).transform
    transform3857to4326 = pyproj.Transformer.from_crs(epsg3857, epsg4326, always_xy=True).transform

    geometry3857 = shapely.ops.transform(transform4326to3857, geometry4326)

    bboxes = []
    for bounds in _split_bounds(*geometry3857.bounds, maxstep):
        box3857 = shapely.box(*bounds)
        if shapely.intersects(box3857, geometry3857):
            bboxes.append(_convert_bounds_to_cian_bbox(*shapely.ops.transform(transform3857to4326, box3857).bounds))

    return bboxes
