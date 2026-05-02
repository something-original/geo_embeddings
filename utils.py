import geopandas as gpd
from shapely import make_valid
from shapely.wkt import loads


def get_geometry_points(
    dataset_path: str,
    geom_col: str = 'geometry'
) -> list[tuple[float, float]]:

    df = gpd.read_file(dataset_path)
    df[geom_col] = df[geom_col].apply(loads)
    df[geom_col] = df[geom_col].apply(make_valid)
    
    df = gpd.GeoDataFrame(df).set_geometry(geom_col).set_crs('EPSG:4326')
    df['centroids'] = df[geom_col].centroid
    
    return df['centroids'].apply(lambda p: (p.y, p.x)).tolist()
    