import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    status
)

from pydantic import BaseModel, ConfigDict, Field
from parsers.osm_municipal import parse_mun_data, form_mun_geometry
from shapely import make_valid
from shapely.geometry import shape
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

import uvicorn

from config import (
    DB_URL,
    HOST,
    INIT_EMBEDDINGS,
    PORT,
    QDRANT_COLLECTION
)

from api.routes import (
    register_dataset_routes,
    register_embedding_routes,
    register_stats_routes,
)
from embedding_qdrant import (
    ensure_inference_embeddings_qdrant_async,
    make_qdrant_client,
)
from utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

engine = create_async_engine(url=DB_URL)
session_maker = async_sessionmaker(bind=engine)


async def get_session():
    async with session_maker() as session:
        yield session


async def get_engine():
    yield engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if INIT_EMBEDDINGS:
            root_dir = Path(__file__).resolve().parent
            logger.info(
                "INIT_EMBEDDINGS=true: loading default municipal datasets into DB"
            )
            await parse_mun_data(root_dir, engine)
            await form_mun_geometry(engine)
            await ensure_inference_embeddings_qdrant_async()
        else:
            logger.info(
                "INIT_EMBEDDINGS=false: skipping parse_mun_data / form_mun_geometry; "
                "waiting for dataset uploads via API"
            )

        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield
    logger.info("Shutting down")
    await engine.dispose()


app = FastAPI(lifespan=lifespan)

embedding_router = APIRouter(prefix="/embeddings")
datasets_router = APIRouter(prefix="/datasets")
stats_router = APIRouter(prefix="/stats")


def _geojson_geometry_from_body(body: dict[str, Any]) -> dict[str, Any]:
    t = body.get("type")
    if t == "Feature":
        g = body.get("geometry")
        if not isinstance(g, dict):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Feature.geometry must be an object"
            )
        return g
    if isinstance(t, str):
        return body
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "Body must be a GeoJSON Geometry or a Feature with a geometry field",
    )


class FromPolygonResponse(BaseModel):
    """
    Компактное представление:
    - vectors_b64: сырые байты float32 little-endian, матрица (n, dim) по строкам (C-order), закодированные в base64.
    - geojson: FeatureCollection; i-й feature соответствует municipality_ids[i]; в geometry — полигон из БД.
    """

    model_config = ConfigDict(extra="forbid")

    dim: int
    municipality_ids: list[int]
    vectors_b64: str
    geojson: dict[str, Any]
    vectors_encoding: Literal["float32_le_rowmajor_base64"] = (
        "float32_le_rowmajor_base64"
    )
    missing_embedding_ids: list[int] = Field(default_factory=list)


def _qdrant_retrieve_vectors(municipality_ids: list[int]) -> dict[int, list[float]]:
    if not municipality_ids:
        return {}
    client = make_qdrant_client()
    out: dict[int, list[float]] = {}
    chunk = 256
    for i in range(0, len(municipality_ids), chunk):
        part = municipality_ids[i : i + chunk]
        records = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=part,
            with_vectors=True,
        )
        for r in records:
            vid = r.id
            if isinstance(vid, int):
                mid = vid
            else:
                try:
                    mid = int(vid)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
            vec = r.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), None)
            if vec is not None:
                out[mid] = list(vec)
    return out


@embedding_router.post("/from_polygon", response_model=FromPolygonResponse)
async def embeddings_from_polygon(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> FromPolygonResponse:

    geom = _geojson_geometry_from_body(body)
    try:
        g = shape(geom)
    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Invalid GeoJSON geometry: {e}"
        ) from e
    if g.is_empty:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty geometry")
    if not g.is_valid:
        g = make_valid(g)
    geom_for_sql = json.dumps(g.__geo_interface__)

    sql = text(
        """
        SELECT m.id AS mid, ST_AsGeoJSON(m.geometry)::text AS gj
        FROM municipalities m
        WHERE m.geometry IS NOT NULL
          AND ST_Intersects(
              ST_Transform(m.geometry, 4326),
              ST_SetSRID(ST_GeomFromGeoJSON(:poly_json), 4326)
          )
        ORDER BY m.id
        """
    )
    rows = (await session.execute(sql, {"poly_json": geom_for_sql})).mappings().all()
    if not rows:
        return FromPolygonResponse(
            dim=0,
            municipality_ids=[],
            vectors_b64="",
            geojson={"type": "FeatureCollection", "features": []},
        )

    mids = [int(r["mid"]) for r in rows]
    id_to_geom: dict[int, dict[str, Any]] = {}
    for r in rows:
        mid = int(r["mid"])
        try:
            id_to_geom[mid] = json.loads(r["gj"])
        except json.JSONDecodeError:
            continue

    vecs_map = await asyncio.to_thread(_qdrant_retrieve_vectors, mids)
    missing = [mid for mid in mids if mid not in vecs_map]

    present: list[int] = []
    rows_vec: list[list[float]] = []
    features: list[dict[str, Any]] = []
    for mid in mids:
        vec = vecs_map.get(mid)
        if vec is None:
            continue
        gj = id_to_geom.get(mid)
        if gj is None:
            continue
        present.append(mid)
        rows_vec.append([float(x) for x in vec])
        features.append(
            {
                "type": "Feature",
                "id": mid,
                "properties": {"municipality_id": mid},
                "geometry": gj,
            }
        )

    if not present:
        return FromPolygonResponse(
            dim=0,
            municipality_ids=[],
            vectors_b64="",
            geojson={"type": "FeatureCollection", "features": []},
            missing_embedding_ids=missing,
        )

    dim = len(rows_vec[0])
    for mid, row in zip(present, rows_vec, strict=True):
        if len(row) != dim:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Inconsistent embedding dim for municipality_id={mid}",
            )

    arr = np.asarray(rows_vec, dtype=np.float32)
    b64 = base64.b64encode(arr.tobytes(order="C")).decode("ascii")

    return FromPolygonResponse(
        dim=dim,
        municipality_ids=present,
        vectors_b64=b64,
        geojson={"type": "FeatureCollection", "features": features},
        missing_embedding_ids=missing,
    )


register_dataset_routes(datasets_router)
register_embedding_routes(embedding_router)
register_stats_routes(stats_router, get_engine)

app.include_router(embedding_router)
app.include_router(datasets_router)
app.include_router(stats_router)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
