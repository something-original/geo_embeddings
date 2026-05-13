import logging

from contextlib import asynccontextmanager
from pathlib import Path
from parsers.osm_municipal import parse_mun_data, form_mun_geometry
from config import DB_URL, HOST, PORT
from fastapi import APIRouter, Depends, FastAPI, Response, status
import uvicorn

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text

from utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

engine = create_async_engine(url=DB_URL)
session_maker = async_sessionmaker(bind=engine)


async def get_session():
    async with session_maker() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        root_dir = Path(__file__).resolve().parent
        await parse_mun_data(root_dir)
        await form_mun_geometry()
        yield
    except KeyboardInterrupt:
        pass

app = FastAPI(lifespan=lifespan)

embedding_router = APIRouter(prefix="/embeddings")
datasets_router = APIRouter(prefix="/datasets")
stats_router = APIRouter(prefix="/stats")


@stats_router.get("/{query}")
async def test_get(query: int, session=Depends(get_session)):
    res = (await session.execute(
        text(f"SELECT {query};")
    )).scalar()
    return res


@stats_router.post("/")
async def test_post(data: dict, session=Depends(get_session)):
    query = data.get("query", 0)
    if query == 0:
        msg = "zero!"
        return Response(msg, status.HTTP_500_INTERNAL_SERVER_ERROR)

    res = (await session.execute(
        text(f"SELECT {query};")
    )).scalar()
    return {"res": res}


app.include_router(embedding_router)
app.include_router(datasets_router)
app.include_router(stats_router)


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
