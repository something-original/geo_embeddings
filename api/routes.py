"""FastAPI route handlers for datasets, embeddings, and stats."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from api.dataset_worker import run_upload_feature_values, run_upload_territories
from api.embedding_worker import run_train_and_inference
from api.jobs import spawn_background
from api.stats_service import get_embedding_stats, get_indicators_list, get_model_info, get_dataset_sizes

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "datasets" / "uploads"


class AcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "accepted"
    message: str


class UploadTerritoriesParams(BaseModel):
    crs: str = "EPSG:4326"
    geom_col: str
    index_col: str | None = None


class UploadFeatureValuesParams(BaseModel):
    index_col: str | None = None
    target_col: str | None = None
    geom_col: str | None = None
    crs: str = "EPSG:4326"
    same_file_inference: bool = True
    train_file_name: str | None = None
    inference_file_name: str | None = None


class TrainInferenceParams(BaseModel):
    emb_dims: list[int] = Field(default_factory=lambda: [128, 192, 256])


def _parse_params(raw: str, model: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid JSON in params: {e}") from e
    try:
        return model.model_validate(data)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


async def _save_uploads(files: list[UploadFile]) -> dict[str, Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    out: dict[str, Path] = {}
    for uf in files:
        if not uf.filename:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Each file must have a filename")
        dest = UPLOAD_DIR / run_id / uf.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        out[uf.filename] = dest
    return out


def register_dataset_routes(router: APIRouter) -> None:

    @router.post("/upload_territories", response_model=AcceptedResponse, status_code=status.HTTP_200_OK)
    async def upload_territories(
        file: UploadFile = File(...),
        params: str = Form(...),
    ) -> AcceptedResponse:
        p = _parse_params(params, UploadTerritoriesParams)
        saved = await _save_uploads([file])
        csv_path = next(iter(saved.values()))
        spawn_background(
            run_upload_territories,
            str(csv_path),
            p.model_dump(),
            job_key="last_upload_job",
        )
        return AcceptedResponse(message="Territories file accepted; processing in background")

    @router.post("/upload_feature_values", response_model=AcceptedResponse, status_code=status.HTTP_200_OK)
    async def upload_feature_values(
        files: list[UploadFile] = File(...),
        params: str = Form(...),
    ) -> AcceptedResponse:
        p = _parse_params(params, UploadFeatureValuesParams)
        if p.same_file_inference and len(files) != 1:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "same_file_inference=true requires exactly one file",
            )
        if not p.same_file_inference and len(files) < 2:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "same_file_inference=false requires train and inference files",
            )
        saved = await _save_uploads(files)
        spawn_background(
            run_upload_feature_values,
            {k: str(v) for k, v in saved.items()},
            p.model_dump(),
            job_key="last_upload_job",
        )
        return AcceptedResponse(message="Feature values file(s) accepted; processing in background")


def register_embedding_routes(router: APIRouter) -> None:

    @router.post("/train_and_inference", response_model=AcceptedResponse, status_code=status.HTTP_200_OK)
    async def train_and_inference(
        body: TrainInferenceParams | None = None,
    ) -> AcceptedResponse:
        dims = body.emb_dims if body else [128, 192, 256]
        spawn_background(
            run_train_and_inference,
            dims,
            job_key="last_embedding_job",
        )
        return AcceptedResponse(message="Training and inference started in background")


def register_stats_routes(router: APIRouter, get_engine) -> None:

    @router.get("/")
    async def stats_overview(eng=Depends(get_engine)) -> dict[str, Any]:
        return await get_embedding_stats(eng)

    @router.get("/model")
    async def stats_model() -> dict[str, Any]:
        return await get_model_info()

    @router.get("/embedding-dimension")
    async def stats_dim() -> dict[str, int | None]:
        info = await get_model_info()
        return {"embedding_dim": info.get("embedding_dim")}

    @router.get("/dataset-sizes")
    async def stats_sizes(eng=Depends(get_engine)) -> dict[str, int]:
        return await get_dataset_sizes(eng)

    @router.get("/indicators")
    async def stats_indicators(eng=Depends(get_engine)) -> list[dict[str, Any]]:
        return await get_indicators_list(eng)
