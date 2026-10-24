from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..dependencies import checked, get_engine, read_upload, run_inference

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/national-id/extract")
async def extract_national_id(
    request: Request,
    file: Annotated[UploadFile, File(description="Front of the card: JPEG, PNG or WebP.")],
    include_text_regions: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    """Extract Cambodian National ID fields.

    Results with ``review_required: true`` must be confirmed by a person
    before use.
    """
    settings = request.app.state.settings
    engine = get_engine(request)
    data = await read_upload(file, settings)
    result = await run_inference(request, lambda: engine.extract_id(data))
    return checked(result.to_dict(include_text_regions=include_text_regions),
                   "khmer_national_id", settings)


@router.post("/quality-check")
async def quality_check(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image.")],
) -> dict[str, Any]:
    """Check whether an image is usable before extraction."""
    settings = request.app.state.settings
    engine = get_engine(request)
    data = await read_upload(file, settings)
    report = await run_inference(request, lambda: engine.check_image(data))
    return checked(report.to_dict(), "quality_report", settings)
