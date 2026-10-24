from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Request, UploadFile

from ..dependencies import checked, get_engine, read_upload, run_inference

router = APIRouter(tags=["ocr"])


@router.post("/ocr")
async def recognize(request: Request,
                    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image.")],
                    ) -> dict[str, Any]:
    """Recognize general Khmer and English text."""
    settings = request.app.state.settings
    engine = get_engine(request)
    data = await read_upload(file, settings)
    result = await run_inference(request, lambda: engine.recognize(data))
    return checked(result.to_dict(), "general_text", settings)
