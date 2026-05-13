# runtime/routes/vision.py
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from runtime.services.vision_service import (
    build_camera_status_stub,
    build_static_image_analysis_stub,
    capture_and_inspect_single_camera_snapshot,
    capture_single_camera_snapshot,
    get_vision_health_status,
    get_vision_safety_policy,
    inspect_image_bytes_lightweight,
    summarize_vision_foundation,
    validate_single_snapshot_permission_gate,
)

# ----------------------------
# Vision Routes — Phase 9A.2 / Phase 9B.1 / Phase 9B.2 / Phase 9B.3A / Phase 9C.3C / Phase 9I.5E
# ----------------------------
# Purpose:
# - Expose safe Swagger-testable vision endpoints.
# - Keep all vision functionality isolated from chat and voice systems.
# - Provide stable route foundations before real image analysis exists.
# - Add safe image upload validation.
# - Add local lightweight image inspection.
# - Add explicit opt-in OCR request forwarding for Phase 9C.3C.
# - Add explicit snapshot permission gate route for Phase 9I.5E.
# - Add explicit single-frame snapshot capture route for Phase 9I.5E.
# - Add explicit snapshot inspection bridge route for Phase 9I.5E.
#
# Safety:
# - No live monitoring.
# - No streaming.
# - No persistence.
# - No file saving.
# - No websocket integration.
# - No automatic background tasks.
# - No autonomous camera capture.
# - No external AI image model calls.
# - No face recognition.
# - No identity inference.
# - Camera snapshot capture only occurs when explicitly requested.
# - OCR only runs when explicitly requested.


_ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

_MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB


router = APIRouter(
    prefix="/v1/vision",
    tags=["vision"],
)


def _validate_uploaded_image_metadata(
    filename: str | None,
    content_type: str | None,
    size_bytes: int,
) -> dict:
    """
    Phase 9B.2 — Safe image upload validation.

    This only validates upload metadata and size.

    Safety:
    - Does NOT save the file.
    - Does NOT analyze the image.
    - Does NOT inspect image pixels.
    - Does NOT call any AI/image model.
    - Does NOT access camera.
    """
    safe_filename = (filename or "").strip()
    safe_content_type = (content_type or "").strip().lower()

    if not safe_filename:
        return {
            "valid": False,
            "reason": "missing_filename",
            "message": "Upload rejected because the file is missing a filename.",
        }

    if size_bytes <= 0:
        return {
            "valid": False,
            "reason": "empty_file",
            "message": "Upload rejected because the file is empty.",
        }

    if size_bytes > _MAX_IMAGE_UPLOAD_BYTES:
        return {
            "valid": False,
            "reason": "file_too_large",
            "message": (
                "Upload rejected because the image is larger than the Phase 9B.2 " "safety limit."
            ),
        }

    if safe_content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
        return {
            "valid": False,
            "reason": "unsupported_content_type",
            "message": (
                "Upload rejected because only PNG, JPEG, and WEBP images are allowed "
                "during Phase 9B.2."
            ),
        }

    return {
        "valid": True,
        "reason": "accepted",
        "message": "Upload metadata passed Phase 9B.2 validation.",
    }


@router.get("/health")
async def vision_health() -> dict:
    """
    Phase 9I.5E health endpoint.

    Confirms the vision foundation is installed and reachable.
    """
    return get_vision_health_status()


@router.get("/policy")
async def vision_policy() -> dict:
    """
    Returns the active Phase 9 vision safety policy.
    """
    return get_vision_safety_policy()


@router.get("/summary")
async def vision_summary() -> dict:
    """
    Human-readable summary of the current vision foundation.
    """
    return {
        "ok": True,
        "phase": "Phase 9I.5E",
        "summary": summarize_vision_foundation(),
    }


@router.get("/camera/status")
async def camera_status() -> dict:
    """
    Safe camera status endpoint.

    This may probe camera availability/indexes, but does NOT capture frames,
    save images, stream video, monitor live input, analyze scenes, or perform
    face recognition.
    """
    return build_camera_status_stub()


@router.get("/camera/permission")
async def camera_snapshot_permission_gate(
    explicit_snapshot_request: bool = False,
    requested_camera_index: int | None = None,
) -> dict:
    """
    Phase 9I.5E snapshot permission gate endpoint.

    This validates whether a future explicit one-frame snapshot capture would
    be allowed.

    Safety:
    - Does NOT capture frames.
    - Does NOT save images.
    - Does NOT stream video.
    - Does NOT analyze scenes.
    - Does NOT run OCR.
    - Does NOT perform face recognition.
    - Does NOT persist anything.
    """
    return validate_single_snapshot_permission_gate(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
    )


@router.post("/camera/snapshot")
async def camera_single_snapshot_capture(
    explicit_snapshot_request: bool = Form(False),
    requested_camera_index: int | None = Form(None),
) -> dict:
    """
    Phase 9I.5E explicit single-frame snapshot capture endpoint.

    This captures at most one memory-only PNG snapshot when explicitly allowed
    by the permission gate.

    Safety:
    - Requires explicit_snapshot_request=True.
    - Does NOT save images.
    - Does NOT stream video.
    - Does NOT start monitoring.
    - Does NOT analyze scenes.
    - Does NOT run OCR.
    - Does NOT perform face recognition.
    - Does NOT infer identity.
    - Does NOT persist anything.
    """
    return capture_single_camera_snapshot(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
    )


@router.post("/camera/snapshot/inspect")
async def camera_snapshot_inspection_bridge(
    explicit_snapshot_request: bool = Form(False),
    requested_camera_index: int | None = Form(None),
    explicit_ocr_request: bool = Form(False),
) -> dict:
    """
    Phase 9I.5E snapshot inspection bridge endpoint.

    This captures one explicit memory-only camera snapshot, then routes that
    snapshot into the existing lightweight local inspection pipeline.

    Safety:
    - Requires explicit_snapshot_request=True.
    - Does NOT save images.
    - Does NOT stream video.
    - Does NOT start monitoring.
    - Does NOT perform scene understanding.
    - Does NOT perform object recognition.
    - Does NOT perform face recognition.
    - Does NOT infer identity.
    - Does NOT call external AI vision models.
    - Does NOT persist anything.
    - OCR only runs when explicit_ocr_request=True.
    """
    return capture_and_inspect_single_camera_snapshot(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
        explicit_ocr_request=explicit_ocr_request,
    )


@router.post("/analyze")
async def analyze_image_stub(
    image_name: str | None = None,
    user_prompt: str | None = None,
) -> dict:
    """
    Preserved Phase 9A stub endpoint.

    This endpoint intentionally remains a non-analysis stub so older
    route tests stay stable.
    """
    return build_static_image_analysis_stub(
        image_name=image_name,
        user_prompt=user_prompt,
    )


@router.post("/upload")
async def upload_image_metadata_stub(
    file: UploadFile = File(...),
    explicit_ocr_request: bool = Form(False),
) -> dict:
    """
    Phase 9C.3C upload + local lightweight inspection endpoint with explicit OCR opt-in.

    This endpoint:
    - validates image uploads
    - performs local lightweight inspection
    - returns safe structured image metadata
    - optionally forwards explicit OCR opt-in request

    Safety:
    - Does NOT save files.
    - Does NOT use external AI models.
    - Does NOT perform OCR unless explicit_ocr_request=True.
    - Does NOT perform face recognition.
    - Does NOT access camera.
    - Does NOT persist anything.
    """
    content = await file.read()
    size_bytes = len(content)

    validation = _validate_uploaded_image_metadata(
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=size_bytes,
    )

    if not validation["valid"]:
        return {
            "ok": False,
            "phase": "Phase 9I.5E",
            "mode": "local_lightweight_image_inspection",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": size_bytes,
            "max_allowed_bytes": _MAX_IMAGE_UPLOAD_BYTES,
            "allowed_content_types": sorted(_ALLOWED_IMAGE_CONTENT_TYPES),
            "validation": validation,
            "saved": False,
            "analyzed": False,
            "camera_accessed": False,
            "explicit_ocr_request": explicit_ocr_request,
            "ocr_executed": False,
            "text_extracted": False,
            "message": validation["message"],
        }

    inspection = inspect_image_bytes_lightweight(
        image_bytes=content,
        filename=file.filename,
        content_type=file.content_type,
        explicit_ocr_request=explicit_ocr_request,
    )

    inspection["validation"] = validation
    inspection["max_allowed_bytes"] = _MAX_IMAGE_UPLOAD_BYTES
    inspection["allowed_content_types"] = sorted(_ALLOWED_IMAGE_CONTENT_TYPES)

    return inspection
