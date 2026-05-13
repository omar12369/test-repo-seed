# runtime/services/vision_service.py
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from io import BytesIO
from shutil import which
from typing import Any, Optional

# ----------------------------
# Vision Service — Phase 9I.5E
# ----------------------------
# Purpose:
# - Establish the safe foundation for SEED Runtime vision support.
# - Provide structured response shapes for future image/camera interpretation.
# - Keep this isolated from chat, voice, memory, and websocket systems.
# - Add local lightweight image inspection for Phase 9B.3A.
# - Refine deterministic local feature summaries for Phase 9B.3B.
# - Add deterministic local image quality assessment for Phase 9B.3C.
# - Add preprocessing readiness assessment for Phase 9B.3D.
# - Add controlled OCR foundation planning for Phase 9C.1.
# - Add OCR dependency and execution preparation awareness for Phase 9C.2.
# - Add explicit opt-in OCR execution pathway for Phase 9I.5E.
# - Add explicit inspection-flow OCR execution opt-in for Phase 9I.5E.
# - Add OCR result cleanup and quality refinement for Phase 9I.5E.
# - Add local in-memory OCR preprocessing enhancement for Phase 9I.5E.
# - Add safe camera availability detection foundation for Phase 9I.5E.
# - Add camera index identification metadata for Phase 9I.5E.
# - Add explicit single-snapshot permission gate for Phase 9I.5E.
# - Add explicit single-frame camera snapshot capture for Phase 9I.5E.
# - Add snapshot inspection bridge for Phase 9I.5E.
# - Add structured OCR result layer for Phase 9I.5E.
# - Add deterministic OCR quality refinement and adaptive preprocessing profiles for Phase 9I.5E.
# - Add controlled deterministic scene-condition classification foundation for Phase 9I.5E.
# - Add deterministic OCR and scene coordination layer for Phase 9I.5E.
# - Add deterministic human-readable vision response synthesis for Phase 9I.5E.
# - Add deterministic vision-to-voice coordination metadata for Phase 9I.5E.
# - Add controlled voice payload integration packaging for Phase 9I.5E.
# - Add explicit voice handoff bridge metadata for Phase 9I.5E.
# - Add optional explicit voice execution gate metadata for Phase 9I.5E.
# - Add end-to-end vision-to-voice orchestration validation metadata for Phase 9I.5E.
# - Add deterministic OCR-to-narration construction for Phase 9I.5E.
# - Add structured deterministic scene profile packaging for Phase 10B.
# - Add deterministic narration planning metadata for Phase 10C.
# - Add deterministic narration readiness scoring metadata for Phase 10D.
#
# Safety:
# - Camera detection, snapshot permission gating, and explicit one-frame capture only.
# - No file writes.
# - Snapshot capture returns memory-only PNG bytes when explicitly requested.
# - No external AI image model calls.
# - No background monitoring.
# - No face recognition.
# - No identity inference.
# - No semantic scene understanding or object recognition.
# - No persistence.
# - OCR execution requires explicit function-level opt-in.
# - No automatic OCR execution.
# - No route registration in this file.


VISION_SERVICE_VERSION = "9I.5E"

# Phase 9I.5E controlled high-resolution snapshot preferences.
# These are best-effort camera property requests only; unsupported cameras may
# safely fall back to their default capture resolution.
SNAPSHOT_TARGET_WIDTH = 1280
SNAPSHOT_TARGET_HEIGHT = 720
SNAPSHOT_TARGET_FPS = 30
SNAPSHOT_WARMUP_READS = 3


@dataclass
class VisionHealthStatus:
    ok: bool
    phase: str
    service: str
    version: str
    camera_enabled: bool
    camera_detection_enabled: bool
    camera_available: bool
    camera_count: int
    preferred_camera_index: Optional[int]
    live_monitoring_enabled: bool
    image_analysis_enabled: bool
    ocr_foundation_installed: bool
    ocr_dependency_detection_enabled: bool
    ocr_execution_enabled: bool
    ocr_runtime_ready: bool
    ocr_requires_explicit_opt_in: bool
    message: str
    created_at: str


@dataclass
class VisionSafetyPolicy:
    camera_access: str
    live_monitoring: str
    face_identity_recognition: str
    sensitive_attribute_inference: str
    persistence: str
    file_writes: str
    ocr_execution: str
    ocr_engine_loaded: bool
    ocr_user_permission_required: bool
    ocr_automatic_execution: str
    text_extraction: str
    user_trigger_required: bool


@dataclass
class VisionAnalysisStub:
    ok: bool
    phase: str
    mode: str
    summary: str
    detected_items: list[str]
    visible_text: Optional[str]
    uncertainty: str
    safety_policy: dict[str, Any]
    created_at: str


@dataclass
class OCRStubResult:
    ok: bool
    phase: str
    mode: str
    filename: Optional[str]
    ocr_foundation_installed: bool
    ocr_enabled: bool
    ocr_engine_loaded: bool
    ocr_dependencies_available: bool
    pytesseract_available: bool
    tesseract_binary_available: bool
    ocr_runtime_ready: bool
    ocr_execution_allowed: bool
    ocr_execution_performed: bool
    ocr_execution_recommended: bool
    ocr_readiness: str
    extracted_text: Optional[str]
    text_detected: bool
    confidence: Optional[float]
    dependency_report: dict[str, Any]
    summary: str
    uncertainty: str
    safety_policy: dict[str, Any]
    created_at: str


@dataclass
class OCRExecutionResult:
    ok: bool
    phase: str
    mode: str
    filename: Optional[str]
    explicit_ocr_request: bool
    ocr_enabled: bool
    ocr_dependencies_available: bool
    ocr_execution_allowed: bool
    ocr_execution_performed: bool
    text_extracted: bool
    text_detected: bool
    extracted_text: Optional[str]
    cleaned_extracted_text: Optional[str]
    extracted_text_length: int
    normalized_text_lines: list[str]
    line_count: int
    word_count: int
    character_count: int
    text_density: float
    ocr_text_quality_score: float
    ocr_text_structure: str
    structured_text_summary: str
    ocr_preprocessing_profile: str
    ocr_preprocessing_profile_reason: str
    ocr_readability_score: float
    ocr_readability_label: str
    ocr_noise_score: float
    ocr_noise_label: str
    ocr_refinement_summary: str
    preprocessing_applied: bool
    preprocessing_steps: list[str]
    preprocessing_summary: str
    ocr_quality_label: str
    ocr_confidence_label: str
    confidence: Optional[float]
    error: Optional[str]
    message: str
    summary: str
    dependency_report: dict[str, Any]
    safety_policy: dict[str, Any]
    saved: bool
    camera_accessed: bool
    created_at: str
    raw_ocr_text: Optional[str] = None
    fallback_ocr_text: Optional[str] = None
    ocr_config_used: Optional[str] = None
    adaptive_threshold_applied: bool = False
    inversion_applied: bool = False
    preprocessed_image_width: Optional[int] = None
    preprocessed_image_height: Optional[int] = None
    selected_ocr_pass: Optional[str] = None
    ocr_pass_diagnostics: Optional[list[dict[str, Any]]] = None


@dataclass
class CameraSnapshotPermissionGateResult:
    ok: bool
    phase: str
    mode: str
    explicit_snapshot_request: bool
    requested_camera_index: Optional[int]
    selected_camera_index: Optional[int]
    camera_available: bool
    camera_count: int
    available_camera_indexes: list[int]
    permission_required: bool
    permission_granted: bool
    single_capture_allowed: bool
    snapshot_capture_enabled: bool
    frame_captured: bool
    frame_analyzed: bool
    frame_saved: bool
    live_monitoring_enabled: bool
    error: Optional[str]
    message: str
    safety_policy: dict[str, Any]
    created_at: str


@dataclass
class CameraSnapshotCaptureResult:
    ok: bool
    phase: str
    mode: str
    explicit_snapshot_request: bool
    requested_camera_index: Optional[int]
    selected_camera_index: Optional[int]
    permission_gate: dict[str, Any]
    snapshot_capture_enabled: bool
    single_capture_allowed: bool
    camera_available: bool
    camera_opened: bool
    frame_captured: bool
    frame_analyzed: bool
    frame_saved: bool
    live_monitoring_enabled: bool
    image_format: Optional[str]
    image_mode: Optional[str]
    width: Optional[int]
    height: Optional[int]
    size_bytes: int
    image_bytes_base64: Optional[str]
    error: Optional[str]
    message: str
    safety_policy: dict[str, Any]
    saved: bool
    camera_released: bool
    created_at: str


@dataclass
class CameraSnapshotInspectionBridgeResult:
    ok: bool
    phase: str
    mode: str
    explicit_snapshot_request: bool
    explicit_ocr_request: bool
    requested_camera_index: Optional[int]
    selected_camera_index: Optional[int]
    snapshot_capture_performed: bool
    snapshot_capture_ok: bool
    inspection_performed: bool
    inspection_ok: bool
    ocr_execution_allowed: bool
    ocr_execution_performed: bool
    text_extracted: bool
    snapshot_result: dict[str, Any]
    inspection_result: Optional[dict[str, Any]]
    error: Optional[str]
    message: str
    summary: str
    safety_policy: dict[str, Any]
    saved: bool
    camera_accessed: bool
    camera_released: bool
    live_monitoring_enabled: bool
    frame_saved: bool
    analyzed_by_ai_model: bool
    created_at: str


@dataclass
class LocalImageInspectionResult:
    ok: bool
    phase: str
    mode: str
    filename: Optional[str]
    content_type: Optional[str]
    size_bytes: int
    image_format: Optional[str]
    image_mode: Optional[str]
    width: Optional[int]
    height: Optional[int]
    orientation: str
    aspect_ratio: Optional[float]
    aspect_category: str
    resolution_category: str
    brightness_estimate: Optional[float]
    brightness_label: str
    color_profile: str
    sharpness_estimate: Optional[float]
    sharpness_label: str
    contrast_estimate: Optional[float]
    contrast_label: str
    cleanliness_label: str
    quality_assessment: str
    scene_classification: str
    scene_confidence: str
    scene_summary: str
    scene_profile: dict[str, Any]
    narration_plan: dict[str, Any]
    narration_readiness: dict[str, Any]
    voice_bridge_refinement: dict[str, Any]
    ocr_priority_recommendation: str
    coordination_mode: str
    vision_routing_decision: str
    ocr_action_recommendation: str
    scene_ocr_alignment: str
    coordination_confidence: str
    coordination_summary: str
    vision_response_summary: str
    vision_response_detail: str
    vision_response_recommended_next_action: str
    vision_response_safety_note: str
    vision_voice_ready: bool
    vision_voice_summary: str
    vision_voice_priority: str
    vision_voice_narration: str
    vision_voice_safety_note: str
    vision_voice_payload_ready: bool
    vision_voice_payload: dict[str, Any]
    vision_voice_payload_format: str
    vision_voice_speech_style: str
    vision_voice_delivery_mode: str
    voice_handoff_ready: bool
    voice_handoff_payload: dict[str, Any]
    voice_handoff_target_engine: str
    voice_handoff_safe: bool
    voice_handoff_summary: str
    voice_execution_available: bool
    voice_execution_allowed: bool
    voice_execution_request: dict[str, Any]
    voice_execution_safe: bool
    voice_execution_summary: str
    ocr_readiness: str
    ocr_foundation_status: dict[str, Any]
    ocr_environment_status: dict[str, Any]
    ocr_execution_recommended: bool
    explicit_ocr_request: bool
    ocr_execution_result: Optional[dict[str, Any]]
    object_detection_readiness: str
    model_input_readiness: str
    preprocessing_recommendation: str
    preprocessing_summary: str
    inspection_confidence: str
    summary: str
    detected_items: list[str]
    visible_text: Optional[str]
    uncertainty: str
    safety_policy: dict[str, Any]
    saved: bool
    analyzed_by_ai_model: bool
    camera_accessed: bool
    ocr_executed: bool
    text_extracted: bool
    created_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_orientation(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "unknown"

    if width == height:
        return "square"

    if width > height:
        return "landscape"

    return "portrait"


def _calculate_aspect_ratio(width: Optional[int], height: Optional[int]) -> Optional[float]:
    if not width or not height:
        return None

    if height == 0:
        return None

    return round(width / height, 4)


def _classify_aspect_category(
    width: Optional[int],
    height: Optional[int],
    aspect_ratio: Optional[float],
) -> str:
    if not width or not height or aspect_ratio is None:
        return "unknown"

    if width == height:
        return "square"

    if aspect_ratio < 0.55:
        return "tall_portrait"

    if aspect_ratio < 0.85:
        return "standard_portrait"

    if aspect_ratio < 1.15:
        return "near_square"

    if aspect_ratio < 1.45:
        return "standard_landscape"

    if aspect_ratio < 1.9:
        return "widescreen_landscape"

    return "ultrawide_landscape"


def _classify_resolution(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "unknown"

    total_pixels = width * height

    if total_pixels < 500_000:
        return "low_resolution"

    if total_pixels < 1_500_000:
        return "standard_resolution"

    if total_pixels < 4_000_000:
        return "high_resolution"

    return "ultra_resolution"


def _classify_brightness(value: Optional[float]) -> str:
    if value is None:
        return "unknown"

    if value < 35:
        return "very_dark"

    if value < 85:
        return "dim"

    if value < 165:
        return "balanced"

    if value < 225:
        return "bright"

    return "overexposed"


def _classify_color_profile(image_mode: Optional[str]) -> str:
    safe_mode = (image_mode or "").strip().upper()

    if not safe_mode:
        return "unknown"

    if safe_mode in {"1", "L", "LA"}:
        return "grayscale_or_luminance"

    if safe_mode in {"RGB", "RGBA", "P"}:
        return "color"

    if safe_mode in {"CMYK", "YCbCr", "HSV"}:
        return "alternate_color_space"

    return "unknown"


def _estimate_sharpness(grayscale_image: Any) -> tuple[Optional[float], str]:
    try:
        working_image = grayscale_image.copy()
        working_image.thumbnail((256, 256))

        width, height = working_image.size
        if width < 3 or height < 3:
            return None, "unknown"

        pixels = list(working_image.getdata())
        differences: list[float] = []

        for y in range(height):
            row_start = y * width
            for x in range(width - 1):
                current_value = pixels[row_start + x]
                next_value = pixels[row_start + x + 1]
                differences.append(abs(float(current_value) - float(next_value)))

        if not differences:
            return None, "unknown"

        mean_difference = sum(differences) / len(differences)
        variance = sum((value - mean_difference) ** 2 for value in differences) / len(differences)
        sharpness_estimate = round(float(variance), 2)

        if sharpness_estimate < 20:
            return sharpness_estimate, "blurry"

        if sharpness_estimate < 80:
            return sharpness_estimate, "soft"

        if sharpness_estimate < 220:
            return sharpness_estimate, "moderate_detail"

        return sharpness_estimate, "sharp"

    except Exception:
        return None, "unknown"


def _estimate_contrast(grayscale_stat: Any) -> tuple[Optional[float], str]:
    try:
        stddev = getattr(grayscale_stat, "stddev", None)
        if not stddev:
            return None, "unknown"

        contrast_estimate = round(float(stddev[0]), 2)

        if contrast_estimate < 25:
            return contrast_estimate, "low_contrast"

        if contrast_estimate < 75:
            return contrast_estimate, "balanced_contrast"

        return contrast_estimate, "high_contrast"

    except Exception:
        return None, "unknown"


def _estimate_image_cleanliness(
    image_format: Optional[str],
    size_bytes: int,
    width: Optional[int],
    height: Optional[int],
    sharpness_label: str,
    contrast_label: str,
) -> str:
    if not width or not height or size_bytes <= 0:
        return "unknown"

    total_pixels = width * height
    bytes_per_pixel = size_bytes / total_pixels if total_pixels else 0
    safe_format = (image_format or "").strip().upper()

    if sharpness_label == "blurry" and contrast_label == "low_contrast":
        return "soft_or_degraded"

    if safe_format in {"JPEG", "JPG"} and bytes_per_pixel < 0.25:
        return "likely_compressed"

    if bytes_per_pixel < 0.12:
        return "likely_compressed"

    if sharpness_label in {"sharp", "moderate_detail"} and contrast_label in {
        "balanced_contrast",
        "high_contrast",
    }:
        return "clean"

    return "acceptable"


def _classify_quality_assessment(
    resolution_category: str,
    brightness_label: str,
    sharpness_label: str,
    contrast_label: str,
    cleanliness_label: str,
) -> str:
    score = 0

    if resolution_category in {"high_resolution", "ultra_resolution"}:
        score += 2
    elif resolution_category == "standard_resolution":
        score += 1

    if brightness_label in {"balanced", "bright"}:
        score += 2

    if sharpness_label == "sharp":
        score += 2
    elif sharpness_label == "moderate_detail":
        score += 1

    if contrast_label in {"balanced_contrast", "high_contrast"}:
        score += 1

    if cleanliness_label == "clean":
        score += 2
    elif cleanliness_label == "acceptable":
        score += 1

    if score >= 8:
        return "excellent"

    if score >= 5:
        return "good"

    if score >= 3:
        return "acceptable"

    return "poor"


def _classify_ocr_readiness(
    brightness_label: str,
    sharpness_label: str,
    contrast_label: str,
    resolution_category: str,
) -> str:
    score = 0

    if brightness_label == "balanced":
        score += 1

    if sharpness_label in {"sharp", "moderate_detail"}:
        score += 1

    if contrast_label in {"balanced_contrast", "high_contrast"}:
        score += 1

    if resolution_category in {"high_resolution", "ultra_resolution"}:
        score += 1

    if score >= 4:
        return "high"

    if score >= 2:
        return "moderate"

    return "low"


def _classify_object_detection_readiness(
    sharpness_label: str,
    quality_assessment: str,
    cleanliness_label: str,
) -> str:
    score = 0

    if sharpness_label in {"sharp", "moderate_detail"}:
        score += 1

    if quality_assessment in {"excellent", "good"}:
        score += 1

    if cleanliness_label == "clean":
        score += 1

    if score >= 3:
        return "high"

    if score >= 2:
        return "moderate"

    return "low"


def _classify_model_input_readiness(
    resolution_category: str,
    quality_assessment: str,
    inspection_confidence: str,
) -> str:
    score = 0

    if resolution_category in {"high_resolution", "ultra_resolution"}:
        score += 1

    if quality_assessment in {"excellent", "good"}:
        score += 1

    if inspection_confidence == "high":
        score += 1

    if score >= 3:
        return "high"

    if score >= 2:
        return "moderate"

    return "low"


def _build_preprocessing_recommendation(
    brightness_label: str,
    sharpness_label: str,
    contrast_label: str,
    cleanliness_label: str,
) -> str:
    recommendations: list[str] = []

    if brightness_label in {"very_dark", "dim"}:
        recommendations.append("brightness_normalization")

    if brightness_label == "overexposed":
        recommendations.append("highlight_reduction")

    if sharpness_label in {"blurry", "soft"}:
        recommendations.append("sharpening_filter")

    if contrast_label == "low_contrast":
        recommendations.append("contrast_enhancement")

    if cleanliness_label in {"likely_compressed", "soft_or_degraded"}:
        recommendations.append("artifact_reduction")

    if not recommendations:
        return "no_preprocessing_recommended"

    return ", ".join(recommendations)


def _build_preprocessing_summary(
    ocr_readiness: str,
    object_detection_readiness: str,
    model_input_readiness: str,
    preprocessing_recommendation: str,
) -> str:
    return (
        "Deterministic preprocessing readiness assessment indicates "
        f"OCR readiness is {ocr_readiness}, object detection readiness is "
        f"{object_detection_readiness}, and general model input readiness is "
        f"{model_input_readiness}. Recommended preprocessing actions: "
        f"{preprocessing_recommendation}."
    )


def _estimate_inspection_confidence(
    image_format: Optional[str],
    image_mode: Optional[str],
    width: Optional[int],
    height: Optional[int],
    aspect_ratio: Optional[float],
    brightness_estimate: Optional[float],
    color_profile: str,
    sharpness_estimate: Optional[float],
    contrast_estimate: Optional[float],
    quality_assessment: str,
) -> str:
    score = 0

    if image_format:
        score += 1

    if image_mode:
        score += 1

    if width and height:
        score += 1

    if aspect_ratio is not None:
        score += 1

    if brightness_estimate is not None:
        score += 1

    if color_profile != "unknown":
        score += 1

    if sharpness_estimate is not None:
        score += 1

    if contrast_estimate is not None:
        score += 1

    if quality_assessment != "unknown":
        score += 1

    if score >= 8:
        return "high"

    if score >= 5:
        return "moderate"

    return "low"


def _build_local_inspection_summary(
    width: Optional[int],
    height: Optional[int],
    orientation: str,
    resolution_category: str,
    aspect_category: str,
    brightness_label: str,
    color_profile: str,
    image_mode: Optional[str],
    sharpness_label: str,
    contrast_label: str,
    cleanliness_label: str,
    quality_assessment: str,
) -> str:
    dimensions = "unknown dimensions"
    if width and height:
        dimensions = f"{width}x{height}"

    mode_text = image_mode or "unknown mode"

    return (
        "Local lightweight inspection completed. The uploaded image has "
        f"{dimensions} dimensions, {orientation} orientation, "
        f"{aspect_category} proportions, and is classified as "
        f"{resolution_category}. The image uses {mode_text} mode with a "
        f"{color_profile} profile and appears {brightness_label} by approximate "
        f"local luminance analysis. Technical quality assessment indicates "
        f"{sharpness_label} detail, {contrast_label}, {cleanliness_label} image "
        f"cleanliness, and an overall {quality_assessment} local quality rating."
    )


def _classify_environment_condition(brightness_label: str) -> str:
    """
    Phase 9I.5E deterministic environment-light classification.

    This is a technical lighting-condition label only. It is NOT semantic
    scene understanding and does not identify objects, people, or places.
    """
    if brightness_label in {"very_dark", "dim"}:
        return "low_light_environment"

    if brightness_label in {"bright", "overexposed"}:
        return "bright_environment"

    if brightness_label == "balanced":
        return "balanced_environment"

    return "unknown_environment"


def _estimate_scene_confidence(
    inspection_confidence: str,
    ocr_readiness: str,
    quality_assessment: str,
    ocr_execution_result: Optional[dict[str, Any]] = None,
) -> str:
    """
    Phase 9I.5E deterministic scene-condition confidence.

    This confidence only describes how much local technical metadata was
    available for the heuristic classification.
    """
    score = 0

    if inspection_confidence == "high":
        score += 2
    elif inspection_confidence == "moderate":
        score += 1

    if quality_assessment in {"excellent", "good"}:
        score += 2
    elif quality_assessment == "acceptable":
        score += 1

    if ocr_readiness in {"high", "moderate"}:
        score += 1

    if ocr_execution_result and bool(ocr_execution_result.get("ocr_execution_performed", False)):
        score += 1

    if score >= 5:
        return "high"

    if score >= 3:
        return "moderate"

    return "low"


def _classify_ocr_priority_recommendation(
    scene_classification: str,
    ocr_readiness: str,
    text_density: float,
    text_extracted: bool,
) -> str:
    """
    Phase 9I.5E deterministic OCR routing guidance.

    This only recommends OCR priority for future routing. OCR still requires
    explicit opt-in and never runs automatically.
    """
    if text_extracted and text_density >= 0.05:
        return "high"

    if scene_classification in {"text_heavy_document_like", "document_like", "screen_like"}:
        if ocr_readiness in {"high", "moderate"}:
            return "high"
        return "moderate"

    if scene_classification == "text_present_image":
        return "moderate"

    if scene_classification == "photo_like":
        return "low"

    if ocr_readiness == "high":
        return "moderate"

    if ocr_readiness == "moderate":
        return "low"

    return "not_recommended"


def _build_controlled_scene_summary(
    scene_classification: str,
    environment_condition: str,
    scene_confidence: str,
    ocr_priority_recommendation: str,
) -> str:
    return (
        "Phase 9I.5E controlled scene-condition classification completed. "
        f"The image is classified as {scene_classification} with "
        f"{scene_confidence} deterministic confidence. Lighting condition is "
        f"{environment_condition}. OCR routing priority recommendation is "
        f"{ocr_priority_recommendation}. This is technical image-condition "
        "classification only, not semantic scene understanding."
    )


def _classify_controlled_scene_condition(
    brightness_label: str,
    contrast_label: str,
    sharpness_label: str,
    cleanliness_label: str,
    aspect_category: str,
    color_profile: str,
    quality_assessment: str,
    ocr_readiness: str,
    inspection_confidence: str,
    ocr_execution_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E controlled deterministic scene-condition classification.

    This is intentionally limited to safe technical labels such as document-like,
    screen-like, photo-like, text-heavy, and lighting-condition labels.

    It does NOT:
    - perform object recognition
    - perform face recognition
    - infer identity
    - infer sensitive attributes
    - understand semantic scene content
    - call external AI vision models
    - persist image or OCR data
    """
    environment_condition = _classify_environment_condition(brightness_label)

    text_density = 0.0
    text_extracted = False
    word_count = 0

    if ocr_execution_result:
        try:
            text_density = float(ocr_execution_result.get("text_density", 0.0) or 0.0)
        except Exception:
            text_density = 0.0
        text_extracted = bool(ocr_execution_result.get("text_extracted", False))
        try:
            word_count = int(ocr_execution_result.get("word_count", 0) or 0)
        except Exception:
            word_count = 0

    is_landscape_screen_shape = aspect_category in {
        "standard_landscape",
        "widescreen_landscape",
        "ultrawide_landscape",
    }
    has_text_density = text_extracted and (text_density >= 0.05 or word_count >= 3)
    is_text_heavy = text_extracted and (text_density >= 0.15 or word_count >= 12)

    is_screen_like = (
        is_landscape_screen_shape
        and brightness_label in {"bright", "overexposed", "balanced"}
        and contrast_label in {"balanced_contrast", "high_contrast"}
        and cleanliness_label in {"clean", "acceptable"}
    )

    is_document_like = (
        ocr_readiness in {"high", "moderate"}
        and contrast_label in {"balanced_contrast", "high_contrast"}
        and sharpness_label in {"moderate_detail", "sharp"}
        and cleanliness_label in {"clean", "acceptable"}
    )

    if is_text_heavy and is_document_like:
        scene_classification = "text_heavy_document_like"
    elif is_text_heavy:
        scene_classification = "text_heavy_image"
    elif is_document_like:
        scene_classification = "document_like"
    elif is_screen_like:
        scene_classification = "screen_like"
    elif has_text_density:
        scene_classification = "text_present_image"
    elif color_profile == "color" and quality_assessment in {"excellent", "good", "acceptable"}:
        scene_classification = "photo_like"
    elif environment_condition == "low_light_environment":
        scene_classification = "low_light_image"
    elif environment_condition == "bright_environment":
        scene_classification = "bright_image"
    else:
        scene_classification = "general_image_condition"

    scene_confidence = _estimate_scene_confidence(
        inspection_confidence=inspection_confidence,
        ocr_readiness=ocr_readiness,
        quality_assessment=quality_assessment,
        ocr_execution_result=ocr_execution_result,
    )

    ocr_priority_recommendation = _classify_ocr_priority_recommendation(
        scene_classification=scene_classification,
        ocr_readiness=ocr_readiness,
        text_density=text_density,
        text_extracted=text_extracted,
    )

    scene_summary = _build_controlled_scene_summary(
        scene_classification=scene_classification,
        environment_condition=environment_condition,
        scene_confidence=scene_confidence,
        ocr_priority_recommendation=ocr_priority_recommendation,
    )

    return {
        "scene_classification": scene_classification,
        "scene_confidence": scene_confidence,
        "scene_summary": scene_summary,
        "ocr_priority_recommendation": ocr_priority_recommendation,
        "environment_condition": environment_condition,
    }


def _extract_ocr_coordination_signals(
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic OCR signal extraction for coordination.

    This only reads already-produced OCR metadata. It does NOT execute OCR,
    persist text, call external services, infer identity, or perform semantic
    scene understanding.
    """
    if not ocr_execution_result:
        return {
            "ocr_available_for_coordination": False,
            "ocr_execution_performed": False,
            "text_extracted": False,
            "text_density": 0.0,
            "word_count": 0,
            "line_count": 0,
            "ocr_readability_label": "not_available",
            "ocr_noise_label": "not_available",
            "ocr_preprocessing_profile": "not_available",
            "ocr_text_structure": "not_available",
        }

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or default)
        except Exception:
            return default

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value or default)
        except Exception:
            return default

    return {
        "ocr_available_for_coordination": True,
        "ocr_execution_performed": bool(ocr_execution_result.get("ocr_execution_performed", False)),
        "text_extracted": bool(ocr_execution_result.get("text_extracted", False)),
        "text_density": _safe_float(ocr_execution_result.get("text_density", 0.0)),
        "word_count": _safe_int(ocr_execution_result.get("word_count", 0)),
        "line_count": _safe_int(ocr_execution_result.get("line_count", 0)),
        "ocr_readability_label": str(ocr_execution_result.get("ocr_readability_label", "unknown")),
        "ocr_noise_label": str(ocr_execution_result.get("ocr_noise_label", "unknown")),
        "ocr_preprocessing_profile": str(
            ocr_execution_result.get("ocr_preprocessing_profile", "unknown")
        ),
        "ocr_text_structure": str(ocr_execution_result.get("ocr_text_structure", "unknown")),
    }


def _classify_scene_ocr_alignment(
    scene_classification: str,
    ocr_priority_recommendation: str,
    ocr_readiness: str,
    ocr_signals: dict[str, Any],
) -> str:
    """
    Phase 9I.5E deterministic scene/OCR alignment label.

    This coordinates technical labels only. It does NOT infer semantic content.
    """
    text_extracted = bool(ocr_signals.get("text_extracted", False))
    text_density = float(ocr_signals.get("text_density", 0.0) or 0.0)
    word_count = int(ocr_signals.get("word_count", 0) or 0)

    text_oriented_scene = scene_classification in {
        "text_heavy_document_like",
        "text_heavy_image",
        "document_like",
        "screen_like",
        "text_present_image",
    }

    if text_oriented_scene and text_extracted and (text_density >= 0.05 or word_count >= 3):
        return "strong_alignment"

    if text_oriented_scene and ocr_priority_recommendation in {"high", "moderate"}:
        return "probable_alignment"

    if scene_classification == "photo_like" and not text_extracted:
        return "ocr_not_aligned"

    if ocr_readiness == "low" and not text_extracted:
        return "weak_alignment"

    if text_extracted:
        return "partial_alignment"

    return "undetermined_alignment"


def _build_vision_routing_decision(
    scene_classification: str,
    ocr_priority_recommendation: str,
    scene_ocr_alignment: str,
    ocr_signals: dict[str, Any],
) -> str:
    """
    Phase 9I.5E deterministic routing decision.

    This is route guidance only. It does NOT automatically execute OCR, save
    data, call external models, or trigger autonomous behavior.
    """
    readability = str(ocr_signals.get("ocr_readability_label", "not_available"))
    noise = str(ocr_signals.get("ocr_noise_label", "not_available"))

    if (
        scene_ocr_alignment == "strong_alignment"
        and readability
        in {
            "high_readability",
            "moderate_readability",
        }
        and noise in {"low_noise", "moderate_noise", "none"}
    ):
        return "route_to_structured_ocr_review"

    if scene_classification in {"document_like", "text_heavy_document_like"}:
        return "route_to_document_ocr_path"

    if scene_classification == "screen_like":
        return "route_to_screen_ocr_path"

    if ocr_priority_recommendation == "high":
        return "route_to_ocr_first_path"

    if ocr_priority_recommendation == "moderate":
        return "route_to_optional_ocr_path"

    if scene_classification == "photo_like":
        return "route_to_visual_inspection_only"

    return "route_to_general_inspection_path"


def _build_ocr_action_recommendation(
    explicit_ocr_request: bool,
    ocr_priority_recommendation: str,
    scene_ocr_alignment: str,
    ocr_signals: dict[str, Any],
) -> str:
    """
    Phase 9I.5E deterministic OCR action recommendation.

    OCR remains explicit opt-in only. This function does not trigger OCR.
    """
    ocr_performed = bool(ocr_signals.get("ocr_execution_performed", False))
    text_extracted = bool(ocr_signals.get("text_extracted", False))

    if ocr_performed and text_extracted:
        return "ocr_completed_review_structured_text"

    if ocr_performed and not text_extracted:
        return "ocr_completed_no_text_detected"

    if not explicit_ocr_request and ocr_priority_recommendation == "high":
        return "ocr_recommended_requires_explicit_opt_in"

    if not explicit_ocr_request and ocr_priority_recommendation == "moderate":
        return "ocr_optional_requires_explicit_opt_in"

    if scene_ocr_alignment == "ocr_not_aligned":
        return "ocr_not_recommended_for_current_image_condition"

    return "continue_local_inspection_without_automatic_ocr"


def _estimate_coordination_confidence(
    scene_confidence: str,
    inspection_confidence: str,
    scene_ocr_alignment: str,
    ocr_signals: dict[str, Any],
) -> str:
    """
    Phase 9I.5E deterministic coordination confidence.

    This only reflects availability/consistency of technical metadata.
    """
    score = 0

    if scene_confidence == "high":
        score += 2
    elif scene_confidence == "moderate":
        score += 1

    if inspection_confidence == "high":
        score += 2
    elif inspection_confidence == "moderate":
        score += 1

    if scene_ocr_alignment in {"strong_alignment", "probable_alignment"}:
        score += 2
    elif scene_ocr_alignment in {"partial_alignment", "weak_alignment"}:
        score += 1

    if bool(ocr_signals.get("ocr_available_for_coordination", False)):
        score += 1

    if score >= 6:
        return "high"

    if score >= 3:
        return "moderate"

    return "low"


def _build_coordination_summary(
    vision_routing_decision: str,
    ocr_action_recommendation: str,
    scene_ocr_alignment: str,
    coordination_confidence: str,
) -> str:
    return (
        "Phase 9I.5E OCR and scene coordination completed. Deterministic routing "
        f"decision: {vision_routing_decision}. OCR action recommendation: "
        f"{ocr_action_recommendation}. Scene/OCR alignment: "
        f"{scene_ocr_alignment}. Coordination confidence: "
        f"{coordination_confidence}. This is rules-based local coordination only, "
        "not autonomous multimodal reasoning or semantic scene understanding."
    )


def _build_ocr_scene_coordination_metadata(
    scene_classification: str,
    scene_confidence: str,
    ocr_priority_recommendation: str,
    ocr_readiness: str,
    inspection_confidence: str,
    explicit_ocr_request: bool,
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic OCR/scene coordination metadata builder.

    This unifies inspection, scene-condition classification, and OCR metadata
    into routing guidance only. It does NOT execute OCR, persist data, perform
    semantic scene understanding, infer identity, or call external AI models.
    """
    ocr_signals = _extract_ocr_coordination_signals(ocr_execution_result)

    scene_ocr_alignment = _classify_scene_ocr_alignment(
        scene_classification=scene_classification,
        ocr_priority_recommendation=ocr_priority_recommendation,
        ocr_readiness=ocr_readiness,
        ocr_signals=ocr_signals,
    )
    vision_routing_decision = _build_vision_routing_decision(
        scene_classification=scene_classification,
        ocr_priority_recommendation=ocr_priority_recommendation,
        scene_ocr_alignment=scene_ocr_alignment,
        ocr_signals=ocr_signals,
    )
    ocr_action_recommendation = _build_ocr_action_recommendation(
        explicit_ocr_request=explicit_ocr_request,
        ocr_priority_recommendation=ocr_priority_recommendation,
        scene_ocr_alignment=scene_ocr_alignment,
        ocr_signals=ocr_signals,
    )
    coordination_confidence = _estimate_coordination_confidence(
        scene_confidence=scene_confidence,
        inspection_confidence=inspection_confidence,
        scene_ocr_alignment=scene_ocr_alignment,
        ocr_signals=ocr_signals,
    )
    coordination_summary = _build_coordination_summary(
        vision_routing_decision=vision_routing_decision,
        ocr_action_recommendation=ocr_action_recommendation,
        scene_ocr_alignment=scene_ocr_alignment,
        coordination_confidence=coordination_confidence,
    )

    return {
        "coordination_mode": "deterministic_ocr_scene_coordination",
        "vision_routing_decision": vision_routing_decision,
        "ocr_action_recommendation": ocr_action_recommendation,
        "scene_ocr_alignment": scene_ocr_alignment,
        "coordination_confidence": coordination_confidence,
        "coordination_summary": coordination_summary,
    }


def _score_scene_profile_readability(
    ocr_readiness: str,
    ocr_execution_result: Optional[dict[str, Any]],
    quality_assessment: str,
    sharpness_label: str,
    contrast_label: str,
) -> tuple[float, str]:
    """
    Phase 10B deterministic scene-profile readability scoring.

    This uses already-computed local technical metadata only. It does NOT run
    OCR, call an AI model, infer semantic content, identify objects, recognize
    faces, persist data, or trigger autonomous behavior.
    """
    if ocr_execution_result:
        try:
            score = float(ocr_execution_result.get("ocr_readability_score", 0.0) or 0.0)
        except Exception:
            score = 0.0

        if score > 1.0:
            score = score / 100.0

        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            return round(score, 3), "high"
        if score >= 0.4:
            return round(score, 3), "moderate"
        if bool(ocr_execution_result.get("text_extracted", False)):
            return max(round(score, 3), 0.35), "moderate"
        return round(score, 3), "low"

    score = 0.0

    if ocr_readiness == "high":
        score += 0.4
    elif ocr_readiness == "moderate":
        score += 0.25

    if quality_assessment in {"excellent", "good"}:
        score += 0.25
    elif quality_assessment == "acceptable":
        score += 0.15

    if sharpness_label in {"sharp", "moderate_detail"}:
        score += 0.2

    if contrast_label in {"balanced_contrast", "high_contrast"}:
        score += 0.15

    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        return round(score, 3), "high"
    if score >= 0.4:
        return round(score, 3), "moderate"
    return round(score, 3), "low"


def _score_scene_profile_framing(
    width: Optional[int],
    height: Optional[int],
    orientation: str,
    aspect_category: str,
    resolution_category: str,
    sharpness_label: str,
    cleanliness_label: str,
) -> tuple[float, str]:
    """
    Phase 10B deterministic scene-profile framing quality scoring.

    This is technical metadata packaging only. It does NOT perform object
    detection, identity inference, face recognition, semantic scene
    understanding, persistence, streaming, or autonomous analysis.
    """
    score = 0.0

    if width and height:
        score += 0.2

    if resolution_category in {"high_resolution", "ultra_resolution"}:
        score += 0.3
    elif resolution_category == "standard_resolution":
        score += 0.2

    if orientation in {"landscape", "portrait", "square"}:
        score += 0.1

    if aspect_category not in {"unknown"}:
        score += 0.1

    if sharpness_label == "sharp":
        score += 0.2
    elif sharpness_label == "moderate_detail":
        score += 0.15
    elif sharpness_label == "soft":
        score += 0.05

    if cleanliness_label == "clean":
        score += 0.1
    elif cleanliness_label == "acceptable":
        score += 0.05

    score = max(0.0, min(1.0, score))

    if score >= 0.75:
        return round(score, 3), "good"
    if score >= 0.4:
        return round(score, 3), "acceptable"
    return round(score, 3), "poor"


def _classify_scene_profile_ocr_confidence_band(
    ocr_execution_result: Optional[dict[str, Any]],
    ocr_readiness: str,
) -> str:
    """
    Phase 10B deterministic OCR confidence banding for scene profiles.

    This reads existing OCR metadata only and does not execute OCR.
    """
    if not ocr_execution_result:
        if ocr_readiness == "high":
            return "moderate"
        if ocr_readiness == "moderate":
            return "low"
        return "none"

    text_extracted = bool(ocr_execution_result.get("text_extracted", False))
    confidence_label = str(ocr_execution_result.get("ocr_confidence_label", "")).lower()
    readability_label = str(ocr_execution_result.get("ocr_readability_label", "")).lower()
    selected_ocr_pass = str(ocr_execution_result.get("selected_ocr_pass", "") or "")

    if text_extracted and (
        "high" in confidence_label or "high" in readability_label or selected_ocr_pass.strip()
    ):
        return "high"

    if text_extracted:
        return "moderate"

    if bool(ocr_execution_result.get("ocr_execution_performed", False)):
        return "low"

    return "none"


def _build_scene_profile_recommended_next_action(
    vision_response_recommended_next_action: str,
    readability_label: str,
    framing_quality_label: str,
    ocr_confidence_band: str,
    text_extracted: bool,
) -> str:
    """
    Phase 10B deterministic scene-profile next-action selection.

    This provides guidance only. It does not trigger OCR, camera capture, voice,
    persistence, streaming, or autonomous behavior.
    """
    if text_extracted and ocr_confidence_band in {"moderate", "high"}:
        return "ready_for_narration"

    if framing_quality_label == "poor":
        return "retry_framing"

    if readability_label in {"moderate", "high"} and ocr_confidence_band in {
        "low",
        "moderate",
        "high",
    }:
        return "use_ocr"

    if vision_response_recommended_next_action in {
        "retry_with_clearer_text_or_improved_lighting_if_text_is_expected",
        "use_explicit_ocr_path_for_text_focused_followup",
        "rerun_with_explicit_ocr_request_true_if_text_extraction_is_desired",
    }:
        return "use_ocr"

    return "continue_inspection"


def _build_scene_profile(
    scene_classification: str,
    brightness_label: str,
    width: Optional[int],
    height: Optional[int],
    orientation: str,
    aspect_category: str,
    resolution_category: str,
    sharpness_label: str,
    contrast_label: str,
    cleanliness_label: str,
    quality_assessment: str,
    ocr_readiness: str,
    vision_response_recommended_next_action: str,
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 10B structured deterministic scene profile builder.

    This packages existing technical metadata into a cleaner scene_profile
    object. It does NOT add semantic scene understanding, object recognition,
    face recognition, identity inference, persistence, streaming, autonomous
    capture, automatic OCR, or autonomous narration.
    """
    readability_score, readability_label = _score_scene_profile_readability(
        ocr_readiness=ocr_readiness,
        ocr_execution_result=ocr_execution_result,
        quality_assessment=quality_assessment,
        sharpness_label=sharpness_label,
        contrast_label=contrast_label,
    )
    framing_quality_score, framing_quality_label = _score_scene_profile_framing(
        width=width,
        height=height,
        orientation=orientation,
        aspect_category=aspect_category,
        resolution_category=resolution_category,
        sharpness_label=sharpness_label,
        cleanliness_label=cleanliness_label,
    )
    ocr_confidence_band = _classify_scene_profile_ocr_confidence_band(
        ocr_execution_result=ocr_execution_result,
        ocr_readiness=ocr_readiness,
    )
    text_extracted = bool(
        ocr_execution_result and ocr_execution_result.get("text_extracted", False)
    )
    recommended_next_action = _build_scene_profile_recommended_next_action(
        vision_response_recommended_next_action=vision_response_recommended_next_action,
        readability_label=readability_label,
        framing_quality_label=framing_quality_label,
        ocr_confidence_band=ocr_confidence_band,
        text_extracted=text_extracted,
    )

    return {
        "scene_profile_version": "10B",
        "scene_profile_available": True,
        "scene_type": scene_classification,
        "readability_score": readability_score,
        "readability_label": readability_label,
        "framing_quality_score": framing_quality_score,
        "framing_quality_label": framing_quality_label,
        "lighting_condition": brightness_label,
        "ocr_confidence_band": ocr_confidence_band,
        "recommended_next_action": recommended_next_action,
        "source_phase": "Phase 10B",
        "source_mode": "deterministic_scene_profile_packaging",
        "semantic_scene_understanding_enabled": False,
        "object_recognition_enabled": False,
        "face_recognition_enabled": False,
        "identity_inference_enabled": False,
        "automatic_ocr_enabled": False,
        "autonomous_narration_enabled": False,
        "persistence_enabled": False,
        "streaming_enabled": False,
        "safety_note": "deterministic_local_scene_profile_only",
    }


def _classify_narration_plan_priority(
    scene_profile: dict[str, Any],
    text_extracted: bool,
    coordination_confidence: str,
) -> str:
    """
    Phase 10C deterministic narration-priority classification.

    This reads existing technical metadata only. It does NOT invoke TTS,
    speak automatically, stream audio, persist data, call external services,
    or perform semantic scene understanding.
    """
    recommended_next_action = str(scene_profile.get("recommended_next_action", ""))
    readability_label = str(scene_profile.get("readability_label", "low"))
    ocr_confidence_band = str(scene_profile.get("ocr_confidence_band", "none"))

    if text_extracted and recommended_next_action == "ready_for_narration":
        return "high"

    if ocr_confidence_band in {"moderate", "high"} and readability_label in {
        "moderate",
        "high",
    }:
        return "high"

    if recommended_next_action in {"retry_framing", "use_ocr"}:
        return "moderate"

    if coordination_confidence == "high":
        return "moderate"

    return "low"


def _classify_narration_plan_style(
    scene_profile: dict[str, Any],
    text_extracted: bool,
) -> str:
    """
    Phase 10C deterministic narration-style selection.

    This only selects a safe style label for future explicit narration. It does
    NOT execute voice output or modify the voice runtime.
    """
    recommended_next_action = str(scene_profile.get("recommended_next_action", ""))
    readability_label = str(scene_profile.get("readability_label", "low"))
    framing_quality_label = str(scene_profile.get("framing_quality_label", "poor"))
    lighting_condition = str(scene_profile.get("lighting_condition", "unknown"))

    if text_extracted and recommended_next_action == "ready_for_narration":
        return "calm_informative"

    if framing_quality_label == "poor" or lighting_condition in {
        "very_dark",
        "dim",
        "overexposed",
    }:
        return "careful_retry_guidance"

    if readability_label == "high":
        return "concise_confirmation"

    return "calm_informative"


def _classify_narration_plan_pacing(
    scene_profile: dict[str, Any],
    narration_style: str,
) -> str:
    """
    Phase 10C deterministic narration-pacing selection.

    This is metadata only. It does not invoke pyttsx3, Edge-TTS, websocket
    audio, or any autonomous speech process.
    """
    readability_label = str(scene_profile.get("readability_label", "low"))
    ocr_confidence_band = str(scene_profile.get("ocr_confidence_band", "none"))

    if narration_style == "careful_retry_guidance":
        return "clear_slow"

    if readability_label == "low" or ocr_confidence_band in {"none", "low"}:
        return "slow"

    return "balanced"


def _build_narration_plan_reason(
    scene_profile: dict[str, Any],
    narration_priority: str,
    narration_style: str,
    recommended_pacing: str,
) -> str:
    """
    Phase 10C deterministic narration-plan explanation.

    This explanation is generated from local metadata only and does not infer
    semantic content from the image.
    """
    scene_type = str(scene_profile.get("scene_type", "unknown"))
    readability_label = str(scene_profile.get("readability_label", "unknown"))
    framing_quality_label = str(scene_profile.get("framing_quality_label", "unknown"))
    ocr_confidence_band = str(scene_profile.get("ocr_confidence_band", "unknown"))
    recommended_next_action = str(scene_profile.get("recommended_next_action", "unknown"))

    return (
        "Phase 10C deterministic narration planning completed from the existing "
        f"scene profile. Scene type: {scene_type}; readability: {readability_label}; "
        f"framing quality: {framing_quality_label}; OCR confidence band: "
        f"{ocr_confidence_band}; scene-profile next action: "
        f"{recommended_next_action}; selected narration priority: "
        f"{narration_priority}; selected narration style: {narration_style}; "
        f"recommended pacing: {recommended_pacing}. No autonomous speech, live "
        "monitoring, streaming, persistence, semantic scene understanding, object "
        "recognition, face recognition, or identity inference was enabled."
    )


def _build_narration_plan(
    scene_profile: dict[str, Any],
    coordination_confidence: str,
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 10C deterministic narration planning metadata builder.

    This creates a future voice-delivery planning object from existing scene
    profile and OCR metadata only. It does NOT invoke TTS, execute voice output,
    stream audio, persist content, modify the voice service, monitor camera
    input, or perform semantic AI vision reasoning.
    """
    text_extracted = bool(
        ocr_execution_result and ocr_execution_result.get("text_extracted", False)
    )
    narration_priority = _classify_narration_plan_priority(
        scene_profile=scene_profile,
        text_extracted=text_extracted,
        coordination_confidence=coordination_confidence,
    )
    narration_style = _classify_narration_plan_style(
        scene_profile=scene_profile,
        text_extracted=text_extracted,
    )
    recommended_pacing = _classify_narration_plan_pacing(
        scene_profile=scene_profile,
        narration_style=narration_style,
    )
    narration_reason = _build_narration_plan_reason(
        scene_profile=scene_profile,
        narration_priority=narration_priority,
        narration_style=narration_style,
        recommended_pacing=recommended_pacing,
    )

    return {
        "narration_plan_version": "10C",
        "narration_plan_available": True,
        "narration_priority": narration_priority,
        "narration_style": narration_style,
        "recommended_pacing": recommended_pacing,
        "recommended_delivery_profile": "vision_narration_calm",
        "recommended_pause_profile": (
            "standard_clear_pauses" if recommended_pacing == "balanced" else "slower_clarity_pauses"
        ),
        "should_offer_voice": bool(narration_priority in {"moderate", "high"}),
        "voice_execution_requires_explicit_request": True,
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "voice_stack_modified": False,
        "source_phase": "Phase 10C",
        "source_mode": "deterministic_narration_planning",
        "narration_reason": narration_reason,
        "safety_note": "deterministic_local_narration_plan_only",
    }


def _classify_narration_readiness_visual_clarity(
    readability_label: str,
    framing_quality_label: str,
) -> str:
    """
    Phase 10D deterministic visual-clarity label for narration readiness.

    This reads existing scene-profile metadata only. It does NOT analyze images,
    run OCR, invoke TTS, perform semantic scene understanding, persist data,
    stream video/audio, or trigger autonomous behavior.
    """
    if readability_label == "high" and framing_quality_label == "good":
        return "high"

    if readability_label in {"moderate", "high"} and framing_quality_label in {
        "acceptable",
        "good",
    }:
        return "moderate"

    return "low"


def _classify_narration_readiness_lighting_support(lighting_condition: str) -> str:
    """
    Phase 10D deterministic lighting support label for narration readiness.

    This is technical metadata packaging only and does not infer scene content.
    """
    if lighting_condition in {"balanced", "bright"}:
        return "good"

    if lighting_condition in {"dim", "very_dark", "overexposed"}:
        return "poor"

    return "moderate"


def _classify_narration_readiness_length(
    narration_ready: bool,
    narration_priority: str,
    text_extracted: bool,
    visual_clarity: str,
) -> str:
    """
    Phase 10D deterministic narration-length guidance.

    This only recommends future explicit narration length. It does not speak or
    modify any voice route.
    """
    if not narration_ready:
        return "short"

    if (
        text_extracted
        and narration_priority == "high"
        and visual_clarity
        in {
            "moderate",
            "high",
        }
    ):
        return "detailed"

    if narration_priority in {"moderate", "high"}:
        return "medium"

    return "short"


def _build_narration_readiness_reason(
    readiness_score: float,
    narration_ready: bool,
    recommended_narration_length: str,
    ocr_priority: str,
    visual_clarity: str,
    lighting_support: str,
    scene_next_action: str,
) -> str:
    """
    Phase 10D deterministic narration-readiness explanation.

    This explanation is generated from local metadata only.
    """
    return (
        "Phase 10D deterministic narration readiness scoring completed from "
        "existing scene_profile, narration_plan, and OCR metadata. "
        f"Readiness score: {readiness_score}; narration ready: {narration_ready}; "
        f"recommended narration length: {recommended_narration_length}; OCR priority: "
        f"{ocr_priority}; visual clarity: {visual_clarity}; lighting support: "
        f"{lighting_support}; scene-profile next action: {scene_next_action}. "
        "No autonomous narration, TTS invocation, live monitoring, streaming, "
        "persistence, semantic scene understanding, object recognition, face "
        "recognition, or identity inference was enabled."
    )


def _build_narration_readiness(
    scene_profile: dict[str, Any],
    narration_plan: dict[str, Any],
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 10D deterministic narration readiness scoring.

    This evaluates whether the current deterministic inspection result is ready
    for a future explicit voice handoff. It does NOT invoke voice output, call
    pyttsx3, call Edge-TTS, stream audio, save audio, persist OCR/image content,
    modify routes, monitor camera input, or perform semantic AI vision reasoning.
    """

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except Exception:
            return default

    readability_score = _safe_float(scene_profile.get("readability_score", 0.0))
    framing_quality_score = _safe_float(scene_profile.get("framing_quality_score", 0.0))
    readability_label = str(scene_profile.get("readability_label", "low"))
    framing_quality_label = str(scene_profile.get("framing_quality_label", "poor"))
    lighting_condition = str(scene_profile.get("lighting_condition", "unknown"))
    ocr_confidence_band = str(scene_profile.get("ocr_confidence_band", "none"))
    scene_next_action = str(scene_profile.get("recommended_next_action", "continue_inspection"))
    narration_priority = str(narration_plan.get("narration_priority", "low"))
    should_offer_voice = bool(narration_plan.get("should_offer_voice", False))
    text_extracted = bool(
        ocr_execution_result and ocr_execution_result.get("text_extracted", False)
    )

    ocr_score = {
        "none": 0.0,
        "low": 0.25,
        "moderate": 0.55,
        "high": 0.8,
    }.get(ocr_confidence_band, 0.0)
    priority_score = {
        "low": 0.15,
        "moderate": 0.45,
        "high": 0.7,
    }.get(narration_priority, 0.15)

    readiness_score = (
        readability_score * 0.3
        + framing_quality_score * 0.2
        + ocr_score * 0.25
        + priority_score * 0.15
        + (0.1 if text_extracted else 0.0)
    )
    readiness_score = round(max(0.0, min(1.0, readiness_score)), 3)

    visual_clarity = _classify_narration_readiness_visual_clarity(
        readability_label=readability_label,
        framing_quality_label=framing_quality_label,
    )
    lighting_support = _classify_narration_readiness_lighting_support(
        lighting_condition=lighting_condition,
    )

    narration_ready = bool(
        (scene_next_action == "ready_for_narration" and should_offer_voice)
        or (text_extracted and readiness_score >= 0.55)
    )

    if text_extracted and ocr_confidence_band in {"moderate", "high"}:
        ocr_priority = "high"
    elif scene_next_action == "use_ocr":
        ocr_priority = "moderate"
    elif ocr_confidence_band == "low":
        ocr_priority = "low"
    else:
        ocr_priority = "none"

    recommended_narration_length = _classify_narration_readiness_length(
        narration_ready=narration_ready,
        narration_priority=narration_priority,
        text_extracted=text_extracted,
        visual_clarity=visual_clarity,
    )

    readiness_reason = _build_narration_readiness_reason(
        readiness_score=readiness_score,
        narration_ready=narration_ready,
        recommended_narration_length=recommended_narration_length,
        ocr_priority=ocr_priority,
        visual_clarity=visual_clarity,
        lighting_support=lighting_support,
        scene_next_action=scene_next_action,
    )

    return {
        "readiness_version": "10D",
        "narration_readiness_available": True,
        "narration_ready": narration_ready,
        "readiness_score": readiness_score,
        "readiness_label": (
            "high" if readiness_score >= 0.75 else "moderate" if readiness_score >= 0.45 else "low"
        ),
        "recommended_narration_length": recommended_narration_length,
        "ocr_priority": ocr_priority,
        "visual_clarity": visual_clarity,
        "lighting_support": lighting_support,
        "safe_for_voice_handoff": bool(narration_ready and should_offer_voice),
        "voice_execution_requires_explicit_request": True,
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "voice_stack_modified": False,
        "semantic_understanding_enabled": False,
        "object_recognition_enabled": False,
        "face_recognition_enabled": False,
        "identity_inference_enabled": False,
        "automatic_ocr_enabled": False,
        "autonomous_narration_enabled": False,
        "persistence_enabled": False,
        "streaming_enabled": False,
        "source_phase": "Phase 10D",
        "source_mode": "deterministic_narration_readiness_scoring",
        "readiness_reason": readiness_reason,
        "safety_note": "deterministic_local_narration_readiness_only",
    }


def _build_voice_bridge_refinement_metadata(
    narration_plan: dict[str, Any],
    narration_readiness: dict[str, Any],
    vision_voice_priority: str,
    vision_voice_payload_ready: bool,
    voice_handoff_ready: bool,
    voice_handoff_safe: bool,
    voice_execution_available: bool,
) -> dict[str, Any]:
    """
    Phase 10E controlled narration execution bridge refinement metadata.

    This strengthens the deterministic bridge between scene_profile,
    narration_plan, narration_readiness, voice_handoff_payload, and the future
    explicit /v1/voice/vision_execute route call. It does NOT invoke TTS,
    speak automatically, stream audio, persist audio/text/image content, modify
    voice routes, or enable autonomous narration.
    """
    readiness_score = narration_readiness.get("readiness_score", 0.0)
    readiness_label = str(narration_readiness.get("readiness_label", "unknown"))
    recommended_narration_length = str(
        narration_readiness.get("recommended_narration_length", "short")
    )
    recommended_delivery_profile = str(
        narration_plan.get("recommended_delivery_profile", "vision_narration_calm")
    )
    recommended_pause_profile = str(
        narration_plan.get("recommended_pause_profile", "standard_clear_pauses")
    )
    safe_for_voice_handoff = bool(
        narration_readiness.get("safe_for_voice_handoff", False)
        and voice_handoff_safe
        and voice_handoff_ready
    )

    return {
        "bridge_refinement_version": "10E",
        "bridge_refinement_available": True,
        "bridge_mode": "controlled_explicit_narration_execution_bridge",
        "readiness_score": readiness_score,
        "readiness_label": readiness_label,
        "recommended_narration_length": recommended_narration_length,
        "recommended_delivery_profile": recommended_delivery_profile,
        "recommended_pause_profile": recommended_pause_profile,
        "vision_voice_priority": vision_voice_priority,
        "vision_voice_payload_ready": bool(vision_voice_payload_ready),
        "voice_handoff_ready": bool(voice_handoff_ready),
        "voice_handoff_safe": bool(voice_handoff_safe),
        "voice_execution_available": bool(voice_execution_available),
        "safe_for_voice_handoff": safe_for_voice_handoff,
        "explicit_voice_execution_required": True,
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "voice_stack_modified": False,
        "route_called": False,
        "autonomous_narration_enabled": False,
        "semantic_scene_understanding_enabled": False,
        "source_phase": "Phase 10E",
        "source_mode": "deterministic_voice_bridge_refinement_metadata",
        "bridge_summary": (
            "Phase 10E refined the controlled narration execution bridge by "
            "carrying narration readiness, delivery profile, pause profile, "
            "and explicit voice execution safety metadata forward into the "
            "voice handoff contract. No speech, audio generation, streaming, "
            "persistence, autonomous narration, route call, or voice stack "
            "modification occurred."
        ),
        "safety_note": "deterministic_controlled_voice_bridge_refinement_only",
    }


def _build_vision_response_summary(
    scene_classification: str,
    scene_ocr_alignment: str,
    vision_routing_decision: str,
    ocr_action_recommendation: str,
) -> str:
    """
    Phase 9I.5E deterministic vision response summary builder.

    This produces a concise human-readable summary from existing deterministic
    metadata only. It does NOT call an LLM, infer semantic content, persist data,
    or perform autonomous reasoning.
    """
    if ocr_action_recommendation == "ocr_completed_review_structured_text":
        return (
            "The image was inspected locally, OCR completed successfully, and "
            f"the image condition is classified as {scene_classification}. "
            f"Routing decision: {vision_routing_decision}."
        )

    if ocr_action_recommendation == "ocr_completed_no_text_detected":
        return (
            "The image was inspected locally and OCR ran, but no readable text "
            f"was confidently extracted. The image condition is classified as "
            f"{scene_classification}."
        )

    if ocr_action_recommendation in {
        "ocr_recommended_requires_explicit_opt_in",
        "ocr_optional_requires_explicit_opt_in",
    }:
        return (
            "The image was inspected locally and appears compatible with OCR "
            f"routing guidance. Scene/OCR alignment is {scene_ocr_alignment}, "
            "but OCR still requires explicit opt-in."
        )

    if vision_routing_decision == "route_to_visual_inspection_only":
        return (
            "The image was inspected locally and current deterministic routing "
            "favors visual inspection only rather than OCR-first processing."
        )

    return (
        "The image was inspected locally and deterministic vision synthesis "
        f"completed. Routing decision: {vision_routing_decision}."
    )


def _build_vision_response_detail(
    scene_classification: str,
    scene_confidence: str,
    ocr_priority_recommendation: str,
    vision_routing_decision: str,
    ocr_action_recommendation: str,
    scene_ocr_alignment: str,
    coordination_confidence: str,
    ocr_execution_result: Optional[dict[str, Any]],
) -> str:
    """
    Phase 9I.5E deterministic detail synthesis.

    This is template-driven and uses only already-computed local metadata.
    """
    readability = "not_available"
    noise = "not_available"
    text_structure = "not_available"
    text_density = 0.0

    if ocr_execution_result:
        readability = str(ocr_execution_result.get("ocr_readability_label", "unknown"))
        noise = str(ocr_execution_result.get("ocr_noise_label", "unknown"))
        text_structure = str(ocr_execution_result.get("ocr_text_structure", "unknown"))
        try:
            text_density = float(ocr_execution_result.get("text_density", 0.0) or 0.0)
        except Exception:
            text_density = 0.0

    return (
        "Phase 9I.5E deterministic vision response synthesis completed. "
        f"Scene-condition label: {scene_classification}; scene confidence: "
        f"{scene_confidence}; OCR priority recommendation: "
        f"{ocr_priority_recommendation}; routing decision: "
        f"{vision_routing_decision}; OCR action recommendation: "
        f"{ocr_action_recommendation}; scene/OCR alignment: "
        f"{scene_ocr_alignment}; coordination confidence: "
        f"{coordination_confidence}; OCR readability: {readability}; OCR noise: "
        f"{noise}; OCR text structure: {text_structure}; OCR text density: "
        f"{text_density}. This synthesis is generated from deterministic local "
        "metadata only."
    )


def _build_vision_response_recommended_next_action(
    ocr_action_recommendation: str,
    vision_routing_decision: str,
    text_extracted: bool,
) -> str:
    """
    Phase 9I.5E deterministic next-action recommendation.

    This is user-facing guidance only. It does NOT trigger actions.
    """
    if text_extracted:
        return "review_structured_ocr_text"

    if ocr_action_recommendation == "ocr_completed_no_text_detected":
        return "retry_with_clearer_text_or_improved_lighting_if_text_is_expected"

    if ocr_action_recommendation == "ocr_recommended_requires_explicit_opt_in":
        return "rerun_with_explicit_ocr_request_true_if_text_extraction_is_desired"

    if ocr_action_recommendation == "ocr_optional_requires_explicit_opt_in":
        return "optional_rerun_with_explicit_ocr_request_true"

    if vision_routing_decision == "route_to_visual_inspection_only":
        return "continue_visual_inspection_without_ocr"

    if vision_routing_decision in {
        "route_to_document_ocr_path",
        "route_to_screen_ocr_path",
        "route_to_ocr_first_path",
    }:
        return "use_explicit_ocr_path_for_text_focused_followup"

    return "continue_general_local_inspection"


def _build_vision_response_safety_note() -> str:
    return (
        "Safety preserved: this response synthesis is deterministic and local-only. "
        "No image was persisted, no live monitoring or streaming was enabled, no "
        "face recognition or identity inference was performed, no semantic scene "
        "understanding was introduced, and no external AI vision model was called."
    )


def _build_vision_response_synthesis_metadata(
    scene_classification: str,
    scene_confidence: str,
    ocr_priority_recommendation: str,
    vision_routing_decision: str,
    ocr_action_recommendation: str,
    scene_ocr_alignment: str,
    coordination_confidence: str,
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic human-readable vision response synthesis.

    This unifies existing deterministic outputs into a user-facing response
    layer. It does NOT call an LLM, infer semantic scene content, persist data,
    monitor video, perform identity inference, or trigger autonomous behavior.
    """
    text_extracted = bool(
        ocr_execution_result and ocr_execution_result.get("text_extracted", False)
    )

    vision_response_summary = _build_vision_response_summary(
        scene_classification=scene_classification,
        scene_ocr_alignment=scene_ocr_alignment,
        vision_routing_decision=vision_routing_decision,
        ocr_action_recommendation=ocr_action_recommendation,
    )
    vision_response_detail = _build_vision_response_detail(
        scene_classification=scene_classification,
        scene_confidence=scene_confidence,
        ocr_priority_recommendation=ocr_priority_recommendation,
        vision_routing_decision=vision_routing_decision,
        ocr_action_recommendation=ocr_action_recommendation,
        scene_ocr_alignment=scene_ocr_alignment,
        coordination_confidence=coordination_confidence,
        ocr_execution_result=ocr_execution_result,
    )
    vision_response_recommended_next_action = _build_vision_response_recommended_next_action(
        ocr_action_recommendation=ocr_action_recommendation,
        vision_routing_decision=vision_routing_decision,
        text_extracted=text_extracted,
    )
    vision_response_safety_note = _build_vision_response_safety_note()

    return {
        "vision_response_summary": vision_response_summary,
        "vision_response_detail": vision_response_detail,
        "vision_response_recommended_next_action": vision_response_recommended_next_action,
        "vision_response_safety_note": vision_response_safety_note,
    }


def _normalize_voice_narration_text(text: str, max_length: int = 700) -> str:
    """
    Phase 9I.5E deterministic voice narration normalizer.

    This prepares already-generated deterministic vision text for future voice
    readback. It does NOT invoke TTS, speak automatically, stream audio, persist
    content, or call external services.
    """
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return "No vision narration is available for this inspection result."

    safe_max_length = max(80, int(max_length))
    if len(cleaned) <= safe_max_length:
        return cleaned

    truncated = cleaned[: safe_max_length - 3].rstrip()
    return f"{truncated}..."


def _classify_vision_voice_priority(
    vision_response_recommended_next_action: str,
    ocr_action_recommendation: str,
    text_extracted: bool,
    coordination_confidence: str,
) -> str:
    """
    Phase 9I.5E deterministic voice priority label.

    This label only helps future voice layers decide how important the narration
    payload is. It does NOT trigger speech.
    """
    if text_extracted:
        return "high"

    if ocr_action_recommendation == "ocr_completed_no_text_detected":
        return "moderate"

    if vision_response_recommended_next_action in {
        "retry_with_clearer_text_or_improved_lighting_if_text_is_expected",
        "use_explicit_ocr_path_for_text_focused_followup",
        "rerun_with_explicit_ocr_request_true_if_text_extraction_is_desired",
    }:
        return "moderate"

    if coordination_confidence == "high":
        return "moderate"

    return "low"


def _build_vision_voice_summary(
    vision_response_summary: str,
    vision_voice_priority: str,
) -> str:
    return (
        "Phase 9I.5E vision-to-voice coordination prepared a deterministic "
        f"voice-ready summary with {vision_voice_priority} narration priority. "
        f"Summary: {_normalize_voice_narration_text(vision_response_summary, max_length=260)}"
    )


def _build_vision_voice_narration(
    vision_response_summary: str,
    vision_response_recommended_next_action: str,
    ocr_action_recommendation: str,
    scene_classification: str,
) -> str:
    """
    Phase 9I.5E deterministic narration payload builder.

    This creates speech-friendly text only. It does NOT call pyttsx3, generate
    audio, trigger websocket streaming, or change the voice service.
    """
    base = _normalize_voice_narration_text(vision_response_summary, max_length=420)

    if ocr_action_recommendation == "ocr_completed_no_text_detected":
        followup = (
            "OCR was requested and completed, but no readable text was confidently "
            "extracted. A clearer view, stronger lighting, or closer framing may help."
        )
    elif vision_response_recommended_next_action == "review_structured_ocr_text":
        followup = "Structured OCR text is available for review."
    elif vision_response_recommended_next_action in {
        "rerun_with_explicit_ocr_request_true_if_text_extraction_is_desired",
        "optional_rerun_with_explicit_ocr_request_true",
        "use_explicit_ocr_path_for_text_focused_followup",
    }:
        followup = "Text-focused follow-up should use the explicit OCR path."
    elif vision_response_recommended_next_action == "continue_visual_inspection_without_ocr":
        followup = "Current routing favors visual inspection without OCR."
    else:
        followup = "Continue with local deterministic inspection if more detail is needed."

    narration = (
        f"{base} The current technical image condition is {scene_classification}. " f"{followup}"
    )
    return _normalize_voice_narration_text(narration, max_length=700)


def _build_vision_voice_safety_note() -> str:
    return (
        "Voice safety preserved: Phase 9I.5E only prepares deterministic narration "
        "metadata. It does not speak automatically, invoke TTS, stream audio, save "
        "audio, monitor live input, or alter the stable voice stack."
    )


def _build_vision_voice_coordination_metadata(
    vision_response_summary: str,
    vision_response_recommended_next_action: str,
    ocr_action_recommendation: str,
    scene_classification: str,
    coordination_confidence: str,
    ocr_execution_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic vision-to-voice coordination metadata builder.

    This creates a voice-ready narration payload from already-generated local
    deterministic vision synthesis. It does NOT call voice services, does NOT
    speak automatically, does NOT stream audio, and does NOT persist content.
    """
    text_extracted = bool(
        ocr_execution_result and ocr_execution_result.get("text_extracted", False)
    )
    vision_voice_priority = _classify_vision_voice_priority(
        vision_response_recommended_next_action=vision_response_recommended_next_action,
        ocr_action_recommendation=ocr_action_recommendation,
        text_extracted=text_extracted,
        coordination_confidence=coordination_confidence,
    )
    vision_voice_narration = _build_vision_voice_narration(
        vision_response_summary=vision_response_summary,
        vision_response_recommended_next_action=vision_response_recommended_next_action,
        ocr_action_recommendation=ocr_action_recommendation,
        scene_classification=scene_classification,
    )
    vision_voice_summary = _build_vision_voice_summary(
        vision_response_summary=vision_response_summary,
        vision_voice_priority=vision_voice_priority,
    )
    vision_voice_safety_note = _build_vision_voice_safety_note()

    return {
        "vision_voice_ready": True,
        "vision_voice_summary": vision_voice_summary,
        "vision_voice_priority": vision_voice_priority,
        "vision_voice_narration": vision_voice_narration,
        "vision_voice_safety_note": vision_voice_safety_note,
    }


def _build_vision_voice_payload(
    vision_voice_ready: bool,
    vision_voice_summary: str,
    vision_voice_priority: str,
    vision_voice_narration: str,
    vision_voice_safety_note: str,
    vision_response_recommended_next_action: str,
    vision_routing_decision: str,
    ocr_action_recommendation: str,
    scene_classification: str,
    coordination_confidence: str,
    narration_readiness: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E controlled voice payload builder.

    This packages existing deterministic narration metadata into a structured
    payload for a future explicit handoff to the stable voice runtime. It does
    NOT invoke pyttsx3, generate audio, stream audio, save audio, modify voice
    routes, or trigger autonomous speech.
    """
    safe_narration = _normalize_voice_narration_text(
        vision_voice_narration,
        max_length=700,
    )
    safe_summary = _normalize_voice_narration_text(
        vision_voice_summary,
        max_length=360,
    )
    safe_narration_readiness = dict(narration_readiness or {})
    readiness_score = safe_narration_readiness.get("readiness_score", 0.0)
    readiness_label = str(safe_narration_readiness.get("readiness_label", "unknown"))
    recommended_narration_length = str(
        safe_narration_readiness.get("recommended_narration_length", "short")
    )
    safe_for_voice_handoff = bool(safe_narration_readiness.get("safe_for_voice_handoff", False))

    return {
        "payload_type": "vision_voice_narration_payload",
        "payload_version": "9I.3",
        "ready": bool(vision_voice_ready),
        "narration": safe_narration,
        "summary": safe_summary,
        "priority": vision_voice_priority,
        "speech_style": "calm_informative",
        "delivery_mode": "explicit_opt_in_only",
        "source_phase": "Phase 9I.5E",
        "source_mode": "deterministic_vision_voice_payload_packaging",
        "recommended_next_action": vision_response_recommended_next_action,
        "vision_routing_decision": vision_routing_decision,
        "ocr_action_recommendation": ocr_action_recommendation,
        "scene_classification": scene_classification,
        "coordination_confidence": coordination_confidence,
        "narration_readiness": safe_narration_readiness,
        "narration_readiness_score": readiness_score,
        "narration_readiness_label": readiness_label,
        "recommended_narration_length": recommended_narration_length,
        "safe_for_voice_handoff": safe_for_voice_handoff,
        "explicit_voice_execution_required": True,
        "bridge_refinement_source_phase": "Phase 10E",
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "requires_explicit_voice_handoff": True,
        "safety_note": vision_voice_safety_note,
    }


def _build_vision_voice_payload_metadata(
    vision_voice_ready: bool,
    vision_voice_summary: str,
    vision_voice_priority: str,
    vision_voice_narration: str,
    vision_voice_safety_note: str,
    vision_response_recommended_next_action: str,
    vision_routing_decision: str,
    ocr_action_recommendation: str,
    scene_classification: str,
    coordination_confidence: str,
    narration_readiness: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E controlled voice payload integration metadata builder.

    This creates a stable payload contract for future optional voice handoff.
    It deliberately avoids direct imports or calls into the voice service so the
    existing voice stack remains untouched and stable.
    """
    vision_voice_payload = _build_vision_voice_payload(
        vision_voice_ready=vision_voice_ready,
        vision_voice_summary=vision_voice_summary,
        vision_voice_priority=vision_voice_priority,
        vision_voice_narration=vision_voice_narration,
        vision_voice_safety_note=vision_voice_safety_note,
        vision_response_recommended_next_action=vision_response_recommended_next_action,
        vision_routing_decision=vision_routing_decision,
        ocr_action_recommendation=ocr_action_recommendation,
        scene_classification=scene_classification,
        coordination_confidence=coordination_confidence,
        narration_readiness=narration_readiness,
    )

    return {
        "vision_voice_payload_ready": bool(vision_voice_ready),
        "vision_voice_payload": vision_voice_payload,
        "vision_voice_payload_format": "deterministic_local_voice_payload",
        "vision_voice_speech_style": "calm_informative",
        "vision_voice_delivery_mode": "explicit_opt_in_only",
        "voice_bridge_refinement_source_phase": "Phase 10E",
    }


def _build_explicit_voice_handoff_payload(
    vision_voice_payload_ready: bool,
    vision_voice_payload: dict[str, Any],
    vision_voice_payload_format: str,
    vision_voice_speech_style: str,
    vision_voice_delivery_mode: str,
) -> dict[str, Any]:
    """
    Phase 9I.5E explicit voice handoff payload builder.

    This creates a stable compatibility contract that a future explicit route can
    pass to the existing voice runtime. It does NOT import the voice service,
    invoke pyttsx3, generate audio, stream audio, save audio, or trigger speech.
    """
    safe_payload = dict(vision_voice_payload or {})
    narration = _normalize_voice_narration_text(
        str(safe_payload.get("narration", "")),
        max_length=700,
    )
    summary = _normalize_voice_narration_text(
        str(safe_payload.get("summary", "")),
        max_length=360,
    )
    priority = str(safe_payload.get("priority", "low"))
    readiness_score = safe_payload.get("narration_readiness_score", 0.0)
    readiness_label = str(safe_payload.get("narration_readiness_label", "unknown"))
    recommended_narration_length = str(safe_payload.get("recommended_narration_length", "short"))
    safe_for_voice_handoff = bool(safe_payload.get("safe_for_voice_handoff", False))

    return {
        "handoff_type": "explicit_vision_to_voice_handoff",
        "handoff_version": "9I.3",
        "ready": bool(vision_voice_payload_ready),
        "safe": True,
        "target_engine": "pyttsx3",
        "target_runtime": "seed_runtime_voice_service",
        "source_payload_format": vision_voice_payload_format,
        "speech_style": vision_voice_speech_style,
        "delivery_mode": vision_voice_delivery_mode,
        "text": narration,
        "summary": summary,
        "priority": priority,
        "narration_readiness_score": readiness_score,
        "narration_readiness_label": readiness_label,
        "recommended_narration_length": recommended_narration_length,
        "safe_for_voice_handoff": safe_for_voice_handoff,
        "bridge_refinement_source_phase": "Phase 10E",
        "voice_request": {
            "text": narration,
            "engine": "pyttsx3",
            "speech_style": vision_voice_speech_style,
            "priority": priority,
            "narration_readiness_score": readiness_score,
            "narration_readiness_label": readiness_label,
            "recommended_narration_length": recommended_narration_length,
            "safe_for_voice_handoff": safe_for_voice_handoff,
            "source": "vision_service",
            "source_phase": "Phase 9I.5E",
            "delivery_mode": "explicit_opt_in_only",
        },
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "requires_explicit_voice_route_call": True,
        "voice_stack_modified": False,
        "safety_note": (
            "Voice handoff safety preserved: this payload is a deterministic "
            "compatibility contract only. It does not invoke TTS, speak "
            "automatically, stream audio, save audio, or modify the stable voice stack."
        ),
    }


def _build_explicit_voice_handoff_metadata(
    vision_voice_payload_ready: bool,
    vision_voice_payload: dict[str, Any],
    vision_voice_payload_format: str,
    vision_voice_speech_style: str,
    vision_voice_delivery_mode: str,
) -> dict[str, Any]:
    """
    Phase 9I.5E explicit voice handoff bridge metadata builder.

    This prepares an optional handoff contract for a future explicit voice route.
    It deliberately avoids direct coupling to voice services so the stable voice
    runtime remains untouched.
    """
    voice_handoff_payload = _build_explicit_voice_handoff_payload(
        vision_voice_payload_ready=vision_voice_payload_ready,
        vision_voice_payload=vision_voice_payload,
        vision_voice_payload_format=vision_voice_payload_format,
        vision_voice_speech_style=vision_voice_speech_style,
        vision_voice_delivery_mode=vision_voice_delivery_mode,
    )

    return {
        "voice_handoff_ready": bool(vision_voice_payload_ready),
        "voice_handoff_payload": voice_handoff_payload,
        "voice_handoff_target_engine": "pyttsx3",
        "voice_handoff_safe": True,
        "voice_handoff_summary": (
            "Phase 9I.5E explicit voice handoff bridge prepared a deterministic "
            "payload for a future explicit call into the stable pyttsx3 voice "
            "runtime. No speech, audio generation, streaming, persistence, or "
            "voice stack modification occurred."
        ),
    }


def _build_optional_voice_execution_request(
    voice_handoff_ready: bool,
    voice_handoff_safe: bool,
    voice_handoff_payload: dict[str, Any],
    voice_handoff_target_engine: str,
) -> dict[str, Any]:
    """
    Phase 9I.5E optional explicit voice execution request builder.

    This creates a safety-gated execution request contract only. It does NOT
    import the voice service, invoke pyttsx3, generate audio, stream audio, save
    audio, modify routes, or trigger speech.
    """
    safe_handoff_payload = dict(voice_handoff_payload or {})
    text = _normalize_voice_narration_text(
        str(safe_handoff_payload.get("text", "")),
        max_length=700,
    )
    priority = str(safe_handoff_payload.get("priority", "low"))
    speech_style = str(safe_handoff_payload.get("speech_style", "calm_informative"))
    readiness_score = safe_handoff_payload.get("narration_readiness_score", 0.0)
    readiness_label = str(safe_handoff_payload.get("narration_readiness_label", "unknown"))
    recommended_narration_length = str(
        safe_handoff_payload.get("recommended_narration_length", "short")
    )
    safe_for_voice_handoff = bool(safe_handoff_payload.get("safe_for_voice_handoff", False))

    execution_available = bool(voice_handoff_ready and voice_handoff_safe and text)

    return {
        "request_type": "optional_explicit_voice_execution_request",
        "request_version": "9I.3",
        "execution_available": execution_available,
        "execution_allowed": False,
        "requires_explicit_voice_execution_request": True,
        "target_engine": voice_handoff_target_engine or "pyttsx3",
        "target_runtime": "seed_runtime_voice_service",
        "text": text,
        "priority": priority,
        "speech_style": speech_style,
        "narration_readiness_score": readiness_score,
        "narration_readiness_label": readiness_label,
        "recommended_narration_length": recommended_narration_length,
        "safe_for_voice_handoff": safe_for_voice_handoff,
        "delivery_mode": "explicit_opt_in_only",
        "source": "vision_service",
        "source_phase": "Phase 9I.5E",
        "safe_to_offer_for_voice_execution": execution_available,
        "auto_execute_enabled": False,
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "voice_stack_modified": False,
        "route_called": False,
        "safety_note": (
            "Voice execution safety preserved: Phase 9I.5E prepares an explicit "
            "execution request contract only. It does not invoke pyttsx3, speak, "
            "stream audio, save audio, call voice routes, or modify the stable "
            "voice runtime."
        ),
    }


def _build_optional_voice_execution_metadata(
    voice_handoff_ready: bool,
    voice_handoff_payload: dict[str, Any],
    voice_handoff_target_engine: str,
    voice_handoff_safe: bool,
) -> dict[str, Any]:
    """
    Phase 9I.5E optional explicit voice execution gate metadata builder.

    This is the safety boundary before any future real voice execution bridge.
    It validates that a handoff payload can be offered for explicit execution
    later, but it deliberately does not execute the voice request.
    """
    voice_execution_request = _build_optional_voice_execution_request(
        voice_handoff_ready=voice_handoff_ready,
        voice_handoff_safe=voice_handoff_safe,
        voice_handoff_payload=voice_handoff_payload,
        voice_handoff_target_engine=voice_handoff_target_engine,
    )

    voice_execution_available = bool(voice_execution_request.get("execution_available", False))

    return {
        "voice_execution_available": voice_execution_available,
        "voice_execution_allowed": False,
        "voice_execution_request": voice_execution_request,
        "voice_execution_safe": True,
        "voice_execution_summary": (
            "Phase 9I.5E optional explicit voice execution gate prepared a "
            "deterministic request contract for a future explicit call into the "
            "stable pyttsx3 voice runtime. Execution remains blocked here: no "
            "speech, TTS invocation, route call, streaming, persistence, or voice "
            "stack modification occurred."
        ),
    }


def _should_recommend_ocr_execution(ocr_readiness: str) -> bool:
    """
    Phase 9C OCR routing foundation.

    This only recommends whether OCR may be worth considering.
    OCR execution still requires an explicit function-level opt-in.
    """
    return ocr_readiness in {"high", "moderate"}


def _detect_optional_ocr_dependencies() -> dict[str, Any]:
    """
    Phase 9C.2+ OCR dependency detection.

    This only checks whether optional OCR-related dependencies appear available.
    It does not call OCR and does not extract text.
    """
    pytesseract_available = find_spec("pytesseract") is not None
    tesseract_binary_path = which("tesseract")
    tesseract_binary_available = tesseract_binary_path is not None

    ocr_dependencies_available = pytesseract_available and tesseract_binary_available

    if ocr_dependencies_available:
        message = (
            "Optional OCR dependencies appear available. OCR execution is available "
            "only through explicit function-level opt-in."
        )
    elif pytesseract_available and not tesseract_binary_available:
        message = (
            "The pytesseract Python package appears available, but the Tesseract binary "
            "was not detected on PATH. OCR execution cannot run."
        )
    elif not pytesseract_available and tesseract_binary_available:
        message = (
            "The Tesseract binary appears available on PATH, but the pytesseract Python "
            "package was not detected. OCR execution cannot run."
        )
    else:
        message = "Optional OCR dependencies were not fully detected. OCR execution cannot run."

    return {
        "ok": True,
        "phase": "Phase 9I.5E",
        "mode": "ocr_dependency_detection",
        "ocr_dependency_detection_enabled": True,
        "ocr_dependencies_available": ocr_dependencies_available,
        "pytesseract_available": pytesseract_available,
        "tesseract_binary_available": tesseract_binary_available,
        "tesseract_binary_path": tesseract_binary_path,
        "ocr_runtime_ready": ocr_dependencies_available,
        "ocr_execution_allowed": False,
        "ocr_execution_performed": False,
        "text_extraction_enabled": False,
        "message": message,
        "created_at": _utc_now(),
    }


def validate_ocr_environment() -> dict[str, Any]:
    """
    Phase 9I.5E OCR environment validation.

    This validates dependency visibility only. OCR execution is separate and
    requires explicit opt-in through run_local_ocr_on_image_bytes(...).
    """
    dependency_report = _detect_optional_ocr_dependencies()

    return {
        "ok": True,
        "phase": "Phase 9I.5E",
        "mode": "ocr_environment_validation",
        "ocr_foundation_installed": True,
        "ocr_dependency_detection_enabled": True,
        "ocr_dependencies_available": dependency_report["ocr_dependencies_available"],
        "pytesseract_available": dependency_report["pytesseract_available"],
        "tesseract_binary_available": dependency_report["tesseract_binary_available"],
        "ocr_runtime_ready": dependency_report["ocr_dependencies_available"],
        "ocr_enabled": True,
        "ocr_engine_loaded": False,
        "execution_allowed": False,
        "execution_performed": False,
        "text_extraction_enabled": False,
        "explicit_opt_in_required": True,
        "dependency_report": dependency_report,
        "message": (
            "OCR environment validation is available. OCR dependencies may be ready, "
            "but OCR execution only occurs through explicit function-level opt-in."
        ),
        "safety_policy": get_vision_safety_policy(),
        "created_at": _utc_now(),
    }


def _build_camera_label(index: int) -> str:
    return f"Camera_{index}"


def _build_camera_descriptor(
    probe_result: dict[str, Any],
    recommended_default_index: Optional[int],
) -> dict[str, Any]:
    index = int(probe_result.get("index", -1))
    available = bool(probe_result.get("available", False))

    return {
        "index": index,
        "label": _build_camera_label(index) if index >= 0 else "Camera_unknown",
        "backend": "CAP_DSHOW",
        "available": available,
        "opened": bool(probe_result.get("opened", False)),
        "reason": probe_result.get("reason", "unknown"),
        "recommended_default": bool(
            available
            and recommended_default_index is not None
            and index == recommended_default_index
        ),
        "frame_captured": False,
        "frame_analyzed": False,
        "frame_saved": False,
        "live_monitoring_enabled": False,
    }


def _build_camera_identification_summary(
    camera_available: bool,
    camera_count: int,
    preferred_camera_index: Optional[int],
) -> str:
    if not camera_available:
        return (
            "No available camera indexes were identified. No frames were captured, "
            "saved, streamed, or analyzed."
        )

    if camera_count == 1:
        return (
            f"One available camera index was identified. Recommended default camera "
            f"index: {preferred_camera_index}. No frames were captured, saved, "
            "streamed, or analyzed."
        )

    return (
        f"{camera_count} available camera indexes were identified. Recommended default "
        f"camera index: {preferred_camera_index}. No frames were captured, saved, "
        "streamed, or analyzed."
    )


def _probe_camera_index(index: int) -> dict[str, Any]:
    """
    Phase 9I.5E safe camera index probe.

    This performs a minimal availability probe only.
    It does NOT capture, save, stream, analyze, or persist frames.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return {
            "index": index,
            "label": _build_camera_label(index),
            "backend": "CAP_DSHOW",
            "available": False,
            "opened": False,
            "reason": "opencv_not_available",
            "frame_captured": False,
            "frame_analyzed": False,
            "frame_saved": False,
        }

    capture = None

    try:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        opened = bool(capture.isOpened())

        return {
            "index": index,
            "label": _build_camera_label(index),
            "backend": "CAP_DSHOW",
            "available": opened,
            "opened": opened,
            "reason": "available" if opened else "not_available",
            "frame_captured": False,
            "frame_analyzed": False,
            "frame_saved": False,
        }

    except Exception as exc:
        return {
            "index": index,
            "label": _build_camera_label(index),
            "backend": "CAP_DSHOW",
            "available": False,
            "opened": False,
            "reason": "camera_probe_failed",
            "detail": repr(exc),
            "frame_captured": False,
            "frame_analyzed": False,
            "frame_saved": False,
        }

    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


def detect_camera_availability(max_indexes: int = 3) -> dict[str, Any]:
    """
    Phase 9I.5E safe camera availability detection.

    This function checks whether a camera device appears available.
    It does NOT:
    - capture frames
    - save images
    - analyze images
    - stream video
    - start monitoring
    - perform OCR
    - perform face recognition
    """
    try:
        import cv2  # type: ignore

        opencv_available = True
        opencv_version = getattr(cv2, "__version__", None)
    except Exception:
        opencv_available = False
        opencv_version = None

    safe_max_indexes = max(0, min(int(max_indexes), 10))

    if not opencv_available:
        return {
            "ok": True,
            "phase": "Phase 9I.5E",
            "mode": "camera_availability_detection",
            "camera_detection_enabled": True,
            "camera_identification_enabled": False,
            "opencv_available": False,
            "opencv_version": opencv_version,
            "camera_available": False,
            "camera_count": 0,
            "preferred_camera_index": None,
            "available_camera_indexes": [],
            "available_camera_descriptors": [],
            "camera_indexes_checked": [],
            "camera_probe_results": [],
            "camera_descriptors": [],
            "identification_summary": (
                "OpenCV is unavailable, so camera indexes could not be identified."
            ),
            "camera_accessed": False,
            "frame_captured": False,
            "frame_analyzed": False,
            "frame_saved": False,
            "live_monitoring_enabled": False,
            "message": (
                "OpenCV is not available, so camera availability cannot be probed. "
                "No camera frames were captured or analyzed."
            ),
            "safety_policy": get_vision_safety_policy(),
            "created_at": _utc_now(),
        }

    probe_results = [_probe_camera_index(index) for index in range(safe_max_indexes)]
    available_indexes = [
        result["index"] for result in probe_results if bool(result.get("available", False))
    ]

    camera_count = len(available_indexes)
    camera_available = camera_count > 0
    preferred_camera_index = available_indexes[0] if available_indexes else None

    camera_descriptors = [
        _build_camera_descriptor(
            probe_result=result,
            recommended_default_index=preferred_camera_index,
        )
        for result in probe_results
    ]

    available_camera_descriptors = [
        descriptor for descriptor in camera_descriptors if bool(descriptor.get("available", False))
    ]

    identification_summary = _build_camera_identification_summary(
        camera_available=camera_available,
        camera_count=camera_count,
        preferred_camera_index=preferred_camera_index,
    )

    return {
        "ok": True,
        "phase": "Phase 9I.5E",
        "mode": "camera_availability_detection",
        "camera_detection_enabled": True,
        "camera_identification_enabled": True,
        "opencv_available": True,
        "opencv_version": opencv_version,
        "camera_available": camera_available,
        "camera_count": camera_count,
        "preferred_camera_index": preferred_camera_index,
        "available_camera_indexes": available_indexes,
        "available_camera_descriptors": available_camera_descriptors,
        "camera_indexes_checked": list(range(safe_max_indexes)),
        "camera_probe_results": probe_results,
        "camera_descriptors": camera_descriptors,
        "camera_accessed": False,
        "frame_captured": False,
        "frame_analyzed": False,
        "frame_saved": False,
        "live_monitoring_enabled": False,
        "identification_summary": identification_summary,
        "message": (
            "Camera availability and index identification completed. No frames were "
            "captured, saved, streamed, analyzed, or persisted."
        ),
        "safety_policy": get_vision_safety_policy(),
        "created_at": _utc_now(),
    }


def validate_single_snapshot_permission_gate(
    explicit_snapshot_request: bool = False,
    requested_camera_index: Optional[int] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E single snapshot permission gate.

    This validates whether a future one-time snapshot capture would be allowed.
    It does NOT capture frames, save images, analyze scenes, stream video, or
    access live monitoring.
    """
    camera_status = detect_camera_availability(max_indexes=3)
    available_camera_indexes = list(camera_status.get("available_camera_indexes", []))
    camera_available = bool(camera_status.get("camera_available", False))
    camera_count = int(camera_status.get("camera_count", 0))

    selected_camera_index = requested_camera_index
    if selected_camera_index is None:
        selected_camera_index = camera_status.get("preferred_camera_index")

    if not explicit_snapshot_request:
        result = CameraSnapshotPermissionGateResult(
            ok=True,
            phase="Phase 9I.5E",
            mode="single_snapshot_permission_gate",
            explicit_snapshot_request=False,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            camera_available=camera_available,
            camera_count=camera_count,
            available_camera_indexes=available_camera_indexes,
            permission_required=True,
            permission_granted=False,
            single_capture_allowed=False,
            snapshot_capture_enabled=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            error=None,
            message=(
                "Snapshot permission gate evaluated. Snapshot capture is not allowed "
                "because explicit_snapshot_request was false."
            ),
            safety_policy=get_vision_safety_policy(),
            created_at=_utc_now(),
        )
        return asdict(result)

    if not camera_available:
        result = CameraSnapshotPermissionGateResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="single_snapshot_permission_gate",
            explicit_snapshot_request=True,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            camera_available=False,
            camera_count=0,
            available_camera_indexes=[],
            permission_required=True,
            permission_granted=False,
            single_capture_allowed=False,
            snapshot_capture_enabled=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            error="camera_not_available",
            message=(
                "Snapshot permission was requested, but no available camera index was "
                "detected. No frame was captured."
            ),
            safety_policy=get_vision_safety_policy(),
            created_at=_utc_now(),
        )
        return asdict(result)

    if selected_camera_index not in available_camera_indexes:
        result = CameraSnapshotPermissionGateResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="single_snapshot_permission_gate",
            explicit_snapshot_request=True,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            camera_available=camera_available,
            camera_count=camera_count,
            available_camera_indexes=available_camera_indexes,
            permission_required=True,
            permission_granted=False,
            single_capture_allowed=False,
            snapshot_capture_enabled=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            error="invalid_camera_index",
            message=(
                "Snapshot permission was requested, but the selected camera index is "
                "not currently available. No frame was captured."
            ),
            safety_policy=get_vision_safety_policy(),
            created_at=_utc_now(),
        )
        return asdict(result)

    result = CameraSnapshotPermissionGateResult(
        ok=True,
        phase="Phase 9I.5E",
        mode="single_snapshot_permission_gate",
        explicit_snapshot_request=True,
        requested_camera_index=requested_camera_index,
        selected_camera_index=selected_camera_index,
        camera_available=camera_available,
        camera_count=camera_count,
        available_camera_indexes=available_camera_indexes,
        permission_required=True,
        permission_granted=True,
        single_capture_allowed=True,
        snapshot_capture_enabled=True,
        frame_captured=False,
        frame_analyzed=False,
        frame_saved=False,
        live_monitoring_enabled=False,
        error=None,
        message=(
            "Snapshot permission gate passed for a future single-frame capture. "
            "Phase 9I.5E permission gate itself does not capture frames; "
            "capture occurs only through the explicit single-snapshot helper."
        ),
        safety_policy=get_vision_safety_policy(),
        created_at=_utc_now(),
    )

    return asdict(result)


def capture_single_camera_snapshot(
    explicit_snapshot_request: bool = False,
    requested_camera_index: Optional[int] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E explicit single-frame camera snapshot capture.

    This function captures at most one frame only after the Phase 9D.3
    permission gate passes.

    It does NOT:
    - stream video
    - start live monitoring
    - save images to disk
    - analyze scenes
    - run OCR
    - perform face recognition
    - infer identity
    """
    permission_gate = validate_single_snapshot_permission_gate(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
    )

    if not bool(permission_gate.get("permission_granted", False)):
        result = CameraSnapshotCaptureResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="explicit_single_snapshot_capture",
            explicit_snapshot_request=explicit_snapshot_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=permission_gate.get("selected_camera_index"),
            permission_gate=permission_gate,
            snapshot_capture_enabled=True,
            single_capture_allowed=False,
            camera_available=bool(permission_gate.get("camera_available", False)),
            camera_opened=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            image_format=None,
            image_mode=None,
            width=None,
            height=None,
            size_bytes=0,
            image_bytes_base64=None,
            error=permission_gate.get("error") or "snapshot_permission_not_granted",
            message=(
                "Snapshot capture was blocked by the permission gate. No frame was "
                "captured, saved, streamed, or analyzed."
            ),
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_released=True,
            created_at=_utc_now(),
        )
        return asdict(result)

    selected_camera_index = permission_gate.get("selected_camera_index")

    if selected_camera_index is None:
        result = CameraSnapshotCaptureResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="explicit_single_snapshot_capture",
            explicit_snapshot_request=explicit_snapshot_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=None,
            permission_gate=permission_gate,
            snapshot_capture_enabled=True,
            single_capture_allowed=False,
            camera_available=True,
            camera_opened=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            image_format=None,
            image_mode=None,
            width=None,
            height=None,
            size_bytes=0,
            image_bytes_base64=None,
            error="selected_camera_index_missing",
            message=(
                "Snapshot capture could not proceed because no selected camera index "
                "was available. No frame was captured."
            ),
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_released=True,
            created_at=_utc_now(),
        )
        return asdict(result)

    try:
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        result = CameraSnapshotCaptureResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="explicit_single_snapshot_capture",
            explicit_snapshot_request=explicit_snapshot_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=int(selected_camera_index),
            permission_gate=permission_gate,
            snapshot_capture_enabled=True,
            single_capture_allowed=True,
            camera_available=True,
            camera_opened=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            image_format=None,
            image_mode=None,
            width=None,
            height=None,
            size_bytes=0,
            image_bytes_base64=None,
            error="snapshot_dependencies_unavailable",
            message=f"Snapshot capture dependencies were unavailable: {repr(exc)}",
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_released=True,
            created_at=_utc_now(),
        )
        return asdict(result)

    capture = None
    camera_released = False

    try:
        capture = cv2.VideoCapture(int(selected_camera_index), cv2.CAP_DSHOW)

        # Phase 9I.5E: best-effort high-resolution capture request for OCR input quality.
        # This does not enable streaming, monitoring, persistence, or repeated capture beyond
        # a small explicit warmup/read sequence inside this single user-requested snapshot.
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, SNAPSHOT_TARGET_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SNAPSHOT_TARGET_HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, SNAPSHOT_TARGET_FPS)
        except Exception:
            pass

        camera_opened = bool(capture.isOpened())
        if not camera_opened:
            result = CameraSnapshotCaptureResult(
                ok=False,
                phase="Phase 9I.5E",
                mode="explicit_single_snapshot_capture",
                explicit_snapshot_request=explicit_snapshot_request,
                requested_camera_index=requested_camera_index,
                selected_camera_index=int(selected_camera_index),
                permission_gate=permission_gate,
                snapshot_capture_enabled=True,
                single_capture_allowed=True,
                camera_available=True,
                camera_opened=False,
                frame_captured=False,
                frame_analyzed=False,
                frame_saved=False,
                live_monitoring_enabled=False,
                image_format=None,
                image_mode=None,
                width=None,
                height=None,
                size_bytes=0,
                image_bytes_base64=None,
                error="camera_open_failed",
                message=(
                    "Snapshot capture was allowed, but the selected camera could not "
                    "be opened. No frame was captured."
                ),
                safety_policy=get_vision_safety_policy(),
                saved=False,
                camera_released=False,
                created_at=_utc_now(),
            )
            return asdict(result)

        ret = False
        frame = None

        safe_warmup_reads = max(1, int(SNAPSHOT_WARMUP_READS))
        for _ in range(safe_warmup_reads):
            ret, frame = capture.read()
            if ret and frame is not None:
                # Continue through the short warmup window so the final frame has
                # the best chance of using the requested resolution/focus state.
                continue

        if not ret or frame is None:
            result = CameraSnapshotCaptureResult(
                ok=False,
                phase="Phase 9I.5E",
                mode="explicit_single_snapshot_capture",
                explicit_snapshot_request=explicit_snapshot_request,
                requested_camera_index=requested_camera_index,
                selected_camera_index=int(selected_camera_index),
                permission_gate=permission_gate,
                snapshot_capture_enabled=True,
                single_capture_allowed=True,
                camera_available=True,
                camera_opened=True,
                frame_captured=False,
                frame_analyzed=False,
                frame_saved=False,
                live_monitoring_enabled=False,
                image_format=None,
                image_mode=None,
                width=None,
                height=None,
                size_bytes=0,
                image_bytes_base64=None,
                error="frame_read_failed",
                message=(
                    "Snapshot capture was allowed, but the camera did not return a " "valid frame."
                ),
                safety_policy=get_vision_safety_policy(),
                saved=False,
                camera_released=False,
                created_at=_utc_now(),
            )
            return asdict(result)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_bytes_base64 = base64.b64encode(image_bytes).decode("ascii")

        width, height = image.size

        result = CameraSnapshotCaptureResult(
            ok=True,
            phase="Phase 9I.5E",
            mode="explicit_single_snapshot_capture",
            explicit_snapshot_request=True,
            requested_camera_index=requested_camera_index,
            selected_camera_index=int(selected_camera_index),
            permission_gate=permission_gate,
            snapshot_capture_enabled=True,
            single_capture_allowed=True,
            camera_available=True,
            camera_opened=True,
            frame_captured=True,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            image_format="PNG",
            image_mode=image.mode,
            width=width,
            height=height,
            size_bytes=len(image_bytes),
            image_bytes_base64=image_bytes_base64,
            error=None,
            message=(
                "Explicit single-frame snapshot captured in memory using Phase 9I.5E "
                "best-effort high-resolution camera settings for improved OCR input quality. "
                "The image was not saved, streamed, analyzed, persisted, or sent to OCR."
            ),
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_released=False,
            created_at=_utc_now(),
        )
        return asdict(result)

    except Exception as exc:
        result = CameraSnapshotCaptureResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="explicit_single_snapshot_capture",
            explicit_snapshot_request=explicit_snapshot_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=int(selected_camera_index),
            permission_gate=permission_gate,
            snapshot_capture_enabled=True,
            single_capture_allowed=True,
            camera_available=True,
            camera_opened=False,
            frame_captured=False,
            frame_analyzed=False,
            frame_saved=False,
            live_monitoring_enabled=False,
            image_format=None,
            image_mode=None,
            width=None,
            height=None,
            size_bytes=0,
            image_bytes_base64=None,
            error="snapshot_capture_failed",
            message=f"Snapshot capture failed safely: {repr(exc)}",
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_released=camera_released,
            created_at=_utc_now(),
        )
        return asdict(result)

    finally:
        if capture is not None:
            try:
                capture.release()
                camera_released = True
            except Exception:
                pass


def capture_and_inspect_single_camera_snapshot(
    explicit_snapshot_request: bool = False,
    requested_camera_index: Optional[int] = None,
    explicit_ocr_request: bool = False,
) -> dict[str, Any]:
    """
    Phase 9I.5E snapshot inspection bridge.

    This function safely bridges the existing explicit single-frame snapshot
    capture helper into the existing lightweight local inspection pipeline.

    The bridge ONLY runs when explicit_snapshot_request=True and the existing
    permission gate allows the one-frame capture.

    It does NOT:
    - save images to disk
    - persist image bytes
    - stream video
    - start live monitoring
    - run autonomous capture loops
    - perform semantic scene understanding
    - perform object recognition
    - perform face recognition
    - infer identity
    - call external AI vision models

    OCR remains separately gated and only runs when explicit_ocr_request=True.
    """
    snapshot_result = capture_single_camera_snapshot(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
    )

    snapshot_capture_ok = bool(snapshot_result.get("ok", False))
    snapshot_capture_performed = bool(snapshot_result.get("frame_captured", False))
    selected_camera_index = snapshot_result.get("selected_camera_index")

    if not snapshot_capture_ok or not snapshot_capture_performed:
        result = CameraSnapshotInspectionBridgeResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="snapshot_inspection_bridge",
            explicit_snapshot_request=explicit_snapshot_request,
            explicit_ocr_request=explicit_ocr_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            snapshot_capture_performed=snapshot_capture_performed,
            snapshot_capture_ok=snapshot_capture_ok,
            inspection_performed=False,
            inspection_ok=False,
            ocr_execution_allowed=False,
            ocr_execution_performed=False,
            text_extracted=False,
            snapshot_result=snapshot_result,
            inspection_result=None,
            error=snapshot_result.get("error") or "snapshot_capture_not_available",
            message=(
                "Snapshot inspection bridge did not run inspection because the "
                "explicit single-frame snapshot was not successfully captured. No "
                "image was saved, streamed, persisted, or analyzed by an AI model."
            ),
            summary="Snapshot inspection bridge stopped safely before inspection.",
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_accessed=bool(snapshot_result.get("camera_opened", False)),
            camera_released=bool(snapshot_result.get("camera_released", True)),
            live_monitoring_enabled=False,
            frame_saved=False,
            analyzed_by_ai_model=False,
            created_at=_utc_now(),
        )
        return asdict(result)

    image_bytes_base64 = snapshot_result.get("image_bytes_base64")
    if not image_bytes_base64:
        result = CameraSnapshotInspectionBridgeResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="snapshot_inspection_bridge",
            explicit_snapshot_request=explicit_snapshot_request,
            explicit_ocr_request=explicit_ocr_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            snapshot_capture_performed=True,
            snapshot_capture_ok=True,
            inspection_performed=False,
            inspection_ok=False,
            ocr_execution_allowed=False,
            ocr_execution_performed=False,
            text_extracted=False,
            snapshot_result=snapshot_result,
            inspection_result=None,
            error="snapshot_image_bytes_missing",
            message=(
                "Snapshot capture succeeded, but no memory-only PNG bytes were "
                "available for inspection. No file was saved or persisted."
            ),
            summary="Snapshot inspection bridge stopped safely because image bytes were missing.",
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_accessed=bool(snapshot_result.get("camera_opened", False)),
            camera_released=bool(snapshot_result.get("camera_released", True)),
            live_monitoring_enabled=False,
            frame_saved=False,
            analyzed_by_ai_model=False,
            created_at=_utc_now(),
        )
        return asdict(result)

    try:
        snapshot_image_bytes = base64.b64decode(str(image_bytes_base64).encode("ascii"))
    except Exception as exc:
        result = CameraSnapshotInspectionBridgeResult(
            ok=False,
            phase="Phase 9I.5E",
            mode="snapshot_inspection_bridge",
            explicit_snapshot_request=explicit_snapshot_request,
            explicit_ocr_request=explicit_ocr_request,
            requested_camera_index=requested_camera_index,
            selected_camera_index=selected_camera_index,
            snapshot_capture_performed=True,
            snapshot_capture_ok=True,
            inspection_performed=False,
            inspection_ok=False,
            ocr_execution_allowed=False,
            ocr_execution_performed=False,
            text_extracted=False,
            snapshot_result=snapshot_result,
            inspection_result=None,
            error="snapshot_image_decode_failed",
            message=f"Snapshot image bytes could not be decoded safely: {repr(exc)}",
            summary="Snapshot inspection bridge stopped safely during image decode.",
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_accessed=bool(snapshot_result.get("camera_opened", False)),
            camera_released=bool(snapshot_result.get("camera_released", True)),
            live_monitoring_enabled=False,
            frame_saved=False,
            analyzed_by_ai_model=False,
            created_at=_utc_now(),
        )
        return asdict(result)

    inspection_result = inspect_image_bytes_lightweight(
        image_bytes=snapshot_image_bytes,
        filename="camera_snapshot_memory_only.png",
        content_type="image/png",
        explicit_ocr_request=explicit_ocr_request,
    )

    inspection_ok = bool(inspection_result.get("ok", False))
    ocr_execution_result = inspection_result.get("ocr_execution_result")
    ocr_execution_performed = bool(
        ocr_execution_result and ocr_execution_result.get("ocr_execution_performed", False)
    )
    text_extracted = bool(inspection_result.get("text_extracted", False))

    if inspection_ok:
        message = (
            "Snapshot inspection bridge completed. One explicit in-memory camera "
            "snapshot was captured and routed into the existing lightweight local "
            "inspection pipeline. No image was saved, streamed, persisted, or analyzed "
            "by an external AI vision model."
        )
        summary = (
            "Phase 9I.5E bridge succeeded: explicit snapshot capture routed into "
            "lightweight local inspection."
        )
        error = None
    else:
        message = (
            "Snapshot was captured, but lightweight local inspection failed safely. "
            "No image was saved, streamed, persisted, or analyzed by an external AI "
            "vision model."
        )
        summary = "Phase 9I.5E bridge captured a snapshot but inspection failed safely."
        error = inspection_result.get("error") or "snapshot_inspection_failed"

    result = CameraSnapshotInspectionBridgeResult(
        ok=inspection_ok,
        phase="Phase 9I.5E",
        mode="snapshot_inspection_bridge",
        explicit_snapshot_request=explicit_snapshot_request,
        explicit_ocr_request=explicit_ocr_request,
        requested_camera_index=requested_camera_index,
        selected_camera_index=selected_camera_index,
        snapshot_capture_performed=True,
        snapshot_capture_ok=True,
        inspection_performed=True,
        inspection_ok=inspection_ok,
        ocr_execution_allowed=explicit_ocr_request,
        ocr_execution_performed=ocr_execution_performed,
        text_extracted=text_extracted,
        snapshot_result=snapshot_result,
        inspection_result=inspection_result,
        error=error,
        message=message,
        summary=summary,
        safety_policy=get_vision_safety_policy(),
        saved=False,
        camera_accessed=bool(snapshot_result.get("camera_opened", False)),
        camera_released=bool(snapshot_result.get("camera_released", True)),
        live_monitoring_enabled=False,
        frame_saved=False,
        analyzed_by_ai_model=False,
        created_at=_utc_now(),
    )

    return asdict(result)


def _extract_ocr_text_for_narration(inspection_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic OCR text extraction for narration construction.

    This helper only reads already-produced OCR metadata from the local inspection
    result. It does NOT run OCR, call external services, infer semantic meaning,
    persist text, or trigger voice execution.
    """
    safe_inspection = dict(inspection_result or {})
    ocr_execution_result = safe_inspection.get("ocr_execution_result")
    if not isinstance(ocr_execution_result, dict):
        return {
            "ocr_text_available_for_narration": False,
            "ocr_text_source": "not_available",
            "ocr_text": "",
            "ocr_text_line_count": 0,
            "ocr_text_word_count": 0,
            "ocr_text_character_count": 0,
        }

    candidates = [
        ocr_execution_result.get("cleaned_extracted_text"),
        ocr_execution_result.get("extracted_text"),
        safe_inspection.get("visible_text"),
    ]

    selected_text = ""
    selected_source = "not_available"
    for source_name, candidate in zip(
        ["cleaned_extracted_text", "extracted_text", "visible_text"],
        candidates,
    ):
        cleaned = " ".join(str(candidate or "").split()).strip()
        if cleaned:
            selected_text = cleaned
            selected_source = source_name
            break

    try:
        line_count = int(ocr_execution_result.get("line_count", 0) or 0)
    except Exception:
        line_count = 0
    try:
        word_count = int(ocr_execution_result.get("word_count", 0) or 0)
    except Exception:
        word_count = len(selected_text.split()) if selected_text else 0
    try:
        character_count = int(ocr_execution_result.get("character_count", 0) or 0)
    except Exception:
        character_count = len(selected_text)

    return {
        "ocr_text_available_for_narration": bool(selected_text),
        "ocr_text_source": selected_source,
        "ocr_text": selected_text,
        "ocr_text_line_count": line_count,
        "ocr_text_word_count": word_count,
        "ocr_text_character_count": character_count,
    }


def _build_deterministic_ocr_narration_text(
    ocr_text: str,
    *,
    max_quoted_chars: int = 520,
) -> str:
    """
    Phase 9I.5E deterministic OCR-to-narration text constructor.

    This creates safe spoken text from already-extracted OCR output. It does NOT
    interpret meaning, summarize semantically, hallucinate missing content, call
    an LLM, or invoke TTS.
    """
    cleaned_text = " ".join((ocr_text or "").split()).strip()
    if not cleaned_text:
        return (
            "I completed the explicit camera snapshot and OCR pass, but I did not "
            "detect readable text with enough confidence to narrate it."
        )

    safe_limit = max(120, int(max_quoted_chars))
    truncated = False
    spoken_text = cleaned_text
    if len(spoken_text) > safe_limit:
        spoken_text = spoken_text[: safe_limit - 3].rstrip() + "..."
        truncated = True

    if truncated:
        return (
            "I completed the explicit camera snapshot and OCR pass. I detected readable "
            f"text. The visible text begins: {spoken_text}"
        )

    return (
        "I completed the explicit camera snapshot and OCR pass. I detected readable "
        f"text: {spoken_text}"
    )


def _build_phase_9i2_ocr_narration_metadata(
    inspection_result: Optional[dict[str, Any]],
    *,
    preferred_voice_engine: str = "edge-tts",
) -> dict[str, Any]:
    """
    Phase 9I.5E deterministic OCR narration bridge metadata builder.

    This prepares a voice handoff payload from already-extracted OCR text for a
    future separate explicit call to /v1/voice/vision_execute. It does NOT execute
    voice output, save audio, stream audio, or modify the voice stack.
    """
    ocr_text_info = _extract_ocr_text_for_narration(inspection_result)
    ocr_text = str(ocr_text_info.get("ocr_text", ""))
    narration_text = _build_deterministic_ocr_narration_text(ocr_text)
    narration_ready = bool(ocr_text_info.get("ocr_text_available_for_narration", False))

    voice_handoff_payload = {
        "handoff_type": "explicit_ocr_to_voice_narration_handoff",
        "handoff_version": "9I.3",
        "ready": narration_ready,
        "safe": True,
        "text": narration_text,
        "narration": narration_text,
        "summary": "Deterministic OCR narration prepared from explicit snapshot OCR output.",
        "priority": "high" if narration_ready else "moderate",
        "speech_style": "calm_informative",
        "delivery_mode": "explicit_opt_in_only",
        "source": "vision_service",
        "source_phase": "Phase 9I.5E",
        "ocr_text_source": ocr_text_info.get("ocr_text_source", "not_available"),
        "auto_speak_enabled": False,
        "tts_invoked": False,
        "audio_generated": False,
        "audio_streamed": False,
        "audio_saved": False,
        "requires_explicit_voice_route_call": True,
        "voice_stack_modified": False,
    }

    explicit_voice_execution_json_body = {
        "explicit_voice_execution_request": True,
        "voice_engine_preference": preferred_voice_engine,
        "voice_handoff_payload": voice_handoff_payload,
    }

    if narration_ready:
        narration_summary = (
            "Phase 9I.5E prepared deterministic OCR narration from already-extracted "
            "snapshot text. Voice execution remains a separate explicit call."
        )
        recommended_next_step = "send_ocr_narration_json_body_to_v1_voice_vision_execute"
    else:
        narration_summary = (
            "Phase 9I.5E could not prepare text-positive OCR narration because no readable "
            "OCR text was available. Voice execution remains blocked by payload readiness."
        )
        recommended_next_step = "retry_snapshot_with_clearer_text_or_lighting"

    return {
        "ocr_narration_phase": "Phase 9I.5E",
        "ocr_narration_mode": "deterministic_ocr_to_voice_narration_bridge",
        "ocr_narration_ready": narration_ready,
        "ocr_narration_text_source": ocr_text_info.get("ocr_text_source", "not_available"),
        "ocr_narration_text_available": bool(
            ocr_text_info.get("ocr_text_available_for_narration", False)
        ),
        "ocr_narration_line_count": ocr_text_info.get("ocr_text_line_count", 0),
        "ocr_narration_word_count": ocr_text_info.get("ocr_text_word_count", 0),
        "ocr_narration_character_count": ocr_text_info.get("ocr_text_character_count", 0),
        "ocr_narration_text": narration_text,
        "ocr_narration_voice_handoff_payload": voice_handoff_payload,
        "ocr_narration_voice_execution_json_body": explicit_voice_execution_json_body,
        "ocr_narration_recommended_next_step": recommended_next_step,
        "ocr_narration_summary": narration_summary,
        "ocr_narration_safety_note": (
            "Safety preserved: Phase 9I.5E only constructs deterministic narration from "
            "already-extracted OCR text. It does not invoke pyttsx3, Edge TTS, voice routes, "
            "audio generation, streaming, autonomous narration, live monitoring, image "
            "persistence, face recognition, identity inference, or external AI vision models."
        ),
    }


def _build_end_to_end_vision_voice_orchestration_metadata(
    snapshot_bridge_result: dict[str, Any],
    inspection_result: Optional[dict[str, Any]],
    *,
    preferred_voice_engine: str = "edge-tts",
) -> dict[str, Any]:
    """
    Phase 9I.5E end-to-end vision-to-voice orchestration validation metadata.

    This helper validates that the existing deterministic vision pipeline has
    produced a voice handoff contract that can be passed to the explicit voice
    execution route. It does NOT call voice routes, invoke TTS, generate audio,
    stream audio, persist images, or start autonomous narration.
    """
    safe_inspection = dict(inspection_result or {})
    base_voice_handoff_payload = dict(safe_inspection.get("voice_handoff_payload") or {})
    voice_execution_request = dict(safe_inspection.get("voice_execution_request") or {})
    ocr_narration_metadata = _build_phase_9i2_ocr_narration_metadata(
        inspection_result=safe_inspection,
        preferred_voice_engine=preferred_voice_engine,
    )
    ocr_narration_payload = dict(
        ocr_narration_metadata.get("ocr_narration_voice_handoff_payload") or {}
    )

    voice_handoff_payload = (
        ocr_narration_payload
        if bool(ocr_narration_metadata.get("ocr_narration_ready", False))
        else base_voice_handoff_payload
    )

    voice_handoff_ready = bool(voice_handoff_payload.get("ready", False)) or bool(
        safe_inspection.get("voice_handoff_ready", False)
    )
    voice_handoff_safe = bool(voice_handoff_payload.get("safe", False)) or bool(
        safe_inspection.get("voice_handoff_safe", False)
    )
    voice_execution_available = bool(
        safe_inspection.get("voice_execution_available", False)
        or bool(ocr_narration_metadata.get("ocr_narration_ready", False))
    )
    voice_execution_safe = bool(safe_inspection.get("voice_execution_safe", False)) or bool(
        voice_handoff_payload.get("safe", False)
    )

    handoff_text = str(
        voice_handoff_payload.get("text")
        or voice_handoff_payload.get("narration")
        or voice_execution_request.get("text")
        or ""
    ).strip()

    snapshot_ok = bool(snapshot_bridge_result.get("snapshot_capture_ok", False))
    inspection_ok = bool(snapshot_bridge_result.get("inspection_ok", False))
    orchestration_ready = bool(
        snapshot_ok
        and inspection_ok
        and voice_handoff_ready
        and voice_handoff_safe
        and voice_execution_available
        and voice_execution_safe
        and handoff_text
    )

    voice_execution_json_body = dict(
        ocr_narration_metadata.get("ocr_narration_voice_execution_json_body")
        or {
            "explicit_voice_execution_request": True,
            "voice_engine_preference": preferred_voice_engine,
            "voice_handoff_payload": voice_handoff_payload,
        }
    )

    if orchestration_ready:
        orchestration_summary = (
            "Phase 9I.5E end-to-end orchestration validation succeeded. The explicit "
            "snapshot, local inspection, OCR coordination, deterministic synthesis, "
            "voice handoff payload, and future explicit voice execution contract are "
            "available for a separate user-triggered call to /v1/voice/vision_execute."
        )
        recommended_next_step = "send_voice_execution_json_body_to_v1_voice_vision_execute"
    else:
        orchestration_summary = (
            "Phase 9I.5E end-to-end orchestration validation completed with one or "
            "more required contracts unavailable. No voice execution occurred."
        )
        recommended_next_step = "review_snapshot_inspection_and_voice_handoff_metadata"

    return {
        "end_to_end_orchestration_phase": "Phase 9I.5E",
        "end_to_end_orchestration_mode": "vision_to_voice_orchestration_validation",
        "end_to_end_orchestration_available": True,
        "end_to_end_orchestration_ready": orchestration_ready,
        "end_to_end_snapshot_ok": snapshot_ok,
        "end_to_end_inspection_ok": inspection_ok,
        "end_to_end_voice_handoff_ready": voice_handoff_ready,
        "end_to_end_voice_handoff_safe": voice_handoff_safe,
        "end_to_end_voice_execution_available": voice_execution_available,
        "end_to_end_voice_execution_safe": voice_execution_safe,
        "end_to_end_voice_text_available": bool(handoff_text),
        "end_to_end_voice_route_target": "/v1/voice/vision_execute",
        "end_to_end_voice_engine_preference": preferred_voice_engine,
        "end_to_end_voice_execution_json_body": voice_execution_json_body,
        "end_to_end_uses_ocr_narration_payload": bool(
            ocr_narration_metadata.get("ocr_narration_ready", False)
        ),
        **ocr_narration_metadata,
        "end_to_end_recommended_next_step": recommended_next_step,
        "end_to_end_orchestration_summary": orchestration_summary,
        "end_to_end_safety_note": (
            "Safety preserved: Phase 9I.5E only validates and packages the deterministic "
            "vision-to-voice orchestration contract. It does not invoke pyttsx3, Edge TTS, "
            "voice routes, audio generation, streaming, autonomous narration, live monitoring, "
            "image persistence, face recognition, identity inference, or external AI vision models."
        ),
    }


def validate_end_to_end_vision_voice_orchestration(
    explicit_snapshot_request: bool = False,
    requested_camera_index: Optional[int] = None,
    explicit_ocr_request: bool = False,
    preferred_voice_engine: str = "edge-tts",
) -> dict[str, Any]:
    """
    Phase 9I.5E explicit end-to-end vision-to-voice orchestration validation.

    This function runs the existing explicit snapshot inspection bridge and then
    packages the resulting deterministic voice handoff contract for a future
    separate call to /v1/voice/vision_execute. It does NOT execute voice output.
    """
    bridge_result = capture_and_inspect_single_camera_snapshot(
        explicit_snapshot_request=explicit_snapshot_request,
        requested_camera_index=requested_camera_index,
        explicit_ocr_request=explicit_ocr_request,
    )
    inspection_result = bridge_result.get("inspection_result")
    orchestration_metadata = _build_end_to_end_vision_voice_orchestration_metadata(
        snapshot_bridge_result=bridge_result,
        inspection_result=inspection_result if isinstance(inspection_result, dict) else None,
        preferred_voice_engine=preferred_voice_engine,
    )

    return {
        "ok": bool(bridge_result.get("ok", False))
        and bool(orchestration_metadata.get("end_to_end_orchestration_ready", False)),
        "phase": "Phase 9I.5E",
        "mode": "end_to_end_vision_voice_orchestration_validation",
        "bridge_result": bridge_result,
        **orchestration_metadata,
        "saved": False,
        "voice_executed": False,
        "audio_generated": False,
        "streaming_started": False,
        "autonomous_narration_started": False,
        "created_at": _utc_now(),
    }


def get_vision_safety_policy() -> dict[str, Any]:
    policy = VisionSafetyPolicy(
        camera_access="explicit_single_snapshot_only_no_streaming",
        live_monitoring="disabled",
        face_identity_recognition="not_allowed",
        sensitive_attribute_inference="not_allowed",
        persistence="disabled",
        file_writes="disabled",
        ocr_execution="explicit_opt_in_only",
        ocr_engine_loaded=False,
        ocr_user_permission_required=True,
        ocr_automatic_execution="disabled",
        text_extraction="explicit_opt_in_only",
        user_trigger_required=True,
    )

    return asdict(policy)


def get_ocr_system_status() -> dict[str, Any]:
    """
    Phase 9I.5E controlled OCR status.

    This confirms OCR dependencies and opt-in execution pathway availability
    without executing OCR.
    """
    dependency_report = _detect_optional_ocr_dependencies()

    return {
        "ok": True,
        "phase": "Phase 9I.5E",
        "mode": "ocr_system_status",
        "ocr_foundation_installed": True,
        "ocr_dependency_detection_enabled": True,
        "ocr_dependencies_available": dependency_report["ocr_dependencies_available"],
        "pytesseract_available": dependency_report["pytesseract_available"],
        "tesseract_binary_available": dependency_report["tesseract_binary_available"],
        "ocr_runtime_ready": dependency_report["ocr_dependencies_available"],
        "ocr_enabled": True,
        "ocr_engine_loaded": False,
        "ocr_execution_allowed": False,
        "ocr_execution_performed": False,
        "text_extraction_enabled": False,
        "permission_required": True,
        "explicit_opt_in_required": True,
        "automatic_ocr_disabled": True,
        "dependency_report": dependency_report,
        "message": (
            "OCR dependencies and controlled execution pathway are available. OCR does "
            "not execute automatically and requires explicit opt-in through the OCR "
            "execution helper."
        ),
        "safety_policy": get_vision_safety_policy(),
        "created_at": _utc_now(),
    }


def build_ocr_readiness_stub(
    ocr_readiness: str,
    filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E OCR readiness handoff stub.

    This builds a future OCR contract and dependency awareness report without
    executing OCR.
    """
    dependency_report = _detect_optional_ocr_dependencies()
    ocr_execution_recommended = _should_recommend_ocr_execution(ocr_readiness)

    if ocr_execution_recommended:
        summary = (
            "OCR may be technically reasonable based on current readiness heuristics, "
            "but OCR execution still requires explicit opt-in."
        )
    else:
        summary = (
            "OCR is not recommended by current readiness heuristics. OCR execution "
            "still requires explicit opt-in even when dependencies are available."
        )

    result = OCRStubResult(
        ok=True,
        phase="Phase 9I.5E",
        mode="ocr_readiness_stub",
        filename=filename,
        ocr_foundation_installed=True,
        ocr_enabled=True,
        ocr_engine_loaded=False,
        ocr_dependencies_available=dependency_report["ocr_dependencies_available"],
        pytesseract_available=dependency_report["pytesseract_available"],
        tesseract_binary_available=dependency_report["tesseract_binary_available"],
        ocr_runtime_ready=dependency_report["ocr_dependencies_available"],
        ocr_execution_allowed=False,
        ocr_execution_performed=False,
        ocr_execution_recommended=ocr_execution_recommended,
        ocr_readiness=ocr_readiness,
        extracted_text=None,
        text_detected=False,
        confidence=None,
        dependency_report=dependency_report,
        summary=summary,
        uncertainty=(
            "This is an OCR readiness stub only. No OCR has been executed from this "
            "path, and no text extraction has been performed."
        ),
        safety_policy=get_vision_safety_policy(),
        created_at=_utc_now(),
    )

    return asdict(result)


def _normalize_ocr_text(raw_text: str) -> str:
    cleaned = " ".join((raw_text or "").split())
    return cleaned.strip()


def _extract_normalized_ocr_lines(raw_text: str) -> list[str]:
    """
    Phase 9I.5E structured OCR line normalization.

    This function structures already-extracted OCR text only. It does NOT
    execute OCR, save text, persist text, or call external services.
    """
    lines: list[str] = []

    for raw_line in (raw_text or "").splitlines():
        cleaned_line = " ".join(raw_line.split()).strip()
        if cleaned_line:
            lines.append(cleaned_line)

    return lines


def _count_ocr_words(cleaned_text: str) -> int:
    if not cleaned_text:
        return 0

    return len([word for word in cleaned_text.split() if word.strip()])


def _estimate_ocr_text_density(
    cleaned_text: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> float:
    """
    Phase 9I.5E deterministic OCR density heuristic.

    If dimensions are unavailable, this falls back to text-length-only density.
    """
    text_length = len(cleaned_text or "")

    if text_length <= 0:
        return 0.0

    if image_width and image_height and image_width > 0 and image_height > 0:
        pixels = image_width * image_height
        return round(text_length / max(pixels / 1000.0, 1.0), 4)

    return round(text_length / 1000.0, 4)


def _score_ocr_text_quality(
    cleaned_text: str,
    text_detected: bool,
    line_count: int,
    word_count: int,
) -> float:
    """
    Phase 9I.5E deterministic OCR quality score.

    This is a lightweight heuristic score from 0.0 to 1.0. It is not model
    confidence and does not use external AI services.
    """
    if not text_detected:
        return 0.0

    text_length = len(cleaned_text)
    alpha_count = sum(1 for char in cleaned_text if char.isalpha())
    numeric_count = sum(1 for char in cleaned_text if char.isdigit())
    whitespace_count = sum(1 for char in cleaned_text if char.isspace())
    readable_count = alpha_count + numeric_count + whitespace_count

    readable_ratio = readable_count / max(text_length, 1)

    score = 0.0

    if text_length >= 20:
        score += 0.30
    elif text_length >= 8:
        score += 0.18
    elif text_length >= 3:
        score += 0.08

    if readable_ratio >= 0.80:
        score += 0.30
    elif readable_ratio >= 0.60:
        score += 0.20
    elif readable_ratio >= 0.40:
        score += 0.10

    if word_count >= 6:
        score += 0.20
    elif word_count >= 2:
        score += 0.12
    elif word_count >= 1:
        score += 0.06

    if line_count >= 2:
        score += 0.10
    elif line_count == 1:
        score += 0.05

    if alpha_count > 0:
        score += 0.10

    return round(min(score, 1.0), 3)


def _classify_ocr_text_structure(
    normalized_lines: list[str],
    word_count: int,
    character_count: int,
) -> str:
    if character_count <= 0:
        return "no_text"

    line_count = len(normalized_lines)

    if line_count >= 4 and word_count >= 20:
        return "multi_line_document_like_text"

    if line_count >= 2 and word_count >= 6:
        return "multi_line_text"

    if line_count == 1 and word_count >= 3:
        return "single_line_text"

    if word_count >= 1:
        return "short_text_fragment"

    return "unstructured_text_fragment"


def _build_structured_ocr_summary(
    text_detected: bool,
    line_count: int,
    word_count: int,
    character_count: int,
    ocr_text_structure: str,
    ocr_text_quality_score: float,
) -> str:
    if not text_detected:
        return (
            "Structured OCR result built. No readable text was detected, so line, "
            "word, and character metrics are zero."
        )

    return (
        "Structured OCR result built with "
        f"{line_count} normalized line(s), {word_count} word(s), "
        f"{character_count} character(s), structure label "
        f"{ocr_text_structure}, and deterministic quality score "
        f"{ocr_text_quality_score}."
    )


def _build_structured_ocr_metadata(
    raw_text: str,
    cleaned_text: str,
    text_detected: bool,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E structured OCR metadata builder.

    This only structures OCR text already produced by the explicit OCR helper.
    It does NOT persist, save, upload, infer identity, or call external models.
    """
    normalized_lines = _extract_normalized_ocr_lines(raw_text=raw_text)
    word_count = _count_ocr_words(cleaned_text=cleaned_text)
    character_count = len(cleaned_text or "")
    line_count = len(normalized_lines)
    text_density = _estimate_ocr_text_density(
        cleaned_text=cleaned_text,
        image_width=image_width,
        image_height=image_height,
    )
    ocr_text_quality_score = _score_ocr_text_quality(
        cleaned_text=cleaned_text,
        text_detected=text_detected,
        line_count=line_count,
        word_count=word_count,
    )
    ocr_text_structure = _classify_ocr_text_structure(
        normalized_lines=normalized_lines,
        word_count=word_count,
        character_count=character_count,
    )
    structured_text_summary = _build_structured_ocr_summary(
        text_detected=text_detected,
        line_count=line_count,
        word_count=word_count,
        character_count=character_count,
        ocr_text_structure=ocr_text_structure,
        ocr_text_quality_score=ocr_text_quality_score,
    )

    return {
        "normalized_text_lines": normalized_lines,
        "line_count": line_count,
        "word_count": word_count,
        "character_count": character_count,
        "text_density": text_density,
        "ocr_text_quality_score": ocr_text_quality_score,
        "ocr_text_structure": ocr_text_structure,
        "structured_text_summary": structured_text_summary,
    }


def _select_ocr_preprocessing_profile(
    preprocessing_recommendation: Optional[str],
) -> tuple[str, str]:
    """
    Phase 9I.5E deterministic OCR preprocessing profile selector.

    This chooses a local preprocessing profile from existing deterministic
    inspection recommendations. It does NOT use AI models, save images,
    access camera, or persist text.
    """
    recommendation = (preprocessing_recommendation or "").strip()

    if not recommendation or recommendation == "no_preprocessing_recommended":
        return (
            "balanced_profile",
            "No specific preprocessing recommendation was present, so the "
            "balanced OCR profile was selected.",
        )

    if "highlight_reduction" in recommendation:
        return (
            "screen_profile",
            "Highlight reduction was recommended, so the screen OCR profile was selected.",
        )

    if "brightness_normalization" in recommendation and "contrast_enhancement" in recommendation:
        return (
            "low_light_profile",
            "Brightness normalization and contrast enhancement were recommended, "
            "so the low-light OCR profile was selected.",
        )

    if "artifact_reduction" in recommendation:
        return (
            "compressed_image_profile",
            "Artifact reduction was recommended, so the compressed-image OCR profile was selected.",
        )

    if "sharpening_filter" in recommendation:
        return (
            "document_profile",
            "Sharpening was recommended, so the document OCR profile was selected.",
        )

    if "contrast_enhancement" in recommendation:
        return (
            "document_profile",
            "Contrast enhancement was recommended, so the document OCR profile was selected.",
        )

    return (
        "balanced_profile",
        "A general preprocessing recommendation was present, so the balanced "
        "OCR profile was selected.",
    )


def _estimate_ocr_noise_score(cleaned_text: str, text_detected: bool) -> float:
    """
    Phase 9I.5E deterministic OCR noise heuristic.

    Returns a score from 0.0 to 1.0 where higher means noisier OCR output.
    """
    if not text_detected:
        return 0.0

    text = cleaned_text or ""
    length = len(text)
    if length <= 0:
        return 0.0

    alpha_numeric_space = sum(1 for char in text if char.isalnum() or char.isspace())
    symbol_count = max(length - alpha_numeric_space, 0)
    whitespace_count = sum(1 for char in text if char.isspace())
    very_short_tokens = sum(1 for token in text.split() if len(token.strip()) == 1)
    token_count = max(len(text.split()), 1)

    symbol_ratio = symbol_count / max(length, 1)
    whitespace_ratio = whitespace_count / max(length, 1)
    fragment_ratio = very_short_tokens / token_count

    noise_score = (
        (symbol_ratio * 0.45) + (max(whitespace_ratio - 0.25, 0.0) * 0.20) + (fragment_ratio * 0.35)
    )
    return round(min(noise_score, 1.0), 3)


def _classify_ocr_noise_label(ocr_noise_score: float, text_detected: bool) -> str:
    if not text_detected:
        return "none"

    if ocr_noise_score < 0.15:
        return "low_noise"

    if ocr_noise_score < 0.35:
        return "moderate_noise"

    if ocr_noise_score < 0.60:
        return "high_noise"

    return "very_high_noise"


def _estimate_ocr_readability_score(
    cleaned_text: str,
    text_detected: bool,
    word_count: int,
    line_count: int,
    ocr_text_quality_score: float,
    ocr_noise_score: float,
) -> float:
    """
    Phase 9I.5E deterministic OCR readability heuristic.

    This is not model confidence. It estimates how human-readable the extracted
    OCR text appears after local cleanup and noise checks.
    """
    if not text_detected:
        return 0.0

    score = 0.0
    text_length = len(cleaned_text or "")

    score += min(max(ocr_text_quality_score, 0.0), 1.0) * 0.45

    if word_count >= 12:
        score += 0.20
    elif word_count >= 5:
        score += 0.14
    elif word_count >= 2:
        score += 0.08

    if line_count >= 2:
        score += 0.10
    elif line_count == 1:
        score += 0.05

    if text_length >= 40:
        score += 0.15
    elif text_length >= 15:
        score += 0.10
    elif text_length >= 3:
        score += 0.04

    score -= min(max(ocr_noise_score, 0.0), 1.0) * 0.25

    return round(max(0.0, min(score, 1.0)), 3)


def _classify_ocr_readability_label(ocr_readability_score: float, text_detected: bool) -> str:
    if not text_detected:
        return "no_readable_text"

    if ocr_readability_score >= 0.75:
        return "high_readability"

    if ocr_readability_score >= 0.50:
        return "moderate_readability"

    if ocr_readability_score >= 0.25:
        return "low_readability"

    return "very_low_readability"


def _build_ocr_refinement_summary(
    ocr_preprocessing_profile: str,
    ocr_readability_label: str,
    ocr_noise_label: str,
    ocr_readability_score: float,
    ocr_noise_score: float,
) -> str:
    return (
        "Phase 9I.5E OCR refinement completed using "
        f"{ocr_preprocessing_profile}. Readability is classified as "
        f"{ocr_readability_label} with score {ocr_readability_score}; OCR noise is "
        f"classified as {ocr_noise_label} with score {ocr_noise_score}."
    )


def _build_ocr_refinement_metadata(
    cleaned_text: str,
    text_detected: bool,
    word_count: int,
    line_count: int,
    ocr_text_quality_score: float,
    preprocessing_recommendation: Optional[str],
) -> dict[str, Any]:
    """
    Phase 9I.5E OCR refinement metadata builder.

    This structures deterministic readability/noise metadata only. It does NOT
    persist OCR text, call external services, or infer identity.
    """
    (
        ocr_preprocessing_profile,
        ocr_preprocessing_profile_reason,
    ) = _select_ocr_preprocessing_profile(preprocessing_recommendation)

    ocr_noise_score = _estimate_ocr_noise_score(
        cleaned_text=cleaned_text,
        text_detected=text_detected,
    )
    ocr_noise_label = _classify_ocr_noise_label(
        ocr_noise_score=ocr_noise_score,
        text_detected=text_detected,
    )
    ocr_readability_score = _estimate_ocr_readability_score(
        cleaned_text=cleaned_text,
        text_detected=text_detected,
        word_count=word_count,
        line_count=line_count,
        ocr_text_quality_score=ocr_text_quality_score,
        ocr_noise_score=ocr_noise_score,
    )
    ocr_readability_label = _classify_ocr_readability_label(
        ocr_readability_score=ocr_readability_score,
        text_detected=text_detected,
    )
    ocr_refinement_summary = _build_ocr_refinement_summary(
        ocr_preprocessing_profile=ocr_preprocessing_profile,
        ocr_readability_label=ocr_readability_label,
        ocr_noise_label=ocr_noise_label,
        ocr_readability_score=ocr_readability_score,
        ocr_noise_score=ocr_noise_score,
    )

    return {
        "ocr_preprocessing_profile": ocr_preprocessing_profile,
        "ocr_preprocessing_profile_reason": ocr_preprocessing_profile_reason,
        "ocr_readability_score": ocr_readability_score,
        "ocr_readability_label": ocr_readability_label,
        "ocr_noise_score": ocr_noise_score,
        "ocr_noise_label": ocr_noise_label,
        "ocr_refinement_summary": ocr_refinement_summary,
    }


def _classify_ocr_quality(
    cleaned_text: str,
    text_detected: bool,
) -> str:
    if not text_detected:
        return "no_text_detected"

    text_length = len(cleaned_text)

    if text_length < 3:
        return "very_low_quality"

    alpha_count = sum(1 for char in cleaned_text if char.isalpha())
    ratio = alpha_count / max(text_length, 1)

    if text_length >= 20 and ratio >= 0.6:
        return "high_quality"

    if text_length >= 8 and ratio >= 0.45:
        return "moderate_quality"

    return "low_quality"


def _classify_ocr_confidence_label(
    quality_label: str,
) -> str:
    if quality_label == "high_quality":
        return "high"

    if quality_label == "moderate_quality":
        return "moderate"

    if quality_label == "low_quality":
        return "low"

    if quality_label == "very_low_quality":
        return "very_low"

    return "none"


def _build_ocr_summary(
    text_detected: bool,
    quality_label: str,
    confidence_label: str,
    extracted_text_length: int,
) -> str:
    if not text_detected:
        return "OCR executed successfully, but no readable text was confidently detected."

    return (
        "OCR executed successfully. "
        f"Detected text quality appears {quality_label} with "
        f"{confidence_label} heuristic confidence. "
        f"Normalized extracted text length: {extracted_text_length}."
    )


def _should_preprocess_for_ocr_from_recommendation(
    preprocessing_recommendation: Optional[str],
) -> bool:
    clean_recommendation = (preprocessing_recommendation or "").strip()

    return bool(clean_recommendation and clean_recommendation != "no_preprocessing_recommended")


def _preprocess_image_for_ocr(
    image: Any,
    preprocessing_recommendation: Optional[str] = None,
) -> tuple[Any, bool, list[str], str, str, str]:
    """
    Phase 9I.5A local in-memory OCR preprocessing with deterministic profiles.

    Stability-preserving refinement:
    - Always applies a safe balanced OCR preparation pass when OCR is explicitly requested.
    - Adds OCR-focused upscaling, contrast handling, mild sharpening, and adaptive thresholding.
    - Adds automatic dark-background normalization so white text on a dark screen becomes
      Tesseract-friendly black text on a light background.

    This function does NOT save files, does NOT access the camera, does NOT call
    external AI models, and does NOT persist image or text data.
    """
    clean_recommendation = (preprocessing_recommendation or "").strip()
    (
        ocr_preprocessing_profile,
        ocr_preprocessing_profile_reason,
    ) = _select_ocr_preprocessing_profile(clean_recommendation)

    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # type: ignore

        working_image = image.convert("L")
        preprocessing_steps: list[str] = [
            "grayscale_conversion",
            f"ocr_profile:{ocr_preprocessing_profile}",
        ]

        # Phase 9I.5A: controlled OCR upscaling. Camera-captured monitor text is
        # often readable to humans but too soft/small for Tesseract. Upscaling is
        # local, deterministic, memory-only, and preserves the existing pipeline.
        try:
            width, height = working_image.size
            if width > 0 and height > 0:
                resampling_filter = getattr(
                    getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
                )
                working_image = working_image.resize(
                    (width * 2, height * 2), resample=resampling_filter
                )
                preprocessing_steps.append("ocr_upscale_2x_lanczos")
        except Exception:
            preprocessing_steps.append("ocr_upscale_skipped_safe_fallback")

        # Preserve existing deterministic profile behavior, but now apply the
        # balanced profile even when no recommendation exists. This fixes the
        # prior raw-camera-frame path without altering routes or safety gates.
        if ocr_preprocessing_profile == "low_light_profile":
            working_image = ImageEnhance.Brightness(working_image).enhance(1.45)
            working_image = ImageEnhance.Contrast(working_image).enhance(1.65)
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.extend(
                [
                    "brightness_normalization",
                    "contrast_enhancement",
                    "light_sharpening_filter",
                ]
            )
        elif ocr_preprocessing_profile == "screen_profile":
            working_image = ImageEnhance.Brightness(working_image).enhance(0.92)
            working_image = ImageEnhance.Contrast(working_image).enhance(1.55)
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.extend(
                [
                    "highlight_reduction",
                    "screen_contrast_targeting",
                    "screen_text_sharpening_filter",
                ]
            )
        elif ocr_preprocessing_profile == "compressed_image_profile":
            working_image = working_image.filter(ImageFilter.MedianFilter(size=3))
            working_image = ImageEnhance.Contrast(working_image).enhance(1.45)
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.extend(
                [
                    "artifact_reduction",
                    "contrast_enhancement",
                    "compression_recovery_sharpening",
                ]
            )
        elif ocr_preprocessing_profile == "document_profile":
            working_image = ImageEnhance.Contrast(working_image).enhance(1.65)
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.extend(
                [
                    "document_contrast_targeting",
                    "document_sharpening_filter",
                ]
            )
        else:
            working_image = ImageEnhance.Contrast(working_image).enhance(1.35)
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.extend(
                [
                    "balanced_contrast_targeting",
                    "balanced_sharpening_filter",
                ]
            )

        # Preserve existing direct recommendation behavior as additive safeguards.
        if (
            "brightness_normalization" in clean_recommendation
            and "brightness_normalization" not in preprocessing_steps
        ):
            working_image = ImageEnhance.Brightness(working_image).enhance(1.25)
            preprocessing_steps.append("brightness_normalization")

        if (
            "contrast_enhancement" in clean_recommendation
            and "contrast_enhancement" not in preprocessing_steps
        ):
            working_image = ImageEnhance.Contrast(working_image).enhance(1.25)
            preprocessing_steps.append("contrast_enhancement")

        if "sharpening_filter" in clean_recommendation and not any(
            "sharpen" in step for step in preprocessing_steps
        ):
            working_image = working_image.filter(ImageFilter.SHARPEN)
            preprocessing_steps.append("sharpening_filter")

        if (
            "artifact_reduction" in clean_recommendation
            and "artifact_reduction" not in preprocessing_steps
        ):
            working_image = working_image.filter(ImageFilter.MedianFilter(size=3))
            preprocessing_steps.append("artifact_reduction")

        if (
            "highlight_reduction" in clean_recommendation
            and "highlight_reduction" not in preprocessing_steps
        ):
            working_image = ImageEnhance.Brightness(working_image).enhance(0.90)
            preprocessing_steps.append("highlight_reduction")

        # Phase 9I.5A: adaptive thresholding is especially useful for screen text,
        # dark-background terminals, Notepad glare, and anti-aliased camera text.
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            grayscale_array = np.array(working_image)
            thresholded_array = cv2.adaptiveThreshold(
                grayscale_array,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )

            # Tesseract generally performs better with dark text on a light
            # background. If thresholding leaves a mostly-dark frame, invert it.
            if float(thresholded_array.mean()) < 127.0:
                thresholded_array = cv2.bitwise_not(thresholded_array)
                preprocessing_steps.append("auto_invert_for_tesseract_dark_background")

            working_image = Image.fromarray(thresholded_array).convert("L")
            preprocessing_steps.append("adaptive_thresholding_gaussian_31_11")

        except Exception:
            # Pillow-only fallback keeps the function deterministic even if cv2/numpy
            # are unavailable or fail unexpectedly.
            try:
                working_image = working_image.point(lambda pixel: 255 if pixel > 145 else 0)
                pixel_values = list(working_image.getdata())
                if pixel_values and (sum(pixel_values) / len(pixel_values)) < 127.0:
                    working_image = ImageOps.invert(working_image)
                    preprocessing_steps.append("pil_auto_invert_for_tesseract_dark_background")
                preprocessing_steps.append("pil_binary_threshold_fallback_145")
            except Exception:
                preprocessing_steps.append("thresholding_skipped_safe_fallback")

        working_image = working_image.filter(ImageFilter.SHARPEN)
        preprocessing_steps.append("final_mild_sharpening_filter")

        return (
            working_image,
            True,
            preprocessing_steps,
            (
                "Local in-memory OCR preprocessing applied using "
                f"{ocr_preprocessing_profile}: " + ", ".join(preprocessing_steps) + "."
            ),
            ocr_preprocessing_profile,
            ocr_preprocessing_profile_reason,
        )

    except Exception:
        return (
            image,
            False,
            [],
            "OCR preprocessing was attempted but failed safely; raw image was used.",
            ocr_preprocessing_profile,
            ocr_preprocessing_profile_reason,
        )


def _safe_ocr_debug_text(value: Optional[str], max_length: int = 1200) -> Optional[str]:
    """
    Phase 9I.5C deterministic OCR diagnostics helper.

    This keeps raw OCR visibility bounded for JSON debugging only. It does
    NOT persist text, save images, write files, call external services, or
    alter OCR behavior.
    """
    if value is None:
        return None

    safe_value = str(value)
    if not safe_value:
        return ""

    safe_max_length = max(120, int(max_length))
    if len(safe_value) <= safe_max_length:
        return safe_value

    return safe_value[: safe_max_length - 3] + "..."


def _build_ocr_diagnostic_flags(
    preprocessing_steps: list[str],
) -> dict[str, bool]:
    """
    Phase 9I.5C deterministic OCR preprocessing diagnostics.

    These flags expose what the local preprocessing path attempted. They are
    observability-only and do not change OCR behavior.
    """
    joined_steps = " ".join(preprocessing_steps or []).lower()

    return {
        "adaptive_threshold_applied": "adaptive_threshold" in joined_steps,
        "inversion_applied": "invert" in joined_steps or "inversion" in joined_steps,
    }


def _score_ocr_candidate_text(value: Optional[str]) -> int:
    """
    Phase 9I.5E deterministic OCR candidate scoring.

    This is local heuristic scoring only. It does NOT persist text, save images,
    call external services, infer semantic content, or trigger autonomous action.
    """
    safe_value = (value or "").strip()
    if not safe_value:
        return 0

    alphanumeric_count = sum(1 for character in safe_value if character.isalnum())
    whitespace_count = sum(1 for character in safe_value if character.isspace())
    punctuation_count = sum(1 for character in safe_value if character in ".,:;!?-_()[]{}@#/\\")
    other_count = max(
        0, len(safe_value) - alphanumeric_count - whitespace_count - punctuation_count
    )

    return max(
        0, (alphanumeric_count * 3) + whitespace_count + punctuation_count - (other_count * 2)
    )


def _build_ocr_pass_diagnostic(
    pass_name: str,
    config: str,
    raw_text: Optional[str],
    image_size: tuple[Optional[int], Optional[int]],
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E bounded per-pass OCR diagnostics.

    This exposes runtime metadata only. It does NOT save images, persist OCR
    text, call external services, or change safety gates.
    """
    safe_raw_text = raw_text or ""
    cleaned_text = _normalize_ocr_text(safe_raw_text)
    width, height = image_size

    return {
        "pass_name": pass_name,
        "config": config,
        "text_detected": bool(cleaned_text),
        "raw_text": _safe_ocr_debug_text(safe_raw_text),
        "cleaned_text": _safe_ocr_debug_text(cleaned_text),
        "raw_length": len(safe_raw_text),
        "cleaned_length": len(cleaned_text),
        "score": _score_ocr_candidate_text(cleaned_text),
        "image_width": width,
        "image_height": height,
        "error": error,
    }


def _safe_image_size(image: Any) -> tuple[Optional[int], Optional[int]]:
    try:
        width, height = image.size
        return int(width), int(height)
    except Exception:
        return None, None


def _build_deterministic_ocr_pass_images(
    original_image: Any,
    preprocessed_image: Any,
) -> list[dict[str, Any]]:
    """
    Phase 9I.5E deterministic multi-pass OCR image preparation.

    Passes are local, memory-only, and deterministic:
    - raw upscaled grayscale
    - sharpened grayscale
    - existing preprocessed adaptive path
    - inverted existing preprocessed path

    This does NOT persist images, perform object recognition, infer identity,
    call external AI models, or access the camera.
    """
    from PIL import Image, ImageFilter, ImageOps  # type: ignore

    passes: list[dict[str, Any]] = []

    base_grayscale = original_image.convert("L")
    try:
        width, height = base_grayscale.size
        if width > 0 and height > 0:
            resampling_filter = getattr(
                getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
            )
            base_grayscale = base_grayscale.resize(
                (width * 2, height * 2), resample=resampling_filter
            )
    except Exception:
        pass

    passes.append(
        {
            "pass_name": "raw_upscaled_grayscale_psm6",
            "image": base_grayscale,
            "config": "--oem 3 --psm 6",
        }
    )

    try:
        sharpened = base_grayscale.filter(ImageFilter.SHARPEN)
    except Exception:
        sharpened = base_grayscale
    passes.append(
        {
            "pass_name": "sharpened_grayscale_psm6",
            "image": sharpened,
            "config": "--oem 3 --psm 6",
        }
    )

    passes.append(
        {
            "pass_name": "adaptive_preprocessed_psm6",
            "image": preprocessed_image,
            "config": "--oem 3 --psm 6",
        }
    )

    passes.append(
        {
            "pass_name": "adaptive_preprocessed_sparse_psm11",
            "image": preprocessed_image,
            "config": "--oem 3 --psm 11",
        }
    )

    try:
        inverted_preprocessed = ImageOps.invert(preprocessed_image.convert("L"))
    except Exception:
        inverted_preprocessed = preprocessed_image

    passes.append(
        {
            "pass_name": "inverted_preprocessed_psm6",
            "image": inverted_preprocessed,
            "config": "--oem 3 --psm 6",
        }
    )

    passes.append(
        {
            "pass_name": "inverted_preprocessed_sparse_psm11",
            "image": inverted_preprocessed,
            "config": "--oem 3 --psm 11",
        }
    )

    return passes


def run_local_ocr_on_image_bytes(
    image_bytes: bytes,
    filename: Optional[str] = None,
    explicit_ocr_request: bool = False,
    preprocessing_recommendation: Optional[str] = None,
) -> dict[str, Any]:
    """
    Phase 9I.5E — Controlled Local Multi-Pass OCR Execution Helper.

    OCR remains strictly gated and local-only. Phase 9I.5E adds deterministic
    multi-pass OCR comparison so camera/screen text can be tested across several
    safe preprocessing profiles without guessing which single pass is best.

    This function does NOT:
    - save files
    - access camera
    - run automatically
    - perform face recognition
    - infer identity
    - call external AI vision models
    - persist extracted text
    """
    safe_bytes = image_bytes or b""
    dependency_report = _detect_optional_ocr_dependencies()
    dependencies_available = bool(dependency_report.get("ocr_dependencies_available", False))

    def _empty_result(
        *,
        ok: bool,
        allowed: bool,
        performed: bool,
        error: Optional[str],
        message: str,
        summary: str,
    ) -> dict[str, Any]:
        result = OCRExecutionResult(
            ok=ok,
            phase="Phase 9I.5E",
            mode="controlled_local_multi_pass_ocr_execution",
            filename=filename,
            explicit_ocr_request=bool(explicit_ocr_request),
            ocr_enabled=True,
            ocr_dependencies_available=dependencies_available,
            ocr_execution_allowed=allowed,
            ocr_execution_performed=performed,
            text_extracted=False,
            text_detected=False,
            extracted_text=None,
            cleaned_extracted_text=None,
            extracted_text_length=0,
            normalized_text_lines=[],
            line_count=0,
            word_count=0,
            character_count=0,
            text_density=0.0,
            ocr_text_quality_score=0.0,
            ocr_text_structure="no_text",
            structured_text_summary=(
                "Structured OCR metadata is empty because OCR text was not available."
            ),
            ocr_preprocessing_profile="not_applicable",
            ocr_preprocessing_profile_reason=(
                "OCR preprocessing profile selection was not applicable because "
                "OCR did not produce text."
            ),
            ocr_readability_score=0.0,
            ocr_readability_label="no_readable_text",
            ocr_noise_score=0.0,
            ocr_noise_label="none",
            ocr_refinement_summary=(
                "OCR refinement metadata is empty because OCR text was not available."
            ),
            preprocessing_applied=False,
            preprocessing_steps=[],
            preprocessing_summary="OCR preprocessing was not applied.",
            ocr_quality_label="none",
            ocr_confidence_label="none",
            confidence=None,
            error=error,
            message=message,
            summary=summary,
            dependency_report=dependency_report,
            safety_policy=get_vision_safety_policy(),
            saved=False,
            camera_accessed=False,
            created_at=_utc_now(),
            selected_ocr_pass=None,
            ocr_pass_diagnostics=[],
        )
        return asdict(result)

    if not explicit_ocr_request:
        return _empty_result(
            ok=True,
            allowed=False,
            performed=False,
            error=None,
            message=(
                "OCR execution was not performed because explicit_ocr_request was false. "
                "This is the required safety gate."
            ),
            summary="OCR execution skipped because explicit opt-in was not provided.",
        )

    if not dependencies_available:
        return _empty_result(
            ok=False,
            allowed=False,
            performed=False,
            error="ocr_dependencies_unavailable",
            message=(
                "OCR execution could not run because optional OCR dependencies " "are unavailable."
            ),
            summary="OCR execution skipped because dependencies were unavailable.",
        )

    if not safe_bytes:
        return _empty_result(
            ok=False,
            allowed=True,
            performed=False,
            error="empty_image_bytes",
            message="OCR execution could not run because no image bytes were provided.",
            summary="OCR execution skipped because image bytes were empty.",
        )

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(BytesIO(safe_bytes))
        image.load()
    except Exception as exc:
        return _empty_result(
            ok=False,
            allowed=True,
            performed=False,
            error="image_open_failed",
            message=f"OCR execution could not open image bytes safely: {repr(exc)}",
            summary="OCR execution skipped because the image could not be opened.",
        )

    preprocessing_applied = False
    preprocessing_steps: list[str] = []
    preprocessing_summary = "OCR preprocessing was not applied."
    ocr_preprocessing_profile = "balanced_profile"
    ocr_preprocessing_profile_reason = "Phase 9I.5E default multi-pass OCR preparation."
    ocr_input_image = image
    preprocessed_image_width: Optional[int] = None
    preprocessed_image_height: Optional[int] = None
    ocr_pass_diagnostics: list[dict[str, Any]] = []
    selected_ocr_pass: Optional[str] = None
    selected_raw_text = ""
    selected_cleaned_text = ""
    selected_config = "--oem 3 --psm 6"

    try:
        (
            ocr_input_image,
            preprocessing_applied,
            preprocessing_steps,
            preprocessing_summary,
            ocr_preprocessing_profile,
            ocr_preprocessing_profile_reason,
        ) = _preprocess_image_for_ocr(
            image=image,
            preprocessing_recommendation=preprocessing_recommendation,
        )

        pass_definitions = _build_deterministic_ocr_pass_images(
            original_image=image,
            preprocessed_image=ocr_input_image,
        )

        best_score = -1

        for pass_definition in pass_definitions:
            pass_name = str(pass_definition.get("pass_name", "unknown_ocr_pass"))
            pass_image = pass_definition.get("image")
            pass_config = str(pass_definition.get("config", "--oem 3 --psm 6"))
            raw_text = ""
            pass_error: Optional[str] = None

            try:
                raw_text = pytesseract.image_to_string(pass_image, config=pass_config)
            except Exception as exc:
                pass_error = repr(exc)
                raw_text = ""

            diagnostic = _build_ocr_pass_diagnostic(
                pass_name=pass_name,
                config=pass_config,
                raw_text=raw_text,
                image_size=_safe_image_size(pass_image),
                error=pass_error,
            )
            ocr_pass_diagnostics.append(diagnostic)

            candidate_cleaned = str(diagnostic.get("cleaned_text") or "")
            candidate_score = int(diagnostic.get("score", 0) or 0)

            if candidate_score > best_score:
                best_score = candidate_score
                selected_ocr_pass = pass_name
                selected_raw_text = raw_text or ""
                selected_cleaned_text = candidate_cleaned
                selected_config = pass_config
                try:
                    ocr_input_image = pass_image
                except Exception:
                    pass

        if selected_ocr_pass:
            preprocessing_steps.append(f"multi_pass_ocr_selected:{selected_ocr_pass}")

        preprocessed_image_width, preprocessed_image_height = _safe_image_size(ocr_input_image)

    except Exception as exc:
        return _empty_result(
            ok=False,
            allowed=True,
            performed=False,
            error="ocr_execution_failed",
            message=f"OCR execution failed safely during multi-pass processing: {repr(exc)}",
            summary="OCR execution failed safely during local multi-pass processing.",
        )

    extracted_text = (selected_raw_text or "").strip()
    cleaned_extracted_text = _normalize_ocr_text(selected_cleaned_text or extracted_text)

    text_detected = bool(cleaned_extracted_text)
    extracted_text_length = len(cleaned_extracted_text)

    structured_ocr_metadata = _build_structured_ocr_metadata(
        raw_text=selected_raw_text,
        cleaned_text=cleaned_extracted_text,
        text_detected=text_detected,
    )

    ocr_refinement_metadata = _build_ocr_refinement_metadata(
        cleaned_text=cleaned_extracted_text,
        text_detected=text_detected,
        word_count=structured_ocr_metadata["word_count"],
        line_count=structured_ocr_metadata["line_count"],
        ocr_text_quality_score=structured_ocr_metadata["ocr_text_quality_score"],
        preprocessing_recommendation=preprocessing_recommendation,
    )

    if not text_detected:
        ocr_refinement_metadata["ocr_preprocessing_profile"] = ocr_preprocessing_profile
        ocr_refinement_metadata["ocr_preprocessing_profile_reason"] = (
            ocr_preprocessing_profile_reason
        )

    ocr_quality_label = _classify_ocr_quality(
        cleaned_text=cleaned_extracted_text,
        text_detected=text_detected,
    )

    ocr_confidence_label = _classify_ocr_confidence_label(
        quality_label=ocr_quality_label,
    )

    ocr_summary = _build_ocr_summary(
        text_detected=text_detected,
        quality_label=ocr_quality_label,
        confidence_label=ocr_confidence_label,
        extracted_text_length=extracted_text_length,
    )

    diagnostic_flags = _build_ocr_diagnostic_flags(preprocessing_steps=preprocessing_steps)

    result = OCRExecutionResult(
        ok=True,
        phase="Phase 9I.5E",
        mode="controlled_local_multi_pass_ocr_execution",
        filename=filename,
        explicit_ocr_request=True,
        ocr_enabled=True,
        ocr_dependencies_available=True,
        ocr_execution_allowed=True,
        ocr_execution_performed=True,
        text_extracted=text_detected,
        text_detected=text_detected,
        extracted_text=extracted_text if text_detected else None,
        cleaned_extracted_text=cleaned_extracted_text if text_detected else None,
        extracted_text_length=extracted_text_length,
        normalized_text_lines=structured_ocr_metadata["normalized_text_lines"],
        line_count=structured_ocr_metadata["line_count"],
        word_count=structured_ocr_metadata["word_count"],
        character_count=structured_ocr_metadata["character_count"],
        text_density=structured_ocr_metadata["text_density"],
        ocr_text_quality_score=structured_ocr_metadata["ocr_text_quality_score"],
        ocr_text_structure=structured_ocr_metadata["ocr_text_structure"],
        structured_text_summary=structured_ocr_metadata["structured_text_summary"],
        ocr_preprocessing_profile=ocr_refinement_metadata["ocr_preprocessing_profile"],
        ocr_preprocessing_profile_reason=ocr_refinement_metadata[
            "ocr_preprocessing_profile_reason"
        ],
        ocr_readability_score=ocr_refinement_metadata["ocr_readability_score"],
        ocr_readability_label=ocr_refinement_metadata["ocr_readability_label"],
        ocr_noise_score=ocr_refinement_metadata["ocr_noise_score"],
        ocr_noise_label=ocr_refinement_metadata["ocr_noise_label"],
        ocr_refinement_summary=ocr_refinement_metadata["ocr_refinement_summary"],
        preprocessing_applied=preprocessing_applied,
        preprocessing_steps=preprocessing_steps,
        preprocessing_summary=preprocessing_summary,
        ocr_quality_label=ocr_quality_label,
        ocr_confidence_label=ocr_confidence_label,
        confidence=None,
        error=None,
        message=(
            "Controlled local multi-pass OCR execution completed. No file was saved, "
            "no camera was accessed, and extracted text was not persisted."
        ),
        summary=ocr_summary,
        dependency_report=dependency_report,
        safety_policy=get_vision_safety_policy(),
        saved=False,
        camera_accessed=False,
        created_at=_utc_now(),
        raw_ocr_text=_safe_ocr_debug_text(selected_raw_text),
        fallback_ocr_text=_safe_ocr_debug_text(
            "\n".join(
                str(item.get("raw_text") or "")
                for item in ocr_pass_diagnostics
                if str(item.get("pass_name") or "") != str(selected_ocr_pass or "")
                and str(item.get("raw_text") or "").strip()
            )
        ),
        ocr_config_used=selected_config,
        adaptive_threshold_applied=bool(diagnostic_flags.get("adaptive_threshold_applied", False)),
        inversion_applied=bool(diagnostic_flags.get("inversion_applied", False)),
        preprocessed_image_width=preprocessed_image_width,
        preprocessed_image_height=preprocessed_image_height,
        selected_ocr_pass=selected_ocr_pass,
        ocr_pass_diagnostics=ocr_pass_diagnostics,
    )

    return asdict(result)


def get_vision_health_status() -> dict[str, Any]:
    ocr_environment = validate_ocr_environment()
    camera_status = detect_camera_availability(max_indexes=3)

    status = VisionHealthStatus(
        ok=True,
        phase="Phase 9I.5E",
        service="vision_service",
        version=VISION_SERVICE_VERSION,
        camera_enabled=False,
        camera_detection_enabled=True,
        camera_available=bool(camera_status.get("camera_available", False)),
        camera_count=int(camera_status.get("camera_count", 0)),
        preferred_camera_index=camera_status.get("preferred_camera_index"),
        live_monitoring_enabled=False,
        image_analysis_enabled=True,
        ocr_foundation_installed=True,
        ocr_dependency_detection_enabled=True,
        ocr_execution_enabled=True,
        ocr_runtime_ready=bool(ocr_environment.get("ocr_runtime_ready", False)),
        ocr_requires_explicit_opt_in=True,
        message=(
            "Vision service foundation is installed. Local lightweight image inspection, "
            "quality assessment, preprocessing readiness evaluation, OCR dependency "
            "detection, controlled opt-in OCR execution helper, explicit "
            "inspection-flow OCR execution, and camera snapshot permission "
            "gating, explicit single-frame snapshot capture, and the "
            "Phase 9I.5E snapshot inspection bridge, and controlled "
            "deterministic scene-condition classification are available. Camera "
            "access, live monitoring, persistence, external AI image models, automatic OCR, "
            "and face recognition remain disabled. OCR execution requires explicit opt-in."
        ),
        created_at=_utc_now(),
    )

    return asdict(status)


def build_static_image_analysis_stub(
    image_name: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> dict[str, Any]:
    clean_image_name = (image_name or "").strip()
    clean_prompt = (user_prompt or "").strip()

    if clean_image_name and clean_prompt:
        summary = (
            "Static image analysis is not active on this stub endpoint. The service "
            "received an image label and prompt, but this route only returns a safe "
            "stub response."
        )
    elif clean_image_name:
        summary = (
            "Static image analysis is not active on this stub endpoint. The service "
            "received an image label, but this route only returns a safe stub response."
        )
    elif clean_prompt:
        summary = (
            "Static image analysis is not active on this stub endpoint. The service "
            "received a prompt, but this route only returns a safe stub response."
        )
    else:
        summary = (
            "Static image analysis is not active on this stub endpoint. Use the Phase "
            "9B upload flow for local lightweight image inspection."
        )

    result = VisionAnalysisStub(
        ok=True,
        phase="Phase 9A.1",
        mode="static_image_stub",
        summary=summary,
        detected_items=[],
        visible_text=None,
        uncertainty=(
            "No visual interpretation has been performed by this stub. This endpoint "
            "is preserved for foundation testing."
        ),
        safety_policy=get_vision_safety_policy(),
        created_at=_utc_now(),
    )

    return asdict(result)


def inspect_image_bytes_lightweight(
    image_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    explicit_ocr_request: bool = False,
) -> dict[str, Any]:
    safe_bytes = image_bytes or b""
    size_bytes = len(safe_bytes)

    try:
        from PIL import Image, ImageStat  # type: ignore
    except Exception:
        return {
            "ok": False,
            "phase": "Phase 9I.5E",
            "mode": "local_lightweight_image_inspection",
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "error": "pillow_not_available",
            "message": (
                "Local image inspection requires Pillow. Install it with: " "pip install pillow"
            ),
            "saved": False,
            "analyzed_by_ai_model": False,
            "camera_accessed": False,
            "ocr_executed": False,
            "text_extracted": False,
            "safety_policy": get_vision_safety_policy(),
            "created_at": _utc_now(),
        }

    if size_bytes <= 0:
        return {
            "ok": False,
            "phase": "Phase 9I.5E",
            "mode": "local_lightweight_image_inspection",
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "error": "empty_image_bytes",
            "message": "Image inspection failed because no image bytes were provided.",
            "saved": False,
            "analyzed_by_ai_model": False,
            "camera_accessed": False,
            "ocr_executed": False,
            "text_extracted": False,
            "safety_policy": get_vision_safety_policy(),
            "created_at": _utc_now(),
        }

    try:
        with Image.open(BytesIO(safe_bytes)) as image:
            image.load()

            image_format = image.format
            image_mode = image.mode
            width, height = image.size

            orientation = _classify_orientation(width, height)
            aspect_ratio = _calculate_aspect_ratio(width, height)
            aspect_category = _classify_aspect_category(width, height, aspect_ratio)
            resolution_category = _classify_resolution(width, height)
            color_profile = _classify_color_profile(image_mode)

            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)

            brightness_estimate = round(float(stat.mean[0]), 2) if stat.mean else None
            brightness_label = _classify_brightness(brightness_estimate)

            sharpness_estimate, sharpness_label = _estimate_sharpness(grayscale)
            contrast_estimate, contrast_label = _estimate_contrast(stat)

            cleanliness_label = _estimate_image_cleanliness(
                image_format=image_format,
                size_bytes=size_bytes,
                width=width,
                height=height,
                sharpness_label=sharpness_label,
                contrast_label=contrast_label,
            )

            quality_assessment = _classify_quality_assessment(
                resolution_category=resolution_category,
                brightness_label=brightness_label,
                sharpness_label=sharpness_label,
                contrast_label=contrast_label,
                cleanliness_label=cleanliness_label,
            )

            inspection_confidence = _estimate_inspection_confidence(
                image_format=image_format,
                image_mode=image_mode,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                brightness_estimate=brightness_estimate,
                color_profile=color_profile,
                sharpness_estimate=sharpness_estimate,
                contrast_estimate=contrast_estimate,
                quality_assessment=quality_assessment,
            )

            ocr_readiness = _classify_ocr_readiness(
                brightness_label=brightness_label,
                sharpness_label=sharpness_label,
                contrast_label=contrast_label,
                resolution_category=resolution_category,
            )

            ocr_foundation_status = build_ocr_readiness_stub(
                ocr_readiness=ocr_readiness,
                filename=filename,
            )
            ocr_environment_status = validate_ocr_environment()
            ocr_execution_recommended = bool(
                ocr_foundation_status.get("ocr_execution_recommended", False)
            )

            object_detection_readiness = _classify_object_detection_readiness(
                sharpness_label=sharpness_label,
                quality_assessment=quality_assessment,
                cleanliness_label=cleanliness_label,
            )

            model_input_readiness = _classify_model_input_readiness(
                resolution_category=resolution_category,
                quality_assessment=quality_assessment,
                inspection_confidence=inspection_confidence,
            )

            preprocessing_recommendation = _build_preprocessing_recommendation(
                brightness_label=brightness_label,
                sharpness_label=sharpness_label,
                contrast_label=contrast_label,
                cleanliness_label=cleanliness_label,
            )

            preprocessing_summary = _build_preprocessing_summary(
                ocr_readiness=ocr_readiness,
                object_detection_readiness=object_detection_readiness,
                model_input_readiness=model_input_readiness,
                preprocessing_recommendation=preprocessing_recommendation,
            )

            ocr_execution_result = None
            if explicit_ocr_request:
                ocr_execution_result = run_local_ocr_on_image_bytes(
                    image_bytes=safe_bytes,
                    filename=filename,
                    explicit_ocr_request=True,
                    preprocessing_recommendation=preprocessing_recommendation,
                )

            scene_condition_metadata = _classify_controlled_scene_condition(
                brightness_label=brightness_label,
                contrast_label=contrast_label,
                sharpness_label=sharpness_label,
                cleanliness_label=cleanliness_label,
                aspect_category=aspect_category,
                color_profile=color_profile,
                quality_assessment=quality_assessment,
                ocr_readiness=ocr_readiness,
                inspection_confidence=inspection_confidence,
                ocr_execution_result=ocr_execution_result,
            )
            scene_classification = scene_condition_metadata["scene_classification"]
            scene_confidence = scene_condition_metadata["scene_confidence"]
            scene_summary = scene_condition_metadata["scene_summary"]
            ocr_priority_recommendation = scene_condition_metadata["ocr_priority_recommendation"]

            coordination_metadata = _build_ocr_scene_coordination_metadata(
                scene_classification=scene_classification,
                scene_confidence=scene_confidence,
                ocr_priority_recommendation=ocr_priority_recommendation,
                ocr_readiness=ocr_readiness,
                inspection_confidence=inspection_confidence,
                explicit_ocr_request=explicit_ocr_request,
                ocr_execution_result=ocr_execution_result,
            )
            coordination_mode = coordination_metadata["coordination_mode"]
            vision_routing_decision = coordination_metadata["vision_routing_decision"]
            ocr_action_recommendation = coordination_metadata["ocr_action_recommendation"]
            scene_ocr_alignment = coordination_metadata["scene_ocr_alignment"]
            coordination_confidence = coordination_metadata["coordination_confidence"]
            coordination_summary = coordination_metadata["coordination_summary"]

            vision_response_metadata = _build_vision_response_synthesis_metadata(
                scene_classification=scene_classification,
                scene_confidence=scene_confidence,
                ocr_priority_recommendation=ocr_priority_recommendation,
                vision_routing_decision=vision_routing_decision,
                ocr_action_recommendation=ocr_action_recommendation,
                scene_ocr_alignment=scene_ocr_alignment,
                coordination_confidence=coordination_confidence,
                ocr_execution_result=ocr_execution_result,
            )
            vision_response_summary = vision_response_metadata["vision_response_summary"]
            vision_response_detail = vision_response_metadata["vision_response_detail"]
            vision_response_recommended_next_action = vision_response_metadata[
                "vision_response_recommended_next_action"
            ]
            vision_response_safety_note = vision_response_metadata["vision_response_safety_note"]

            scene_profile = _build_scene_profile(
                scene_classification=scene_classification,
                brightness_label=brightness_label,
                width=width,
                height=height,
                orientation=orientation,
                aspect_category=aspect_category,
                resolution_category=resolution_category,
                sharpness_label=sharpness_label,
                contrast_label=contrast_label,
                cleanliness_label=cleanliness_label,
                quality_assessment=quality_assessment,
                ocr_readiness=ocr_readiness,
                vision_response_recommended_next_action=vision_response_recommended_next_action,
                ocr_execution_result=ocr_execution_result,
            )

            narration_plan = _build_narration_plan(
                scene_profile=scene_profile,
                coordination_confidence=coordination_confidence,
                ocr_execution_result=ocr_execution_result,
            )

            narration_readiness = _build_narration_readiness(
                scene_profile=scene_profile,
                narration_plan=narration_plan,
                ocr_execution_result=ocr_execution_result,
            )

            vision_voice_metadata = _build_vision_voice_coordination_metadata(
                vision_response_summary=vision_response_summary,
                vision_response_recommended_next_action=vision_response_recommended_next_action,
                ocr_action_recommendation=ocr_action_recommendation,
                scene_classification=scene_classification,
                coordination_confidence=coordination_confidence,
                ocr_execution_result=ocr_execution_result,
            )
            vision_voice_ready = vision_voice_metadata["vision_voice_ready"]
            vision_voice_summary = vision_voice_metadata["vision_voice_summary"]
            vision_voice_priority = vision_voice_metadata["vision_voice_priority"]
            vision_voice_narration = vision_voice_metadata["vision_voice_narration"]
            vision_voice_safety_note = vision_voice_metadata["vision_voice_safety_note"]

            vision_voice_payload_metadata = _build_vision_voice_payload_metadata(
                vision_voice_ready=vision_voice_ready,
                vision_voice_summary=vision_voice_summary,
                vision_voice_priority=vision_voice_priority,
                vision_voice_narration=vision_voice_narration,
                vision_voice_safety_note=vision_voice_safety_note,
                vision_response_recommended_next_action=vision_response_recommended_next_action,
                vision_routing_decision=vision_routing_decision,
                ocr_action_recommendation=ocr_action_recommendation,
                scene_classification=scene_classification,
                coordination_confidence=coordination_confidence,
                narration_readiness=narration_readiness,
            )
            vision_voice_payload_ready = vision_voice_payload_metadata["vision_voice_payload_ready"]
            vision_voice_payload = vision_voice_payload_metadata["vision_voice_payload"]
            vision_voice_payload_format = vision_voice_payload_metadata[
                "vision_voice_payload_format"
            ]
            vision_voice_speech_style = vision_voice_payload_metadata["vision_voice_speech_style"]
            vision_voice_delivery_mode = vision_voice_payload_metadata["vision_voice_delivery_mode"]

            voice_handoff_metadata = _build_explicit_voice_handoff_metadata(
                vision_voice_payload_ready=vision_voice_payload_ready,
                vision_voice_payload=vision_voice_payload,
                vision_voice_payload_format=vision_voice_payload_format,
                vision_voice_speech_style=vision_voice_speech_style,
                vision_voice_delivery_mode=vision_voice_delivery_mode,
            )
            voice_handoff_ready = voice_handoff_metadata["voice_handoff_ready"]
            voice_handoff_payload = voice_handoff_metadata["voice_handoff_payload"]
            voice_handoff_target_engine = voice_handoff_metadata["voice_handoff_target_engine"]
            voice_handoff_safe = voice_handoff_metadata["voice_handoff_safe"]
            voice_handoff_summary = voice_handoff_metadata["voice_handoff_summary"]

            voice_execution_metadata = _build_optional_voice_execution_metadata(
                voice_handoff_ready=voice_handoff_ready,
                voice_handoff_payload=voice_handoff_payload,
                voice_handoff_target_engine=voice_handoff_target_engine,
                voice_handoff_safe=voice_handoff_safe,
            )
            voice_execution_available = voice_execution_metadata["voice_execution_available"]
            voice_execution_allowed = voice_execution_metadata["voice_execution_allowed"]
            voice_execution_request = voice_execution_metadata["voice_execution_request"]
            voice_execution_safe = voice_execution_metadata["voice_execution_safe"]
            voice_execution_summary = voice_execution_metadata["voice_execution_summary"]

            voice_bridge_refinement = _build_voice_bridge_refinement_metadata(
                narration_plan=narration_plan,
                narration_readiness=narration_readiness,
                vision_voice_priority=vision_voice_priority,
                vision_voice_payload_ready=vision_voice_payload_ready,
                voice_handoff_ready=voice_handoff_ready,
                voice_handoff_safe=voice_handoff_safe,
                voice_execution_available=voice_execution_available,
            )

    except Exception as exc:
        return {
            "ok": False,
            "phase": "Phase 9I.5E",
            "mode": "local_lightweight_image_inspection",
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "error": "image_open_failed",
            "message": ("Local image inspection could not open the uploaded file as an image."),
            "detail": repr(exc),
            "saved": False,
            "analyzed_by_ai_model": False,
            "camera_accessed": False,
            "ocr_executed": False,
            "text_extracted": False,
            "safety_policy": get_vision_safety_policy(),
            "created_at": _utc_now(),
        }

    summary = _build_local_inspection_summary(
        width=width,
        height=height,
        orientation=orientation,
        resolution_category=resolution_category,
        aspect_category=aspect_category,
        brightness_label=brightness_label,
        color_profile=color_profile,
        image_mode=image_mode,
        sharpness_label=sharpness_label,
        contrast_label=contrast_label,
        cleanliness_label=cleanliness_label,
        quality_assessment=quality_assessment,
    )

    result = LocalImageInspectionResult(
        ok=True,
        phase="Phase 9I.5E",
        mode="local_lightweight_image_inspection",
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        image_format=image_format,
        image_mode=image_mode,
        width=width,
        height=height,
        orientation=orientation,
        aspect_ratio=aspect_ratio,
        aspect_category=aspect_category,
        resolution_category=resolution_category,
        brightness_estimate=brightness_estimate,
        brightness_label=brightness_label,
        color_profile=color_profile,
        sharpness_estimate=sharpness_estimate,
        sharpness_label=sharpness_label,
        contrast_estimate=contrast_estimate,
        contrast_label=contrast_label,
        cleanliness_label=cleanliness_label,
        quality_assessment=quality_assessment,
        scene_classification=scene_classification,
        scene_confidence=scene_confidence,
        scene_summary=scene_summary,
        scene_profile=scene_profile,
        narration_plan=narration_plan,
        narration_readiness=narration_readiness,
        voice_bridge_refinement=voice_bridge_refinement,
        ocr_priority_recommendation=ocr_priority_recommendation,
        coordination_mode=coordination_mode,
        vision_routing_decision=vision_routing_decision,
        ocr_action_recommendation=ocr_action_recommendation,
        scene_ocr_alignment=scene_ocr_alignment,
        coordination_confidence=coordination_confidence,
        coordination_summary=coordination_summary,
        vision_response_summary=vision_response_summary,
        vision_response_detail=vision_response_detail,
        vision_response_recommended_next_action=vision_response_recommended_next_action,
        vision_response_safety_note=vision_response_safety_note,
        vision_voice_ready=vision_voice_ready,
        vision_voice_summary=vision_voice_summary,
        vision_voice_priority=vision_voice_priority,
        vision_voice_narration=vision_voice_narration,
        vision_voice_safety_note=vision_voice_safety_note,
        vision_voice_payload_ready=vision_voice_payload_ready,
        vision_voice_payload=vision_voice_payload,
        vision_voice_payload_format=vision_voice_payload_format,
        vision_voice_speech_style=vision_voice_speech_style,
        vision_voice_delivery_mode=vision_voice_delivery_mode,
        voice_handoff_ready=voice_handoff_ready,
        voice_handoff_payload=voice_handoff_payload,
        voice_handoff_target_engine=voice_handoff_target_engine,
        voice_handoff_safe=voice_handoff_safe,
        voice_handoff_summary=voice_handoff_summary,
        voice_execution_available=voice_execution_available,
        voice_execution_allowed=voice_execution_allowed,
        voice_execution_request=voice_execution_request,
        voice_execution_safe=voice_execution_safe,
        voice_execution_summary=voice_execution_summary,
        ocr_readiness=ocr_readiness,
        ocr_foundation_status=ocr_foundation_status,
        ocr_environment_status=ocr_environment_status,
        ocr_execution_recommended=ocr_execution_recommended,
        explicit_ocr_request=explicit_ocr_request,
        ocr_execution_result=ocr_execution_result,
        object_detection_readiness=object_detection_readiness,
        model_input_readiness=model_input_readiness,
        preprocessing_recommendation=preprocessing_recommendation,
        preprocessing_summary=preprocessing_summary,
        inspection_confidence=inspection_confidence,
        summary=summary,
        detected_items=[],
        visible_text=None,
        uncertainty=(
            "This is lightweight local inspection with controlled deterministic "
            "scene-condition classification only. No semantic scene understanding, "
            "automatic OCR execution, object detection, face recognition, identity "
            "inference, or AI vision model analysis has been performed. OCR execution "
            "exists only through a separate explicit opt-in helper."
        ),
        safety_policy=get_vision_safety_policy(),
        saved=False,
        analyzed_by_ai_model=False,
        camera_accessed=False,
        ocr_executed=bool(
            ocr_execution_result and ocr_execution_result.get("ocr_execution_performed", False)
        ),
        text_extracted=bool(
            ocr_execution_result and ocr_execution_result.get("text_extracted", False)
        ),
        created_at=_utc_now(),
    )

    return asdict(result)


def build_camera_status_stub() -> dict[str, Any]:
    status = detect_camera_availability(max_indexes=3)
    status["mode"] = "camera_status_detection"
    status["camera_enabled"] = False
    status["snapshot_permission_gate_available"] = True
    status["snapshot_capture_enabled"] = True
    status["single_snapshot_capture_available"] = True
    status["snapshot_inspection_bridge_available"] = True
    status["message"] = (
        "Camera status and index identification completed. Phase 9I.5E can detect "
        "available camera indexes, recommend a default index, validate the snapshot "
        "permission gate, and capture one explicit in-memory snapshot through the "
        "single-snapshot helper, and route that explicit memory-only snapshot into "
        "the local lightweight inspection bridge. It does not stream video, monitor "
        "live input, save images, perform semantic scene understanding, or "
        "perform face recognition."
    )
    return status


def summarize_vision_foundation() -> str:
    return (
        "Phase 9I.5E vision foundation is installed. The service can report health, "
        "return safety policy, produce safe stub responses, perform local lightweight "
        "image inspection, generate deterministic local feature summaries, provide "
        "deterministic local image quality assessment, estimate preprocessing "
        "readiness, provide controlled deterministic scene-condition classification, "
        "provide deterministic structured scene profiles, provide deterministic "
        "narration planning and readiness metadata, provide deterministic OCR and "
        "scene coordination guidance, "
        "detect optional OCR dependency availability, apply optional local "
        "in-memory OCR preprocessing, and expose a controlled explicit "
        "opt-in OCR execution helper, and optionally run OCR during "
        "inspection only when explicit_ocr_request=True. External AI vision "
        "models, automatic OCR, persistence, "
        "live monitoring, automatic camera frame capture, and autonomous "
        "camera analysis remain disabled. Camera availability detection, "
        "camera index identification metadata, single-snapshot permission "
        "gating, explicit in-memory single-frame snapshot capture, "
        "snapshot-to-inspection bridging, and Phase 9I.5E end-to-end "
        "vision-to-voice orchestration validation metadata are supported."
    )
