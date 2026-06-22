# src/langgraph/tools/nanobanana_tool.py
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from PIL import Image, ImageOps
from dotenv import load_dotenv
from google import genai  # pip install google-genai pillow python-dotenv
from google.genai import types


# -----------------------------
# Helpers: Files & Canvas
# -----------------------------
def ensure_outdir(base_dir: Path, name: str = "outputs") -> Path:
    outdir = base_dir / name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def timestamp_name(prefix: str = "nanobanana") -> str:
    return f"{prefix}_{datetime.now().strftime('%H%M%S')}.png"


def resize_to_exact_canvas(
    img: Image.Image,
    target_w: int,
    target_h: int,
    keep_aspect: bool = True,
    fill=(0, 0, 0, 0),
) -> Image.Image:
    if keep_aspect:
        return ImageOps.pad(
            img, (target_w, target_h),
            method=Image.LANCZOS, color=fill, centering=(0.5, 0.5)
        )
    return img.resize((target_w, target_h), resample=Image.LANCZOS)


def postprocess_to_input_size(
    generated_path: Path, input_path: Path, keep_aspect: bool = True
) -> Path:
    src = Image.open(input_path).convert("RGBA")
    gen = Image.open(generated_path).convert("RGBA")
    out = resize_to_exact_canvas(
        gen, src.width, src.height, keep_aspect=keep_aspect, fill=(0, 0, 0, 0)
    )
    out.save(generated_path)
    return generated_path


# -----------------------------
# Env: src/.env 만 로드
# -----------------------------
def load_api_key() -> str:
    """
    GEMINI_API_KEY를 src/.env에서 로드.
    (현재 파일: src/langgraph/tools → src는 parents[2])
    """
    src_dir = Path(__file__).resolve().parents[2]  # .../src
    env_path = src_dir / ".env"
    load_dotenv(env_path, override=False)  # 기존 환경변수 우선
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in src/.env")
    return api_key


# -----------------------------
# Core: Image Generation
# -----------------------------
def generate_image(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    prefix: str = "nanobanana",
    model: str = "gemini-2.5-flash-image",
) -> Optional[str]:
    """
    Google Generative AI 호출로 이미지 생성.
    반환: 생성 이미지의 '절대 경로'(str) 또는 None
    """
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    contents: list = [prompt]
    if image_paths:
        for p in image_paths:
            contents.append(Image.open(p))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            # 가능한 한 결정론적으로 만들기 위한 설정
            temperature=0,
            top_p=1.0,
            top_k=1,
            candidate_count=1,
            # 재현 가능한 이미지를 위해 고정 seed 사용
            seed=int(os.getenv("GEMINI_SEED", "42")),
        ),
    )


    base_dir = Path(__file__).resolve().parent          # .../src/langgraph/tools
    outdir = ensure_outdir(base_dir, "outputs")
    output_path = outdir / timestamp_name(prefix)

    try:
        parts = response.candidates[0].content.parts
        for part in parts:
            if getattr(part, "inline_data", None):
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(output_path)
                if image_paths:
                    postprocess_to_input_size(output_path, Path(image_paths[-1]), keep_aspect=True)
                return str(output_path.resolve())
    except Exception as e:
        raise RuntimeError(f"⚠️ Failed to parse response: {e}")

    return None


# -----------------------------
# Public Runner (LangGraph Tool)
# -----------------------------
def run_nanobanana(image_path: str, instruction: str) -> str:
    """
    LangGraph 툴 러너(entrypoint).
    입력: image_path(현재 base), instruction(Planning 노드 생성)
    출력: 생성/후처리된 이미지의 절대 경로(str)
    """
    ipath = Path(image_path)
    if not ipath.exists():
        raise FileNotFoundError(f"[nanobanana] input image not found: {ipath}")

    if not instruction or not instruction.strip():
        raise ValueError("[nanobanana] 'instruction' (text prompt) is required.")

    try:
        out = generate_image(
            prompt=instruction.strip(),
            image_paths=[str(ipath)],
            prefix="nanobanana",
        )
    except Exception as e:
        raise RuntimeError(f"[nanobanana] tool invocation failed: {e}")

    if not out:
        raise RuntimeError("[nanobanana] tool returned no image.")

    opath = Path(out).resolve()
    if not opath.exists():
        raise RuntimeError(f"[nanobanana] generated image not found at: {opath}")

    # 상위 노드에서 r_save_artifact(...)로 표준 저장소로 복사/관리
    return str(opath)
