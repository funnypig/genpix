"""
Mass image generation with LLM judge.
Uses Replicate for generation, Anthropic claude-haiku-4-5 for vision judging.

Usage:
    python generate_images.py

Requirements:
    pip install replicate anthropic
    export REPLICATE_API_TOKEN=...
    export ANTHROPIC_API_KEY=...
"""

import io
import json
import base64
import logging
import replicate
import anthropic
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """
A hand-lettered sign featuring the text "{text}" written in loose, 
flowing mixed cursive and print lowercase script on a dark weathered black chalkboard surface. 
The lettering is pale sky-blue, rendered with a chalk marker with uneven pressure — some strokes 
thick and blotchy, others thin and scratchy. The baseline is extremely uneven with letters randomly 
rising and falling, inconsistent sizing, and shaky trembling strokes as if written quickly by an 
unsteady, unpracticed hand. Paint blobs and pooling visible at stroke endings, slight smearing 
as if hand dragged across wet paint. Letters appear crowded and some are partially cut off at frame 
edges. The handwriting looks rushed, unprofessional, and slightly illegible — written by a non-artist 
with no formal lettering training using a dried-out chalk marker on a dirty, slightly glossy worn 
surface. Photographed at a slight perspective angle, close-cropped, with hints of other text lines 
visible at the top and bottom edges.
"""

TEXTS = [
    "hello world",
    "coffee $3.50",
    "open 9am-5pm",
    "fresh bread daily",
    "wifi password: cafe123",
]

OUTPUT_DIR = Path("output/accepted")
REJECTED_DIR = Path("output/rejected")
LOG_PATH = Path("output/run.log")

Path("output").mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPLICATE_MODEL = "prunaai/z-image-turbo"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

GEN_PARAMS = {
    "height": 128,
    "width": 1024,
    "num_inference_steps": 8,
    "go_fast": True,
}

# ── Generation ────────────────────────────────────────────────────────────────

def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def generate_image(text: str) -> bytes:
    """Call Replicate and return raw image bytes."""
    prompt = build_prompt(text)
    output = replicate.run(
        REPLICATE_MODEL,
        input={"prompt": prompt, **GEN_PARAMS},
    )
    return output.read()


# ── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_MIN_HEIGHT = 384


def _upscale_for_judge(image_bytes: bytes) -> tuple[bytes, str]:
    """Return (png_bytes, 'image/png') scaled so height >= JUDGE_MIN_HEIGHT."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    if h < JUDGE_MIN_HEIGHT:
        scale = JUDGE_MIN_HEIGHT / h
        img = img.resize((int(w * scale), JUDGE_MIN_HEIGHT), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def judge_image(image_bytes: bytes, expected_text: str) -> dict:
    """
    Ask Claude claude-haiku-4-5 (vision) whether the expected_text is fully visible
    and correctly written in the image.

    Returns:
        {
            "passed": bool,
            "visible": bool,
            "exact_match": bool,
            "found_text": str
        }
    """
    client = anthropic.Anthropic()
    judge_bytes, media_type = _upscale_for_judge(image_bytes)
    b64 = base64.standard_b64encode(judge_bytes).decode("utf-8")

    system = "Respond ONLY with a valid JSON object, no markdown, no explanation."

    user = f"""I expected it to read: "{expected_text}"

Return a JSON object:
- "visible": true if any text is visible
- "found_text": text you can read (empty string if none)
- "exact_match": true if found_text matches expected (case-insensitive, ignore minor chalk imperfections)
- "passed": true if visible AND exact_match"""

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=128,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model adds extra chars
        return {
            "passed": False,
            "visible": False,
            "exact_match": False,
            "found_text": "",
        }


# ── Storage ───────────────────────────────────────────────────────────────────

def _unique_stem(dest: Path, image_id: int) -> str:
    stem = str(image_id)
    if not (dest / f"{stem}.jpg").exists():
        return stem
    suffix = 1
    while (dest / f"{image_id}_{suffix}.jpg").exists():
        suffix += 1
    return f"{image_id}_{suffix}"


def save_image(image_bytes: bytes, text: str, image_id: int, passed: bool) -> Path:
    """
    Save image + text sidecar to accepted/ or rejected/.
    Filenames: {image_id}.jpg and {image_id}.txt (or {image_id}_N.jpg if name taken).
    """
    dest = OUTPUT_DIR if passed else REJECTED_DIR
    dest.mkdir(parents=True, exist_ok=True)

    stem = _unique_stem(dest, image_id)
    img_path = dest / f"{stem}.jpg"
    txt_path = dest / f"{stem}.txt"

    img_path.write_bytes(image_bytes)
    txt_path.write_text(text, encoding="utf-8")

    return img_path


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_text(text: str, image_id: int) -> dict:
    """Full pipeline for a single text entry."""
    log.info("[%s] Generating: %r", image_id, text)

    image_bytes = generate_image(text)

    log.info("[%s] Judging...", image_id)
    verdict = judge_image(image_bytes, text)

    passed = verdict.get("passed", False)
    saved_path = save_image(image_bytes, text, image_id, passed)

    status = "ACCEPTED" if passed else "REJECTED"
    log.info("[%s] %s found_text=%r -> %s", image_id, status, verdict.get("found_text", ""), saved_path)

    return {
        "image_id": image_id,
        "text": text,
        "passed": passed,
        "verdict": verdict,
        "path": str(saved_path),
    }


MAX_CONSECUTIVE_REJECTED = 5


def run(texts: list[str]) -> list[dict]:
    """Process all texts and return results."""
    results = []
    consecutive_rejected = 0
    for i, text in enumerate(tqdm(texts, unit="img"), start=1):
        try:
            result = process_text(text, i)
        except Exception as e:
            log.error("[%s] ERROR: %s", i, e)
            result = {"image_id": i, "text": text, "passed": False, "error": str(e)}
        results.append(result)

        if result.get("passed"):
            consecutive_rejected = 0
        else:
            consecutive_rejected += 1
            if consecutive_rejected >= MAX_CONSECUTIVE_REJECTED:
                log.error("Stopping: %d consecutive rejections", consecutive_rejected)
                break

    accepted = sum(1 for r in results if r.get("passed"))
    log.info("Done: %d/%d accepted", accepted, len(results))
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        texts = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    else:
        texts = TEXTS
    run(texts)

