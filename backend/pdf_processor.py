"""
PDF → Markdown processor using NVIDIA NIM vision model.

Pipeline:
  1. PDF → raw markdown (pymupdf4llm) with embedded base64 images
  2. Remove authors, acknowledgements, references
  3. Extract figures as base64 for notebook embedding (NEW)
  4. Vision model (NVIDIA Nemotron-VL via NIM) converts images → text descriptions
  5. Compress whitespace

Returns both the cleaned markdown AND extracted figures for embedding.
"""

import re
import base64
import io
import time
from typing import Optional
from dataclasses import dataclass, field

from PIL import Image
from openai import OpenAI

import pymupdf4llm
from utils import console


DEFAULT_REMOVE = {
    "references", "reference", "acknowledgements", "acknowledgement",
    "acknowledgments", "acknowledgment", "authors", "author",
    "broader impact", "ethics statement",
}


@dataclass
class PDFResult:
    """Result of PDF processing — markdown text plus extracted figures."""
    markdown: str
    figures: list[dict] = field(default_factory=list)
    # Each figure: {"figure_id": int, "image_b64": str, "mime": str,
    #               "caption": str, "description": str, "page": int}


def pdf_to_markdown(pdf_path: str) -> str:
    console.print(f"  [dim]Converting {pdf_path} to markdown…[/dim]")
    return pymupdf4llm.to_markdown(
        pdf_path, pages=None, page_chunks=False, write_images=False,
        use_ocr=True, ocr_language="eng", show_progress=True, embed_images=True,
    )


def remove_boilerplate(text: str, extra: set[str] | None = None) -> str:
    to_remove = DEFAULT_REMOVE | (extra or set())
    lines = text.split("\n")
    out, skip, skip_level = [], False, None
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", line.strip())
        if m:
            lvl, title = m.group(1), re.sub(r"[*_`]", "", m.group(2)).strip().lower()
            if title in to_remove:
                skip, skip_level = True, lvl
                continue
            elif skip and len(lvl) <= len(skip_level):
                skip, skip_level = False, None
        if not skip:
            out.append(line)
    return "\n".join(out)


def _to_jpeg_b64(raw_b64: str, quality: int = 75) -> Optional[str]:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        console.print(f"    [yellow]⚠ image error: {e}[/yellow]")
        return None


def _to_png_b64(raw_b64: str, max_size: int = 100_000) -> Optional[str]:
    """Convert raw base64 image to optimized PNG for notebook embedding."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large (keep under max_size bytes)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() > max_size:
            # Scale down
            ratio = (max_size / buf.tell()) ** 0.5
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)

        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        console.print(f"    [yellow]⚠ PNG conversion error: {e}[/yellow]")
        return None


def _describe_figure(client: OpenAI, model: str, jpeg_b64: str, alt: str, idx: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
                {"type": "text", "text": (
                    f"This is Figure {idx} from an academic paper"
                    f'{f" (caption: {alt})" if alt else ""}.\n'
                    "Describe concisely: figure type, key elements, axes/labels, "
                    "main takeaway. Under 200 words."
                )},
            ]}],
            max_tokens=512, temperature=0.2,
        )
        c = resp.choices[0].message.content or ""
        from utils import strip_think
        return strip_think(c)
    except Exception as e:
        return f"[Vision error: {e}]"


def extract_and_describe_images(
    md: str,
    client: Optional[OpenAI],
    vision_model: str,
    jpeg_quality: int = 75,
) -> tuple[str, list[dict]]:
    """
    Extract images from markdown, describe via vision model, and return both
    the updated markdown AND the extracted figures for notebook embedding.

    Returns:
        (updated_markdown, figures_list)
    """
    pattern = re.compile(
        r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=\s]+?)\)', re.DOTALL)
    matches = list(pattern.finditer(md))

    if not matches:
        return md, []

    console.print(f"  Found {len(matches)} image(s).")
    figures = []

    if client is None:
        # No vision model — extract figures but use placeholders for descriptions
        ctr = [0]
        def ph(m):
            ctr[0] += 1
            alt = m.group(1).strip() or "figure"
            raw = m.group(3).replace("\n", "").replace(" ", "")

            # Extract for notebook embedding
            png_b64 = _to_png_b64(raw)
            if png_b64:
                figures.append({
                    "figure_id": ctr[0],
                    "image_b64": png_b64,
                    "mime": "image/png",
                    "caption": f"Figure {ctr[0]}: {alt}",
                    "description": "",
                })

            return f"\n**[Figure {ctr[0]}: {alt}]**\n"
        return pattern.sub(ph, md), figures

    repls = []
    for i, m in enumerate(matches, 1):
        alt = m.group(1).strip()
        raw = m.group(3).replace("\n", "").replace(" ", "")

        console.print(f"  [{i}/{len(matches)}] Figure {i}… ", end="")
        t0 = time.time()

        # Convert to JPEG for vision model
        jpeg = _to_jpeg_b64(raw, jpeg_quality)
        desc = _describe_figure(client, vision_model, jpeg, alt, i) if jpeg else "[decode failed]"
        console.print(f"[green]✓[/green] {time.time()-t0:.1f}s")

        # Extract as PNG for notebook embedding
        png_b64 = _to_png_b64(raw)
        if png_b64:
            figures.append({
                "figure_id": i,
                "image_b64": png_b64,
                "mime": "image/png",
                "caption": f"Figure {i}" + (f": {alt}" if alt else ""),
                "description": desc,
            })

        label = f"Figure {i}" + (f": {alt}" if alt else "")
        repls.append((m.start(), m.end(), f"\n**[{label}]** {desc}\n"))

    result = md
    for s, e, txt in reversed(repls):
        result = result[:s] + txt + result[e:]

    return result, figures


def extract_figures_from_pdf(pdf_path: str, max_figures: int = 10) -> list[dict]:
    """
    Extract figures directly from PDF using PyMuPDF (fallback method).

    Uses PyMuPDF's image extraction when pymupdf4llm's embedded images
    aren't available or sufficient. Returns base64-encoded figures.
    """
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        figures = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)

            for img_idx, img_info in enumerate(images):
                if len(figures) >= max_figures:
                    break

                try:
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)
                    if img_data and img_data.get("image"):
                        raw_bytes = img_data["image"]
                        # Convert to PNG
                        img = Image.open(io.BytesIO(raw_bytes))

                        # Skip tiny images (likely icons/logos)
                        if img.width < 100 or img.height < 100:
                            continue

                        # Convert to RGB PNG
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        buf = io.BytesIO()
                        img.save(buf, format="PNG", optimize=True)
                        b64 = base64.b64encode(buf.getvalue()).decode()

                        figures.append({
                            "figure_id": len(figures) + 1,
                            "image_b64": b64,
                            "mime": "image/png",
                            "caption": f"Figure {len(figures) + 1}",
                            "description": "",
                            "page": page_num + 1,
                            "width": img.width,
                            "height": img.height,
                        })
                except Exception:
                    continue

        doc.close()
        return figures
    except Exception as e:
        console.print(f"  [yellow]⚠ PyMuPDF extraction failed: {e}[/yellow]")
        return []


def compress_text(text: str) -> str:
    in_code = False
    out = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            out.append(line)
            continue
        line = line.rstrip()
        if re.fullmatch(r"[-=─═┄_*]{4,}", line.strip()):
            continue
        out.append(re.sub(r"(?<=\S) {2,}", " ", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def process_pdf(
    pdf_path: str,
    vision_client: Optional[OpenAI] = None,
    vision_model: str = "nvidia/nemotron-nano-12b-v2-vl",
    jpeg_quality: int = 75,
    extra_remove: set[str] | None = None,
) -> PDFResult:
    """
    Process PDF and return both markdown and extracted figures.

    Returns PDFResult with:
      - markdown: cleaned, described text
      - figures: list of base64-encoded figures for notebook embedding
    """
    raw = pdf_to_markdown(pdf_path)
    cleaned = remove_boilerplate(raw, extra_remove)
    described, figures = extract_and_describe_images(
        cleaned, vision_client, vision_model, jpeg_quality
    )
    compressed = compress_text(described)

    # If we got fewer figures from markdown extraction, try direct PDF extraction
    if len(figures) < 2:
        console.print("  Attempting direct figure extraction from PDF...")
        direct_figures = extract_figures_from_pdf(pdf_path)
        # Merge, avoiding duplicates by size similarity
        for df in direct_figures:
            if not any(f["figure_id"] == df["figure_id"] for f in figures):
                figures.append(df)

    return PDFResult(markdown=compressed, figures=figures)