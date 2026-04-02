"""
Build a local LanceDB catalog from raw ABO product listings.

This script creates a LanceDB-managed ``products`` table in a local database
directory, with local image paths, normalized metadata, a text embedding
column, and a CLIP multimodal embedding column.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import lancedb
import pyarrow as pa
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listings-root",
        type=Path,
        default=Path("data/abo-listings/listings/metadata"),
        help="Directory containing ABO listings_*.json.gz files.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=Path("data/abo-images-small/images/metadata/images.csv.gz"),
        help="ABO images.csv.gz metadata file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/abo-images-small/images/small"),
        help="Directory containing downscaled ABO images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./abo-products-lance"),
        help="Destination LanceDB directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Omit to ingest all matching rows.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model for multimodal embeddings.",
    )
    parser.add_argument(
        "--text-model",
        default="intfloat/multilingual-e5-base",
        help="SentenceTransformer model used for text embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CLIP and text embedding generation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for CLIP inference.",
    )
    return parser.parse_args()


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(value or "")).strip()


def pick_localized(entries: object, *, prefer_english: bool = True) -> list[str]:
    if not entries:
        return []
    if isinstance(entries, list):
        normalized = []
        for entry in entries:
            if isinstance(entry, dict):
                text = entry.get("value")
                if text is None and "standardized_values" in entry:
                    text = ", ".join(entry["standardized_values"])
                if text is None:
                    continue
                normalized.append((str(entry.get("language_tag", "")), strip_html(str(text))))
            else:
                normalized.append(("", strip_html(str(entry))))
        normalized = [(lang, text) for lang, text in normalized if text]
        if not normalized:
            return []
        if prefer_english:
            english = [text for lang, text in normalized if lang.lower().startswith("en")]
            if english:
                return english
        return [text for _, text in normalized]
    if isinstance(entries, dict):
        return [strip_html(json.dumps(entries, ensure_ascii=False))]
    return [strip_html(str(entries))]


def first_text(entries: object) -> str:
    texts = pick_localized(entries)
    return texts[0] if texts else ""


def joined_text(entries: object, *, sep: str = " ") -> str:
    seen = []
    for text in pick_localized(entries):
        if text and text not in seen:
            seen.append(text)
    return sep.join(seen)


def normalize_product_type(raw_value: object) -> str:
    if isinstance(raw_value, list):
        values = []
        for item in raw_value:
            if isinstance(item, dict):
                value = item.get("value")
            else:
                value = item
            if value:
                values.append(str(value))
        return " / ".join(values)
    return str(raw_value or "")


def category_path(listing: dict) -> str:
    nodes = listing.get("node") or []
    paths = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        path = node.get("path") or node.get("node_name") or ""
        path = str(path).strip()
        if path and path not in paths:
            paths.append(path)
    return " | ".join(paths)


def load_image_lookup(images_csv: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with gzip.open(images_csv, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lookup[row["image_id"]] = row["path"]
    return lookup


def iter_listing_files(listings_root: Path) -> Iterator[Path]:
    return iter(sorted(listings_root.glob("listings_*.json.gz")))


def iter_products(args: argparse.Namespace, image_lookup: dict[str, str]) -> Iterator[dict]:
    seen_item_ids: set[str] = set()
    yielded = 0
    for path in iter_listing_files(args.listings_root):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                listing = json.loads(line)
                item_id = str(listing.get("item_id") or "").strip()
                if not item_id or item_id in seen_item_ids:
                    continue

                image_id = listing.get("main_image_id")
                if not image_id:
                    continue
                image_rel = image_lookup.get(image_id)
                if not image_rel:
                    continue
                image_path = args.images_root / image_rel
                if not image_path.exists():
                    continue

                title = first_text(listing.get("item_name"))
                description = joined_text(listing.get("product_description"))
                if not description:
                    description = joined_text(listing.get("bullet_point"))
                brand = first_text(listing.get("brand"))
                material = first_text(listing.get("material"))
                color = first_text(listing.get("color"))
                style = first_text(listing.get("style"))
                product_type = normalize_product_type(listing.get("product_type"))
                cat_path = category_path(listing)
                bullets = joined_text(listing.get("bullet_point"))
                keywords = joined_text(listing.get("item_keywords"))

                fts_text = " ".join(
                    part
                    for part in [
                        title,
                        description,
                        bullets,
                        keywords,
                        product_type,
                        brand,
                        material,
                        color,
                        style,
                        cat_path,
                    ]
                    if part
                )

                yield {
                    "item_id": item_id,
                    "title": title,
                    "description": description,
                    "brand": brand,
                    "product_type": product_type,
                    "category_path": cat_path,
                    "material": material,
                    "color": color,
                    "style": style,
                    "image_path": str(image_path),
                    "fts_text": fts_text,
                }
                seen_item_ids.add(item_id)
                yielded += 1
                if args.limit is not None and yielded >= args.limit:
                    return


def _as_embedding_tensor(vectors) -> torch.Tensor:
    if isinstance(vectors, torch.Tensor):
        return vectors
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(vectors, attr, None)
        if isinstance(value, torch.Tensor):
            if attr == "last_hidden_state":
                return value[:, 0, :]
            return value
    raise TypeError(f"Unsupported embedding output type: {type(vectors)!r}")


def normalize_rows(vectors) -> list[list[float]]:
    vectors = _as_embedding_tensor(vectors)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return vectors.detach().cpu().to(torch.float32).tolist()


def batched(items: list, batch_size: int) -> Iterator[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_clip_image_embeddings(
    rows: list[dict], model: CLIPModel, processor: CLIPProcessor, device: str, batch_size: int
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batched(rows, batch_size):
        images = []
        for row in batch:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            batch_vectors = model.get_image_features(**inputs)
        vectors.extend(normalize_rows(batch_vectors))
    return vectors


def compute_clip_text_embeddings(
    texts: list[str], model: CLIPModel, processor: CLIPProcessor, device: str, batch_size: int
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batched(texts, batch_size):
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            batch_vectors = model.get_text_features(**inputs)
        vectors.extend(normalize_rows(batch_vectors))
    return vectors


def compute_text_embeddings(
    texts: list[str], model: SentenceTransformer, batch_size: int
) -> list[list[float]]:
    vectors = model.encode(
        [f"passage: {text}" for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.astype("float32").tolist()


def make_product_table(product_rows: list[dict]) -> pa.Table:
    text_dim = len(product_rows[0]["text_vec"])
    multimodal_dim = len(product_rows[0]["multimodal_vec"])
    schema = pa.schema(
        [
            pa.field("item_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("description", pa.string()),
            pa.field("brand", pa.string()),
            pa.field("product_type", pa.string()),
            pa.field("category_path", pa.string()),
            pa.field("material", pa.string()),
            pa.field("color", pa.string()),
            pa.field("style", pa.string()),
            pa.field("image_path", pa.string()),
            pa.field("fts_text", pa.string()),
            pa.field("multimodal_vec", pa.list_(pa.float32(), multimodal_dim)),
            pa.field("text_vec", pa.list_(pa.float32(), text_dim)),
        ]
    )
    return pa.Table.from_pylist(product_rows, schema=schema)


def main() -> None:
    args = parse_args()
    image_lookup = load_image_lookup(args.images_csv)

    print(f"Loading CLIP model {args.clip_model} on {args.device}...", file=sys.stderr)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    print(f"Loading text model {args.text_model} on {args.device}...", file=sys.stderr)
    text_model = SentenceTransformer(args.text_model, device=args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(args.output_root))
    table = None
    total_written = 0

    print("Processing products in batches...", file=sys.stderr)
    product_iter = iter_products(args, image_lookup)
    while True:
        chunk = list(itertools.islice(product_iter, args.batch_size))
        if not chunk:
            break

        clip_vectors = compute_clip_image_embeddings(
            chunk, clip_model, clip_processor, args.device, args.batch_size
        )
        text_vectors = compute_text_embeddings(
            [row["fts_text"] for row in chunk], text_model, args.batch_size
        )

        for idx, row in enumerate(chunk):
            row["multimodal_vec"] = clip_vectors[idx]
            row["text_vec"] = text_vectors[idx]

        pa_table = make_product_table(chunk)

        if table is None:
            table = db.create_table("products", data=pa_table, mode="overwrite")
        else:
            table.add(pa_table)

        total_written += len(chunk)
        print(f"  Wrote batch of {len(chunk)} products ({total_written} total)", file=sys.stderr)

    if total_written == 0:
        raise SystemExit("No products with matching local images were found.")

    print(f"Done. Wrote {total_written} products to {args.output_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
