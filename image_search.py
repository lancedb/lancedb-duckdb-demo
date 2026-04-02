"""
Image-semantic search examples using CLIP embeddings over a LanceDB catalog,
queried from DuckDB via the Lance extension.

Demonstrates two retrieval patterns from the blog post:
  1. "Beige shoes" — vector search joined to a DuckDB sales table
  2. "Single-seater couch" — pure vector search returning image paths
"""

from typing import Any

import duckdb
import torch
from transformers import CLIPModel, CLIPProcessor


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


def encode_text_query(text: str, clip_model, clip_processor) -> list[float]:
    """Encode a text query into a CLIP embedding vector."""
    clip_inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        vec = clip_model.get_text_features(**clip_inputs)
        vec = _as_embedding_tensor(vec)
        vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec.squeeze().to(torch.float32).cpu().tolist()


# ---------------------------------------------------------------------------
# Load CLIP model
# ---------------------------------------------------------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor: Any = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------------------------------------------------------------
# Connect to DuckDB with the Lance extension and attach the LanceDB catalog
# ---------------------------------------------------------------------------
con = duckdb.connect("sales.duckdb")
con.execute("INSTALL lance")
con.execute("LOAD lance")
con.execute("ATTACH './abo-products-lance' AS abo (TYPE LANCE)")

# ===========================================================================
# Query 1: Retrieval + Join
#
# Search for "beige shoes" using CLIP, retrieve the top-3 nearest products
# from the Lance catalog restricted to SHOES, and join the results to the
# DuckDB sales table to find which customers purchased matching products.
# ===========================================================================
beige_shoes_vec = encode_text_query("beige shoes", clip_model, clip_processor)

result = con.execute(
    """
    WITH top_products AS (
      SELECT item_id, title, brand, color, product_type, _distance
      FROM lance_vector_search(
        'abo.main.products',
        'multimodal_vec',
        ?,
        k = 3,
        prefilter = true
      )
      WHERE product_type = 'SHOES'
    )
    SELECT s.id, p.item_id, p.title, p.brand, p.color, p._distance
    FROM sales s
    JOIN top_products p USING (item_id)
    ORDER BY p._distance ASC, s.id ASC
    """,
    [beige_shoes_vec],
).pl()

print("--- Beige shoes: retrieval + sales join ---")
print(result)

# ===========================================================================
# Query 2: Aggregation on top of retrieval
#
# Same retrieval as above, but instead of listing individual sales rows,
# count how many customers purchased beige shoes.
# ===========================================================================
count_result = con.execute(
    """
    WITH top_products AS (
      SELECT item_id, _distance
      FROM lance_vector_search(
        'abo.main.products',
        'multimodal_vec',
        ?,
        k = 3,
        prefilter = true
      )
      WHERE product_type = 'SHOES'
    )
    SELECT count(*) AS num_customers_beige_shoes
    FROM sales s
    JOIN top_products p USING (item_id)
    """,
    [beige_shoes_vec],
).pl()

print("\n--- Beige shoes: customer count ---")
print(count_result)

# ===========================================================================
# Query 3: Pure image retrieval with image paths
#
# Search for "colorful floral slip-on shoes" using CLIP. No join — just
# ranked vector search returning image_path so results can be rendered
# directly in a UI or passed to a downstream vision model.
# ===========================================================================
floral_vec = encode_text_query("colorful floral slip-on shoes", clip_model, clip_processor)

floral_result = con.execute(
    """
    SELECT item_id, title, color, image_path, _distance
    FROM lance_vector_search(
      'abo.main.products',
      'multimodal_vec',
      ?,
      k = 5,
      prefilter = true
    )
    WHERE product_type = 'SHOES'
    ORDER BY _distance ASC
    """,
    [floral_vec],
).pl()

print("\n--- Colorful floral shoes: pure image retrieval ---")
print(floral_result)
print(floral_result["image_path"].to_list())
