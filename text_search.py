import duckdb
from sentence_transformers import SentenceTransformer


query = "single seater couch with cushion"

vec = (
    SentenceTransformer("intfloat/multilingual-e5-base")
    .encode([f"query: {query}"], normalize_embeddings=True)[0]
    .astype("float32")
    .tolist()
)

# DuckDB
con = duckdb.connect()
con.execute("INSTALL lance")
con.execute("LOAD lance")
con.execute("ATTACH './abo-products-lance' AS products (TYPE LANCE)")

res = con.execute(
    """
    SELECT item_id, title, brand, product_type, _distance
    FROM lance_vector_search(
      'products.main.products',
      'text_vec',
      ?,
      k = 3,
      prefilter = true
    )
    ORDER BY _distance ASC
    """,
    [vec],
)

print(res.fetchall())
