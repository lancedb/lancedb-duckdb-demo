# LanceDB ｘ DuckDB Demo

A hands-on example showing how LanceDB and DuckDB work together on multimodal data. The repo ingests the [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (ABO) dataset into a LanceDB catalog with CLIP image embeddings and text embeddings, then uses DuckDB with the [Lance extension](https://github.com/lance-format/lance-duckdb) to query, join, and materialize results via SQL.

The LanceDB `products` table stores product metadata, image paths, a CLIP multimodal vector (`multimodal_vec`), and a text-semantic vector (`text_vec`). DuckDB attaches the Lance directory as a namespace and runs SQL directly on top of it -- vector search, joins to local DuckDB tables, and aggregations to answer questions about the data.

## Setup

Requires Python 3.12+.

```bash
# Sync dependencies from pyproject.toml
uv sync
# Add dependencies as needed
uv add ...
```

### Download the ABO dataset

Download the listings metadata and small images from the [ABO dataset page](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) and extract them into `data/`:

After extracting, the layout should look like:

```bash
data/
  abo-listings/listings/metadata/   # listings_*.json.gz files
  abo-images-small/images/
    metadata/images.csv.gz          # image metadata
    small/                          # downscaled image files
```

## Usage

### Ingest products into LanceDB

Embeddings are computed and written to LanceDB in batches, so progress is saved incrementally.

```bash
# Full ingestion (~145K products)
uv run python ingest.py

# Smoke test with a subset
uv run python ingest.py --limit 200

# Larger batches (default is 32)
uv run python ingest.py --batch-size 128
```

The output LanceDB directory is `./abo-products-lance` by default (override with `--output-root`).

### Text-semantic search

Search the catalog by text using the `text_vec` embedding column:

```bash
uv run python text_search.py
```

This encodes the text query with the `intfloat/multilingual-e5-base` multilingual embedding model and runs a top-k vector search over the Lance table via DuckDB.

### Create a DuckDB sales table

Before running the image search, generate a local `sales.duckdb` file with synthetic sales rows linked to shoe products in the Lance catalog:

```bash
uv run python create_duckdb.py
```

This reads shoe `item_id`s from the Lance table and creates a `sales` table with 100 randomly assigned purchases. The image search script joins against this table.

### Image-semantic search with sales join

Search by visual concept using the CLIP `multimodal_vec` column and join results to the `sales` table:

```bash
uv run python image_search.py
```

This encodes the query `"beige shoes"` with CLIP, retrieves the nearest products, and joins them to `sales.duckdb`.

### Query with the DuckDB CLI

You can also query the Lance table directly from the DuckDB CLI:

```bash
duckdb
```

```sql
INSTALL lance;
LOAD lance;
ATTACH './abo-products-lance' AS abo (TYPE LANCE);

SELECT item_id, title, brand, product_type
FROM abo.main.products
LIMIT 10;
```
