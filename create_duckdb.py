import duckdb
import random

random.seed(23)
LIMIT = 100

con = duckdb.connect("./sales.duckdb")
con.execute("INSTALL lance")
con.execute("LOAD lance")
con.execute("ATTACH './abo-products-lance' AS abo (TYPE LANCE)")

shoe_ids = [
    row[0]
    for row in con.execute(
        """
        SELECT item_id
        FROM abo.main.products
        WHERE product_type = 'SHOES'
        LIMIT 25
        """
    ).fetchall()
]

sales_rows = [(i, random.choice(shoe_ids)) for i in range(1, LIMIT + 1)]

con.execute("CREATE OR REPLACE TABLE sales (id INTEGER, item_id VARCHAR)")
con.executemany("INSERT INTO sales VALUES (?, ?)", sales_rows)