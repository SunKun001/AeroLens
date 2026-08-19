# %%
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# ============================================================
# 1. LOAD CREDENTIALS
# ============================================================
# Absolute path so this runs the same whether launched from the
# project root or from inside scripts/.
load_dotenv(r"C:\Users\sunee\flight-price-analyst\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing — check .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# %%
# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv(r"C:\Users\sunee\flight-price-analyst\data\flights_for_db.csv")
print("Loaded shape:", df.shape)

# Postgres/PostgREST rejects NaN (it's not valid JSON) — convert to None
# so those cells become SQL NULL instead of failing the insert.
df = df.where(pd.notnull(df), None)
records = df.to_dict(orient="records")

# %%
# ============================================================
# 3. IDEMPOTENCY GUARD
# ============================================================
# Re-running this script shouldn't silently double the table. A count-only
# query (head=True) checks existing rows without pulling data.
#
# This is a hard stop, not a prompt: the CSV now carries dist_bucket and
# price_outlier (added for contextual outlier detection), so inserting on
# top of old 19-column rows would leave a mixed-schema table rather than a
# clean reload. Truncate first, then re-run.
existing = supabase.table("flights").select("id", count="exact", head=True).execute()
existing_count = existing.count or 0

if existing_count > 0:
    print(f"'flights' already has {existing_count} rows — refusing to insert on top of them.")
    print("Truncate it first in the Supabase SQL Editor, then re-run this script:")
    print("  TRUNCATE TABLE flights RESTART IDENTITY;")
    raise SystemExit(1)

# %%
# ============================================================
# 4. BATCH INSERT
# ============================================================
# A single 93k-row insert times out — Supabase/PostgREST batches
# comfortably in chunks of ~1000.
BATCH = 1000
total = len(records)
failed_batches = []

for i in range(0, total, BATCH):
    batch = records[i:i + BATCH]
    try:
        supabase.table("flights").insert(batch).execute()
        print(f"Inserted {min(i + BATCH, total)} / {total}")
    except Exception as e:
        # Don't let one bad batch kill the whole run — log it and move on
        # so partial progress survives and the failure is easy to retry.
        batch_num = i // BATCH
        print(f"FAILED batch {batch_num} (rows {i}-{min(i + BATCH, total) - 1}): {e}")
        failed_batches.append(batch_num)

if failed_batches:
    print(f"\n{len(failed_batches)} batch(es) failed: {failed_batches}")
else:
    print("\nAll batches inserted successfully.")

# %%
# ============================================================
# 5. VERIFY
# ============================================================
result = supabase.table("flights").select("id", count="exact", head=True).execute()
print(f"\nRow count in 'flights' table: {result.count}")
