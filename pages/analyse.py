import os
import re

import pandas as pd
import streamlit as st
from supabase import create_client


# --- Page config ---
st.set_page_config(
    page_title="Receipt Analysis",
    page_icon="📊",
    layout="wide",
)

# --- Supabase config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ypbyulzvcajbxmfojqla.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "sb_publishable_q_lvS59UFMMfLN0mAj6LDw_1fAIuYZn")


@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def fetch_receipts():
    supabase = get_supabase_client()
    resp = supabase.table("receipt_products").select("*").order("created_at", desc=True).execute()
    data = resp.data or []
    error = getattr(resp, "error", None)
    return data, error


def parse_number(value):
    if value is None:
        return None
    text = str(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


st.title("📊 Receipt Analysis")
st.markdown("Overview of saved receipt items and full history from `receipt_products`.")

if st.button("🔄 Refresh Data"):
    st.rerun()

data, error = fetch_receipts()
if error:
    st.error(f"Supabase error: {error}")
st.caption(f"Loaded {len(data)} rows from `receipt_products`.")
if not data:
    st.info("No data found in `receipt_products`. Save receipts first from the main app.")
    st.stop()

df = pd.DataFrame(data)

# Normalize columns
df["product_quantity_num"] = df.get("product_quantity", "").apply(parse_number)
df["product_price_num"] = df.get("product_price", "").apply(parse_number)
df["total_line_value"] = df["product_quantity_num"].fillna(1) * df["product_price_num"].fillna(0)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Unique Products", df["product_name"].nunique())
col3.metric("Total Spend", f"{df['total_line_value'].sum():.2f}")
col4.metric("Unique Stores", df["store_name"].nunique())

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Top Products by Spend")
    top_products = (
        df.groupby("product_name", dropna=True)["total_line_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_products)

with right:
    st.subheader("Top Products by Quantity")
    top_qty = (
        df.groupby("product_name", dropna=True)["product_quantity_num"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_qty)

st.divider()

st.subheader("All Line Items")
display_cols = [
    "created_at",
    "store_name",
    "receipt_date",
    "product_name",
    "product_quantity",
    "product_price",
    "total_amount",
    "image_url",
]
cols_present = [c for c in display_cols if c in df.columns]
st.dataframe(df[cols_present], use_container_width=True, hide_index=True)

st.divider()

# Transactions summary
st.subheader("All Transactions")


def build_tx_key(row):
    if pd.notna(row.get("image_url")) and str(row.get("image_url")).strip():
        return f"img:{row.get('image_url')}"
    store = str(row.get("store_name") or "").strip()
    date = str(row.get("receipt_date") or "").strip()
    total = str(row.get("total_amount") or "").strip()
    created = str(row.get("created_at") or "").split("T")[0]
    return f"meta:{store}|{date}|{total}|{created}"


df["transaction_key"] = df.apply(build_tx_key, axis=1)

tx = (
    df.groupby("transaction_key", dropna=False)
    .agg(
        created_at=("created_at", "max"),
        store_name=("store_name", "first"),
        receipt_date=("receipt_date", "first"),
        total_amount=("total_amount", "first"),
        items_count=("product_name", "count"),
        items_sum=("total_line_value", "sum"),
        image_url=("image_url", "first"),
    )
    .reset_index(drop=False)
    .sort_values("created_at", ascending=False)
)

st.dataframe(
    tx[
        [
            "created_at",
            "store_name",
            "receipt_date",
            "total_amount",
            "items_count",
            "items_sum",
            "image_url",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
