# Supermarket Ticket OCR App

A Streamlit application that uses Groq's LLM API to extract information from supermarket receipts and tickets using AI vision capabilities.

## Features

- 📸 Upload receipt images (PNG, JPG, JPEG, WEBP)
- 🤖 AI-powered text extraction using Groq's vision model
- 📋 Structured information display
- ✏️ Editable extracted data before saving
- 🗄️ Save products + image URL to Supabase

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Upload a supermarket receipt image

4. Click "Analyze Receipt" to extract information

5. Edit extracted products in the table

6. Click "Save to Supabase" to store results

## Requirements

- Python 3.8+
- Streamlit
- Groq API key (set as `GROQ_API_KEY`)
- Pillow for image processing
- Supabase project (Storage bucket + table)

## Model

The app uses `meta-llama/llama-4-scout-17b-16e-instruct` model via Groq API, which supports vision/image analysis.

## Notes

- For best results, use clear, well-lit images
- Ensure the receipt is fully visible in the image
- Set `GROQ_API_KEY` in your environment before running the app


## Function Reference (Detailed)

Below is a detailed explanation of each main function in `app.py` and how it is used.

### `get_groq_client()`
Creates and caches a Groq client instance used for all LLM calls.
- Uses `@st.cache_resource` so the client is created once per app session.
- Reads the Groq API key from `GROQ_API_KEY` and returns a ready client.
- Used inside the Analyze flow to call the vision model.

### `get_supabase_client()` / `get_supabase_admin_client()`
- `get_supabase_client()` creates a Supabase client with the **anon key** for normal app usage.
- `get_supabase_admin_client()` creates an optional **service-role** client if `SUPABASE_SERVICE_ROLE_KEY` is set.
- The admin client is used only to auto-create the storage bucket if needed.
- Both are cached to avoid reconnecting on every Streamlit rerun.

### `optimize_image(image, max_size)`
- Resizes large receipt images to a maximum resolution (default 1024×1024).
- Reduces payload size before sending to Groq and storage.
- Uses high-quality LANCZOS downsampling for better OCR clarity.

### `image_to_bytes(image, max_size, jpeg_quality)`
- Converts a PIL image into raw bytes + MIME type.
- Applies optional resize and JPEG compression settings.
- Returns `(bytes, mime_type)` so the same output can be used for:
  - Storage upload
  - Base64 encoding for Groq

### `image_bytes_to_base64(image_bytes, mime_type)`
- Encodes the image bytes into a `data:` URL string.
- This is the format required by Groq’s vision API.
- Example output: `data:image/jpeg;base64,....`

### `upload_image_to_supabase(supabase_client, image_bytes, mime_type, file_name)`
- Uploads image bytes to the Supabase Storage bucket (`receipt-images`).
- Generates a unique filename if one isn’t provided.
- Returns a **public URL** for storing in the database.
- Includes retry logic and error handling for RLS or timeout errors.

### `upload_with_fallbacks(supabase_client, image)`
- Attempts multiple compression/size levels if upload fails.
- Starts with 800×800 @ 70% quality, then drops to 600×600 and 480×480.
- Prevents frequent upload failures on slow or unstable connections.

### `parse_products_from_response(response_text)`
- Attempts to parse the LLM response as JSON using the expected schema.
- If JSON is invalid, falls back to text parsing with regex heuristics.
- Returns:
  - `products` list (normalized product_name / quantity / price)
  - `metadata` (store name, date, total) if available

### `save_products_to_supabase(supabase_client, products, store_name, date, total, image_url)`
- Normalizes product rows into database columns.
- Inserts the rows into `receipt_products`.
- Retries on transient connection errors.
- Returns saved rows (if Supabase returns them) and an error string (if any).

## Code Walkthrough

### 1) Initialization and Configuration
- Sets Streamlit page config.
- Configures Supabase URL/key from environment variables (with defaults).
- Creates cached clients for Groq and Supabase.
- Initializes session state for extracted data and edit table.

### 2) Image Upload + Processing
- User uploads a receipt image in Streamlit.
- Image is resized to improve speed and reduce bandwidth.
- A smaller version is used for storage upload, and a slightly larger one for AI analysis.

### 3) Storage Upload (Supabase)
- Image upload is attempted with retry + fallback sizes.
- On success, a public URL is generated and saved in session state.
- On failure, the app continues analysis without storage URL.

### 4) OCR + Data Extraction (Groq)
- Image is sent to Groq vision model with a structured JSON prompt.
- Response is parsed into products and metadata.
- Results are stored in session state.

### 5) Editable Table
- Extracted products are shown in an editable table.
- User can add/remove/edit rows before saving.

### 6) Save to Supabase
- Edited rows are normalized into database fields.
- Rows are inserted into `receipt_products` with optional metadata and `image_url`.

## Supabase Setup

### Table
Ensure a table named `receipt_products` exists with columns:
- `product_name` (text)
- `product_quantity` (text)
- `product_price` (text)
- `store_name` (text, nullable)
- `receipt_date` (text, nullable)
- `total_amount` (text, nullable)
- `image_url` (text, nullable)
- `created_at` (timestamp, default `now()`)

### Storage Bucket
Create a public bucket named `receipt-images` and add policies:
- **INSERT** for `anon` + `authenticated`
- **SELECT** for `anon` + `authenticated`

This enables image uploads and public URL access.
