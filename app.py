import streamlit as st
from groq import Groq
from supabase import create_client, Client
import base64
from PIL import Image
import io
import time
import json
import pandas as pd
import re
import uuid
import os

# --- Page configuration ---
st.set_page_config(
    page_title="Supermarket Ticket OCR",
    page_icon="🛒",
    layout="centered"
)

# --- Groq client ---
@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)

# --- Supabase config ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ypbyulzvcajbxmfojqla.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "sb_publishable_q_lvS59UFMMfLN0mAj6LDw_1fAIuYZn")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client (anon)
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Admin client (optional) for bucket creation
@st.cache_resource
def get_supabase_admin_client():
    if not SUPABASE_SERVICE_ROLE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Storage bucket name used by the app
BUCKET_NAME = "receipt-images"

def optimize_image(image, max_size=(1024, 1024)):
    """Resize image if it's too large to reduce processing time"""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_bytes(image, max_size=(1024, 1024), jpeg_quality=85):
    """Convert PIL Image to bytes and return (bytes, mime_type)."""
    # Make a copy so we don't mutate the original image object
    img = image.copy()
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    # Use JPEG for smaller file size if image is RGB, otherwise PNG
    if img.mode == 'RGB':
        img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=True)
        return buffered.getvalue(), "image/jpeg"
    img.save(buffered, format="PNG")
    return buffered.getvalue(), "image/png"


def image_bytes_to_base64(image_bytes, mime_type):
    """Convert image bytes to base64 data URL."""
    img_str = base64.b64encode(image_bytes).decode()
    return f"data:{mime_type};base64,{img_str}"


def upload_with_fallbacks(supabase_client, image):
    """Try uploading with progressively smaller sizes if upload fails."""
    attempts = [
        ((800, 800), 70),
        ((600, 600), 60),
        ((480, 480), 50),
    ]
    last_error = None
    for max_size, quality in attempts:
        upload_bytes, upload_mime = image_to_bytes(
            image, max_size=max_size, jpeg_quality=quality
        )
        image_url, upload_error = upload_image_to_supabase(
            supabase_client,
            upload_bytes,
            upload_mime,
        )
        if not upload_error:
            return image_url, None
        last_error = upload_error
        # Only retry smaller if timeout/connection reset
        lower_err = upload_error.lower()
        if "timed out" not in lower_err and "timeout" not in lower_err and "10054" not in lower_err:
            break
    return None, last_error

def analyze_image_with_groq(client, image_base64):
    """Analyze image using Groq API with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: Starting API call (Attempt {attempt + 1}/{max_retries})")
            print(f"{'='*60}")
            
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this supermarket receipt/ticket. Extract all products with their details. For each product, provide:\n- Product Name\n- Product Quantity (if available, otherwise put 1)\n- Product Price\n\nReturn the response as a JSON object with this structure:\n{\n  \"store_name\": \"store name\",\n  \"date\": \"date\",\n  \"total\": \"total amount\",\n  \"products\": [\n    {\"product_name\": \"item name\", \"quantity\": \"quantity\", \"price\": \"price\"},\n    ...\n  ]\n}\n\nIf you cannot find quantity for a product, use \"1\" as default. Ensure all prices are in the same currency format as shown on the receipt."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_completion_tokens=2048,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            print(f"\nDEBUG: API call successful!")
            print(f"DEBUG: Response object type: {type(completion)}")
            print(f"DEBUG: Number of choices: {len(completion.choices)}")
            
            if completion.choices:
                message = completion.choices[0].message
                print(f"DEBUG: Message object: {message}")
                print(f"DEBUG: Message content type: {type(message.content)}")
                print(f"\n{'='*60}")
                print(f"DEBUG: AI RESPONSE CONTENT:")
                print(f"{'='*60}")
                print(message.content)
                print(f"{'='*60}\n")
                
                return message.content
            else:
                print("DEBUG: ERROR - No choices in response!")
                return "Error: No response from AI"
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"\n{'='*60}")
            print(f"DEBUG: ERROR CAUGHT (Attempt {attempt + 1})")
            print(f"{'='*60}")
            print(f"Error Type: {error_type}")
            print(f"Error Message: {error_msg}")
            print(f"{'='*60}\n")
            
            if attempt < max_retries - 1:
                print(f"DEBUG: Retrying in 2 seconds...")
                time.sleep(2)
                continue
            
            # Return detailed error message
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                return f"Error: Request timed out. This can happen if:\n- The image is too large (try a smaller image)\n- The API is experiencing high load\n- Network connectivity issues\n\nPlease try again or use a smaller image."
            return f"Error analyzing image: {error_msg}"

def parse_products_from_response(response_text):
    """Parse products from AI response (JSON or text format)"""
    products = []
    
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            if isinstance(data, dict) and 'products' in data:
                products = data['products']
                print(f"DEBUG: Parsed {len(products)} products from JSON")
                return products, data
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON parsing failed: {e}")
        print(f"DEBUG: Trying to extract products from text format...")
    
    # Fallback: Try to extract products from text format
    # Look for patterns like "item name - quantity - price" or similar
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Try to extract product info from various formats
            # This is a basic fallback - AI should ideally return JSON
            if any(keyword in line.lower() for keyword in ['product', 'item', '$', '€', '£']):
                # Simple pattern matching as fallback
                parts = re.split(r'[-|\t|,]+', line)
                if len(parts) >= 2:
                    try:
                        product_name = parts[0].strip()
                        # Try to find price (last number)
                        price_match = re.search(r'[\d,]+\.?\d*', parts[-1])
                        price = price_match.group() if price_match else "0"
                        quantity = "1"  # Default quantity
                        
                        if len(parts) >= 3:
                            qty_match = re.search(r'\d+', parts[1])
                            quantity = qty_match.group() if qty_match else "1"
                        
                        products.append({
                            "product_name": product_name,
                            "quantity": quantity,
                            "price": price
                        })
                    except:
                        continue
    
    print(f"DEBUG: Extracted {len(products)} products from text parsing")
    return products, {}

def ensure_bucket_exists(supabase_client):
    """Ensure storage bucket exists; try to create if missing."""
    try:
        buckets = supabase_client.storage.list_buckets()
        if any(b.get("name") == BUCKET_NAME for b in buckets):
            return None
    except Exception as e:
        return f"Error listing buckets: {e}"

    try:
        supabase_client.storage.create_bucket(BUCKET_NAME, public=True)
        return None
    except Exception as e:
        return f"Error creating bucket '{BUCKET_NAME}': {e}"


def upload_image_to_supabase(supabase_client, image_bytes, mime_type, file_name=None):
    """Upload image to Supabase Storage and return public URL"""
    # Generate unique filename if not provided
    if not file_name:
        file_ext = "jpg" if mime_type == "image/jpeg" else "png"
        file_name = f"{uuid.uuid4()}.{file_ext}"

    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print("DEBUG: Uploading image to Supabase Storage")
            print(f"File name: {file_name}")
            print(f"Attempt: {attempt + 1}/{max_retries}")
            print(f"{'='*60}\n")

            supabase_client.storage.from_(BUCKET_NAME).upload(
                file_name,
                image_bytes,
                file_options={"content-type": mime_type}
            )

            # Get public URL
            public_url = supabase_client.storage.from_(BUCKET_NAME).get_public_url(file_name)

            print(f"DEBUG: Image uploaded successfully")
            print(f"DEBUG: Public URL: {public_url}")
            print(f"{'='*60}\n")

            return public_url, None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue

            error_msg = str(e)
            error_str = str(e)
            print(f"\n{'='*60}")
            print(f"DEBUG: Image upload error: {error_msg}")
            print(f"DEBUG: Error type: {type(e)}")
            print(f"{'='*60}\n")

            # Check for RLS policy error
            if (
                "row-level security" in error_str.lower()
                or "rls" in error_str.lower()
                or "403" in error_str
                or "unauthorized" in error_str.lower()
            ):
                return None, (
                    f"RLS Policy Error: The storage bucket '{BUCKET_NAME}' needs proper "
                    f"RLS policies. Go to Supabase Dashboard > Storage > {BUCKET_NAME} "
                    "bucket > Policies, and create a policy that allows INSERT operations "
                    "for authenticated/anonymous users."
                )

            # If bucket doesn't exist, provide helpful message
            if "bucket" in error_msg.lower() and (
                "not found" in error_msg.lower() or "does not exist" in error_msg.lower()
            ):
                return None, (
                    f"Storage bucket '{BUCKET_NAME}' not found. Please create it in "
                    "Supabase Storage settings (make it public)."
                )

            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                return None, (
                    "Error uploading image: The upload timed out. "
                    "Try again or use a smaller image."
                )
            return None, f"Error uploading image: {error_msg}"

def save_products_to_supabase(supabase_client, products, store_name=None, date=None, total=None, image_url=None):
    """Save products to Supabase table"""
    if not products:
        return None, "No products to save"
    
    # Prepare data for insertion
    insert_data = []
    for i, product in enumerate(products):
        product_data = {
            "product_name": str(product.get("product_name", "")).strip(),
            "product_quantity": str(product.get("quantity", "1")).strip(),
            "product_price": str(product.get("price", "0")).strip()
        }
        if store_name:
            product_data["store_name"] = str(store_name).strip()
        if date:
            product_data["receipt_date"] = str(date).strip()
        if total:
            product_data["total_amount"] = str(total).strip()
        if image_url:
            product_data["image_url"] = str(image_url).strip()
        insert_data.append(product_data)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print("DEBUG: Starting Supabase save operation")
            print(f"Attempt: {attempt + 1}/{max_retries}")
            print(f"Number of products: {len(insert_data)}")
            print(f"Store name: {store_name}")
            print(f"Date: {date}")
            print(f"Total: {total}")
            print(f"Image URL: {image_url}")
            print(f"{'='*60}\n")

            # Insert products into Supabase table
            response = supabase_client.table("receipt_products").insert(insert_data).execute()

            print(f"\nDEBUG: Supabase response received")
            print(f"DEBUG: Response type: {type(response)}")
            print(f"DEBUG: Has data attribute: {hasattr(response, 'data')}")

            if hasattr(response, 'data') and response.data:
                print(f"DEBUG: Successfully inserted {len(response.data)} records")
                print(f"DEBUG: First inserted record: {response.data[0] if response.data else 'None'}")
                print(f"{'='*60}\n")
                return response.data, None
            else:
                print(f"DEBUG: WARNING - Response has no data attribute or data is empty")
                print(f"DEBUG: Response object: {response}")
                print(f"{'='*60}\n")
                return None, "Insert completed but no data returned. Check Supabase dashboard to verify."

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            # final failure
            print(f"\n{'='*60}")
            print(f"DEBUG: Supabase save error")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {error_msg}")
            print(f"{'='*60}\n")

            if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                return None, (
                    "Table 'receipt_products' does not exist. "
                    "Please create it in Supabase with columns: product_name, "
                    "product_quantity, product_price, store_name, receipt_date, "
                    "total_amount, image_url"
                )
            if "permission denied" in error_msg.lower() or "policy" in error_msg.lower():
                return None, f"Permission denied. Check RLS policies in Supabase. Error: {error_msg}"
            return None, f"Error saving to Supabase: {error_msg}"

    return None, "Error saving to Supabase."

# --- Main app UI ---
st.title("🛒 Supermarket Ticket OCR")
st.markdown("Upload a supermarket receipt or ticket to extract information using AI")

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'image_url' not in st.session_state:
    st.session_state.image_url = None
if 'uploaded_file_data' not in st.session_state:
    st.session_state.uploaded_file_data = None
if 'products_df' not in st.session_state:
    st.session_state.products_df = None
if 'raw_response' not in st.session_state:
    st.session_state.raw_response = None

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'webp'],
    help="Upload a clear image of your supermarket receipt"
)

if uploaded_file is not None:
    # Store uploaded file in session state
    st.session_state.uploaded_file_data = uploaded_file
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Optimize and convert image for both upload and AI analysis
    image = optimize_image(image)
    # Use a slightly larger version for AI analysis
    image_bytes, image_mime = image_to_bytes(image, max_size=(1024, 1024), jpeg_quality=85)
    image_base64 = image_bytes_to_base64(image_bytes, image_mime)
    # Use a smaller, more compressed version for storage upload
    upload_bytes, upload_mime = image_to_bytes(image, max_size=(800, 800), jpeg_quality=70)
    
    # Analyze button
    if st.button("🔍 Analyze Receipt", type="primary"):
        print(f"\n{'='*60}")
        print("DEBUG: Analyze button clicked")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Base64 string length: {len(image_base64)}")
        print(f"{'='*60}\n")
        
        with st.spinner("Analyzing receipt with AI... This may take 30-60 seconds."):
            # Upload image to Supabase Storage first
            supabase_client = get_supabase_client()
            admin_client = get_supabase_admin_client()

            # Try to create bucket if missing (requires service role key)
            if admin_client:
                bucket_error = ensure_bucket_exists(admin_client)
                if bucket_error:
                    st.warning(f"⚠️ Bucket setup issue: {bucket_error}")
            else:
                st.info(
                    f"ℹ️ To auto-create the storage bucket, set "
                    f"the SUPABASE_SERVICE_ROLE_KEY environment variable."
                )

            # Upload optimized image bytes to storage (smaller + faster)
            image_url, upload_error = upload_with_fallbacks(
                supabase_client,
                image
            )
            
            if upload_error:
                st.warning(f"⚠️ Image upload failed: {upload_error}")
                st.info("Continuing with analysis, but image won't be saved to storage.")
                image_url = None
            else:
                st.success("✅ Image uploaded to storage!")
            
            # Analyze with Groq
            client = get_groq_client()
            result = analyze_image_with_groq(client, image_base64)
            
            print(f"\n{'='*60}")
            print("DEBUG: Function returned result")
            print(f"Result type: {type(result)}")
            print(f"Result length: {len(str(result)) if result else 0}")
            print(f"{'='*60}\n")
            
            # Check if result is an error
            if result and result.startswith("Error"):
                print("DEBUG: Result is an error, displaying error message")
                st.error(result)
            else:
                # Display results
                print("DEBUG: Result is valid, displaying success")
                st.success("Analysis Complete!")
                
                # Parse products from response
                products, metadata = parse_products_from_response(result)
                
                # Store in session state
                st.session_state.products = products
                st.session_state.metadata = metadata
                st.session_state.image_url = image_url
                st.session_state.analysis_complete = True
                
                # Store raw response for fallback display
                st.session_state.raw_response = result
    
else:
    st.info("👆 Please upload an image file to get started")

# Results + edit section (persists across reruns)
if st.session_state.analysis_complete:
    st.divider()
    st.markdown("### ✅ Extraction Results")

    metadata = st.session_state.metadata
    products = st.session_state.products

    # Display metadata if available
    if metadata:
        col1, col2, col3 = st.columns(3)
        if 'store_name' in metadata:
            col1.metric("Store", metadata.get('store_name', 'N/A'))
        if 'date' in metadata:
            col2.metric("Date", metadata.get('date', 'N/A'))
        if 'total' in metadata:
            col3.metric("Total", metadata.get('total', 'N/A'))

    # Display editable products table
    if products:
        st.markdown("### 📦 Products")
        if st.session_state.products_df is None:
            st.session_state.products_df = pd.DataFrame([{
                "Product Name": p.get("product_name", ""),
                "Product Quantity": p.get("quantity", "1"),
                "Product Price": p.get("price", "0")
            } for p in products])

        st.info("✏️ You can edit the extracted data before saving to Supabase.")
        edited_df = st.data_editor(
            st.session_state.products_df,
            use_container_width=True,
            num_rows="dynamic",
            key="products_editor",
        )
        st.session_state.products_df = edited_df
    else:
        st.warning("⚠️ No products were extracted. Showing full response:")
        if st.session_state.raw_response:
            st.markdown(st.session_state.raw_response)
            st.download_button(
                label="📥 Download Full Results (Text)",
                data=st.session_state.raw_response,
                file_name="receipt_analysis.txt",
                mime="text/plain"
            )

    # Save section AFTER extraction/editing
    st.divider()
    st.markdown("### 💾 Save to Database")

    products_df = st.session_state.products_df
    if products_df is not None:
        st.info(f"📦 {len(products_df)} products ready to save")
    else:
        st.info(f"📦 {len(st.session_state.products)} products ready to save")

    if st.button("💾 Save to Supabase", type="primary", key="save_to_supabase_main"):
        with st.spinner("Saving products to Supabase..."):
            try:
                supabase_client = get_supabase_client()

                # Get metadata values from session state
                metadata = st.session_state.metadata

                store_name_val = metadata.get('store_name') if metadata and isinstance(metadata, dict) else None
                date_val = metadata.get('date') if metadata and isinstance(metadata, dict) else None
                total_val = metadata.get('total') if metadata and isinstance(metadata, dict) else None

                # Use edited data if available
                if st.session_state.products_df is not None:
                    edited_df = st.session_state.products_df.fillna("")
                    products = [
                        {
                            "product_name": str(row.get("Product Name", "")).strip(),
                            "quantity": str(row.get("Product Quantity", "")).strip() or "1",
                            "price": str(row.get("Product Price", "")).strip(),
                        }
                        for _, row in edited_df.iterrows()
                        if str(row.get("Product Name", "")).strip()
                    ]
                else:
                    products = st.session_state.products

                # Get image URL from session state
                image_url_val = st.session_state.image_url

                print(f"DEBUG: Calling save_products_to_supabase with:")
                print(f"  - Products count: {len(products)}")
                print(f"  - Store name: {store_name_val}")
                print(f"  - Date: {date_val}")
                print(f"  - Total: {total_val}")
                print(f"  - Image URL: {image_url_val}")
                print(f"  - Products: {products}")

                saved_data, error = save_products_to_supabase(
                    supabase_client,
                    products,
                    store_name=store_name_val,
                    date=date_val,
                    total=total_val,
                    image_url=image_url_val
                )

                if error:
                    st.error(f"❌ {error}")
                    st.error("Check the terminal/console for detailed error information.")
                else:
                    if saved_data:
                        st.success(f"✅ Successfully saved {len(saved_data)} products to Supabase!")
                        st.info("💡 Tip: You can view the data in your Supabase dashboard under the 'receipt_products' table.")
                        with st.expander("📋 View saved data"):
                            st.json(saved_data[:3] if len(saved_data) > 3 else saved_data)
                    else:
                        st.warning("⚠️ Save operation completed but no data was returned. Please check Supabase dashboard.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                print(f"DEBUG: Unexpected error in button handler: {e}")
                import traceback
                traceback.print_exc()

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Upload Image**: Click on the upload area and select your receipt image
    2. **Analyze**: Click the "Analyze Receipt" button
    3. **View Results**: The extracted information will be displayed below
    4. **Save to Supabase**: Click "Save to Supabase" to store products in database
    
    **Supported formats**: PNG, JPG, JPEG, WEBP
    
    **Tips for best results**:
    - Use clear, well-lit images
    - Ensure the receipt is fully visible
    - Avoid blurry or rotated images
    """)

    st.header("📊 Analytics")
    st.markdown("[Open Analytics](/analyse)")

    st.header("🔧 About")
    st.markdown("""
    This app uses Groq's LLM API with vision capabilities to extract information from supermarket receipts.
    
    **Model**: meta-llama/llama-4-scout-17b-16e-instruct
    
    **Storage Setup**:
    - Bucket name: `receipt-images` (used by the app)
    - Make the bucket **Public**
    - Go to **Policies** tab and create a policy:
      - Policy name: "Allow public uploads"
      - Allowed operation: INSERT
      - Target roles: anon, authenticated
      - USING expression: `true`
      - WITH CHECK expression: `true`
    - Optional: set `SUPABASE_SERVICE_ROLE_KEY` env var to let the app create the bucket automatically
    """)
