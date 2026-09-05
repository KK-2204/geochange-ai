"""GeoChange AI — Streamlit UI (Phase 1: interface + placeholders)."""
from google import genai
import streamlit as st
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
import cv2
import numpy as np
import zipfile
import io
import re

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="GeoChange AI", page_icon="🛰️", layout="wide")

# ============================================================
# Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg: #0A1220;
    --panel: #111C2E;
    --border: #24344C;
    --text: #EAF0F8;
    --muted: #7E90AC;
    --water: #4CC9E8;
    --veg: #4FAE7A;
    --built: #E0A458;
}

/* Hide default Streamlit chrome for a finished, branded feel.
   Comment this block out if you need the menu/deploy button back while debugging. */
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(circle at 12% -10%, #10203A 0%, var(--bg) 55%);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.block-container { max-width: 1040px; padding-top: 2.6rem; padding-bottom: 3rem; }

h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    letter-spacing: -0.01em;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
    color: var(--text) !important;
}

.gc-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
}
.gc-status .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--water);
    box-shadow: 0 0 8px var(--water);
}

.gc-sub {
    color: var(--muted) !important;
    font-size: 1.03rem;
    max-width: 50ch;
    margin-bottom: 1rem !important;
}

.gc-panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.02em;
    margin: 1.6rem 0 0.5rem 2px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.4rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--water) !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--muted) !important; }

/* Image preview */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* Primary button */
[data-testid="stButton"] button {
    background: var(--water) !important;
    color: #06141F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 0 0 3px rgba(76, 201, 232, 0.25);
    transform: translateY(-1px);
}

/* Alerts / spinner text */
[data-testid="stAlert"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--water) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

def load_preview_image(uploaded_file):
    """Safely loads both standard images and multi-band GeoTIFFs into a PIL preview."""
    uploaded_file.seek(0)
    try:
        img = Image.open(uploaded_file)
        # Prevent PIL from passing Float32 ('F') or 16-bit ('I') TIFFs directly to Streamlit
        if img.mode not in ['RGB', 'RGBA', 'L']:
            raise ValueError("Unsupported PIL mode, forcing rasterio fallback")
        return img
    except Exception:
        # Fall back to rasterio for multi-band/Float32 GeoTIFFs
        uploaded_file.seek(0)
        with MemoryFile(uploaded_file) as memfile:
            with memfile.open() as src:
                if src.count >= 3:
                    # S2 band layout: B2=Blue (1), B3=Green (2), B4=Red (3)
                    b = src.read(1).astype(float)
                    g = src.read(2).astype(float)
                    r = src.read(3).astype(float)
                    rgb = np.dstack((r, g, b))
                    # Percentile stretch for clean visual rendering
                    p2, p98 = np.nanpercentile(rgb, (2, 98))
                    rgb_norm = np.clip(255 * (rgb - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    uploaded_file.seek(0)
                    return Image.fromarray(rgb_norm)
                elif src.count == 1:
                    # S1 SAR layout: 1 band (VV backscatter)
                    vv = src.read(1).astype(float)
                    p2, p98 = np.nanpercentile(vv, (2, 98))
                    vv_norm = np.clip(255 * (vv - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    uploaded_file.seek(0)
                    return Image.fromarray(np.dstack((vv_norm, vv_norm, vv_norm)))


def extract_from_zip(uploaded_file):
    """Extracts, identifies by band name, and stacks images from a ZIP."""
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
            files = [f for f in z.namelist() if f.lower().endswith(valid_exts) and not f.startswith('__MACOSX')]
            
            if not files:
                st.error("No valid images found in the ZIP.")
                return None
                
            # 1. Map files to specific band requirements using regex
            band_map = {'B2': None, 'B3': None, 'B4': None, 'B8': None, 'B11': None}
            
            for f in files:
                f_lower = f.lower()
                # Matches patterns like "_b02.", "-b2.", "band02", etc.
                if re.search(r'b0?2[._-]', f_lower): band_map['B2'] = f
                elif re.search(r'b0?3[._-]', f_lower): band_map['B3'] = f
                elif re.search(r'b0?4[._-]', f_lower): band_map['B4'] = f
                elif re.search(r'b0?8[._-]', f_lower): band_map['B8'] = f
                elif re.search(r'b11[._-]', f_lower): band_map['B11'] = f

            # 2. If the ZIP contains at least B2, B3, B4, and B8, stack them
            if all(band_map[k] for k in ['B2', 'B3', 'B4', 'B8']):
                bands_to_stack = [band_map['B2'], band_map['B3'], band_map['B4'], band_map['B8']]
                if band_map['B11']:
                    bands_to_stack.append(band_map['B11'])
                    
                bands_data = []
                base_meta = None
                
                for fname in bands_to_stack:
                    # rasterio reads both TIFF and PNG/JPG arrays seamlessly
                    with MemoryFile(z.read(fname)) as memfile:
                        with memfile.open() as src:
                            bands_data.append(src.read(1))
                            if base_meta is None:
                                base_meta = src.meta.copy()
                                
                # Override metadata to force it into a unified GeoTIFF structure
                base_meta.update(
                    driver='GTiff',
                    count=len(bands_data),
                    dtype=bands_data[0].dtype 
                )
                
                with MemoryFile() as memfile:
                    with memfile.open(**base_meta) as dest:
                        for i, band in enumerate(bands_data, start=1):
                            dest.write(band, i)
                    stacked_bytes = memfile.read()
                    
                stacked_obj = io.BytesIO(stacked_bytes)
                # By naming it .tif, we trick the app into using the correct multi-spectral pipeline
                stacked_obj.name = "stacked_bands.tif"
                return stacked_obj
                
            # 3. Fallback: If it's just a single composite image inside the ZIP
            files.sort(key=lambda x: (not x.lower().endswith('.tif'), not x.lower().endswith('.png')))
            file_obj = io.BytesIO(z.read(files[0]))
            file_obj.name = files[0]
            return file_obj
            
    return uploaded_file

def process_raw_satellite(uploaded_file, name_label="Image"):
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()
    is_tiff = filename.endswith(('.tif', '.tiff'))
    
    quality_msg = None

    if not is_tiff:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert('RGB')
        
        # Check resolution for standard images
        if img.width < 800 or img.height < 800:
            quality_msg = f"💡 **Tip for {name_label}:** Low resolution detected. For better AI accuracy, use higher resolution exports."
            
        rgb = np.array(img)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        total = rgb.shape[0] * rgb.shape[1]
        
        veg_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))) / total) * 100
        water_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([85, 40, 40]), np.array([130, 255, 255]))) / total) * 100
        built_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 255]))) / total) * 100
        
        return water_pct, veg_pct, built_pct, Image.fromarray(rgb), quality_msg

    else:
        with MemoryFile(uploaded_file) as memfile:
            with memfile.open() as src:
                band_count = src.count
                dtype = str(src.dtypes[0])
                
                # Check bit-depth and resolution for TIFFs
                if 'uint8' in dtype or src.width < 800:
                    quality_msg = f"💡 **Tip for {name_label}:** 8-bit or lower resolution detected. The app can analyze this, but for maximum scientific accuracy, download **16-bit or 32-bit TIFFs** with high resolution."

                if band_count >= 4:
                    b, g = src.read(1).astype(float), src.read(2).astype(float)
                    r, nir = src.read(3).astype(float), src.read(4).astype(float)
                    np.seterr(divide='ignore', invalid='ignore')
                    
                    water_pct = (np.count_nonzero(((g - nir) / (g + nir + 1e-5)) > 0.1) / g.size) * 100
                    veg_pct = (np.count_nonzero(((nir - r) / (nir + r + 1e-5)) > 0.2) / g.size) * 100
                    
                    built_pct = None
                    if band_count >= 5:
                        swir = src.read(5).astype(float)
                        built_pct = (np.count_nonzero(((swir - nir) / (swir + nir + 1e-5)) > 0.0) / swir.size) * 100
                        
                    rgb = np.dstack((r, g, b))
                    p2, p98 = np.nanpercentile(rgb, (2, 98))
                    rgb_norm = np.clip(255 * (rgb - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    return water_pct, veg_pct, built_pct, Image.fromarray(rgb_norm), quality_msg
                    
                elif band_count == 3:
                    # (Keep the exact same 3-band math as before)
                    r, g, b = src.read(1), src.read(2), src.read(3)
                    rgb = np.dstack((r, g, b))
                    if rgb.dtype != np.uint8 or rgb.max() > 255:
                        p2, p98 = np.nanpercentile(rgb, (2, 98))
                        rgb = np.clip(255 * (rgb - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    total = rgb.shape[0] * rgb.shape[1]
                    veg_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))) / total) * 100
                    water_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([85, 40, 40]), np.array([130, 255, 255]))) / total) * 100
                    built_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 255]))) / total) * 100
                    return water_pct, veg_pct, built_pct, Image.fromarray(rgb), quality_msg
                    
                elif band_count == 1:
                    # (Keep the exact same 1-band SAR math as before)
                    vv = src.read(1).astype(float)
                    if 'uint8' in dtype:
                        water_pct, built_pct = None, None
                        vv_norm = vv.astype(np.uint8)
                    else:
                        water_pct = (np.count_nonzero(vv < -16) / vv.size) * 100
                        built_pct = None
                        p2, p98 = np.nanpercentile(vv, (2, 98))
                        vv_norm = np.clip(255 * (vv - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    return water_pct, None, built_pct, Image.fromarray(np.dstack((vv_norm, vv_norm, vv_norm))), quality_msg
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()
    is_tiff = filename.endswith(('.tif', '.tiff'))



    # ============================================================
    # PIPELINE A: Standard Web Images (PNG, JPG, JPEG)
    # These only contain 8-bit visual data (0-255), no spectral bands.
    # ============================================================
    if not is_tiff:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert('RGB') # Forces RGBA/Grayscale into standard RGB
        rgb = np.array(img)
        
        # Apply standard HSV visual color math
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        total = rgb.shape[0] * rgb.shape[1]
        
        veg_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))) / total) * 100
        water_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([85, 40, 40]), np.array([130, 255, 255]))) / total) * 100
        
        return water_pct, veg_pct, None, Image.fromarray(rgb)

    # ============================================================
    # PIPELINE B: Raw Geospatial Data (GeoTIFFs)
    # These contain floating-point physics data and invisible bands.
    # ============================================================
    else:
        with MemoryFile(uploaded_file) as memfile:
            with memfile.open() as src:
                band_count = src.count
                dtype = src.dtypes[0]
                
                # MODE 1: MULTI-SPECTRAL (4+ Bands: B, G, R, NIR, SWIR...)
                if band_count >= 4:
                    b, g = src.read(1).astype(float), src.read(2).astype(float)
                    r, nir = src.read(3).astype(float), src.read(4).astype(float)
                    np.seterr(divide='ignore', invalid='ignore')
                    
                    water_pct = (np.count_nonzero(((g - nir) / (g + nir + 1e-5)) > 0.1) / g.size) * 100
                    veg_pct = (np.count_nonzero(((nir - r) / (nir + r + 1e-5)) > 0.2) / g.size) * 100
                    
                    built_pct = None
                    if band_count >= 5:
                        swir = src.read(5).astype(float)
                        built_pct = (np.count_nonzero(((swir - nir) / (swir + nir + 1e-5)) > 0.0) / swir.size) * 100
                        
                    rgb = np.dstack((r, g, b))
                    p2, p98 = np.nanpercentile(rgb, (2, 98))
                    rgb_norm = np.clip(255 * (rgb - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                    return water_pct, veg_pct, built_pct, Image.fromarray(rgb_norm)
                    
                # MODE 2: TRUE COLOR TIFF (3 Bands)
                elif band_count == 3:
                    r, g, b = src.read(1), src.read(2), src.read(3)
                    rgb = np.dstack((r, g, b))
                    
                    if rgb.dtype != np.uint8 or rgb.max() > 255:
                        p2, p98 = np.nanpercentile(rgb, (2, 98))
                        rgb = np.clip(255 * (rgb - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                        
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    total = rgb.shape[0] * rgb.shape[1]
                    
                    veg_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))) / total) * 100
                    water_pct = (cv2.countNonZero(cv2.inRange(hsv, np.array([85, 40, 40]), np.array([130, 255, 255]))) / total) * 100
                    return water_pct, veg_pct, None, Image.fromarray(rgb)
                    
                # MODE 3: SAR or SINGLE BAND (1 Band)
                elif band_count == 1:
                    vv = src.read(1).astype(float)
                    
                    # If it's standard 8-bit, it's just a grayscale picture (No SAR math allowed)
                    if dtype == 'uint8':
                        water_pct = None
                        vv_norm = vv.astype(np.uint8)
                    else:
                        # It is floating-point radar data (SAR)
                        water_pct = (np.count_nonzero(vv < -16) / vv.size) * 100
                        p2, p98 = np.nanpercentile(vv, (2, 98))
                        vv_norm = np.clip(255 * (vv - p2) / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
                        
                    return water_pct, None, None, Image.fromarray(np.dstack((vv_norm, vv_norm, vv_norm)))
# ============================================================
# Header
# ============================================================
st.markdown('<div class="gc-status"><span class="dot"></span>Ready to scan</div>', unsafe_allow_html=True)
st.title("GeoChange AI")
st.markdown('<p class="gc-sub">Compare two images of the same place from different dates.</p>', unsafe_allow_html=True)

# ============================================================
# Side-by-side uploads
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="gc-panel-label">Baseline</div>', unsafe_allow_html=True)
    old_file = st.file_uploader("Upload baseline", type=['jpg', 'png', 'tiff', 'tif', 'zip'], key="old", label_visibility="collapsed")
with col2:
    st.markdown('<div class="gc-panel-label">Recent</div>', unsafe_allow_html=True)
    new_file = st.file_uploader("Upload recent", type=['jpg', 'png', 'tiff', 'tif', 'zip'], key="new", label_visibility="collapsed")

if old_file and new_file:
    # 1. Safely extract files if either upload is a ZIP
    processed_old = extract_from_zip(old_file)
    processed_new = extract_from_zip(new_file)
    
    if processed_old and processed_new:
        # 2. Preview using the extracted objects
        img1 = load_preview_image(processed_old)
        img2 = load_preview_image(processed_new)
        col1.image(img1, use_container_width=True)
        col2.image(img2, use_container_width=True)

        st.write("")
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            run = st.button("Find changes", use_container_width=True)

        if run:
            with st.spinner("Processing multi-sensor spectral data..."):
                # Pass a label so the tip tells them exactly which file to upgrade
                w1, v1, b1, img_base, msg1 = process_raw_satellite(processed_old, "Baseline")
                w2, v2, b2, img_recent, msg2 = process_raw_satellite(processed_new, "Recent")
                
                # Show quality suggestions (Combined to prevent duplicates)
                if msg1 and msg2:
                    st.info("💡 **Optimization Tip:** 8-bit or lower resolution detected in your uploads. The app can analyze these, but for maximum scientific accuracy, use **16-bit or 32-bit TIFFs** with high resolution.")
                elif msg1:
                    st.info(msg1)
                elif msg2:
                    st.info(msg2)
                 
           # --- CALCULATE DELTAS ---
            if w1 is not None and w2 is not None:
                w_diff = w2 - w1
                water_change = f"{'+' if w_diff > 0 else ''}{w_diff:.1f}%"
            else:
                water_change = "N/A"
                
            if v1 is not None and v2 is not None:
                veg_change = f"{'+' if (v2 - v1) > 0 else ''}{(v2 - v1):.1f}%"
            else:
                veg_change = "N/A"
                
            if b1 is not None and b2 is not None:
                built_change = f"{'+' if (b2 - b1) > 0 else ''}{(b2 - b1):.1f}%"
            else:
                built_change = "N/A"

            # Update the UI images
            col1.image(img_base, use_container_width=True)
            col2.image(img_recent, use_container_width=True)

            st.markdown('<div class="gc-panel-label">What did we find?</div>', unsafe_allow_html=True)

            # UI Metrics
            m1, m2, m3 = st.columns(3)
            metrics = [
                (m1, "Water", water_change, "var(--water)"),
                (m2, "Vegetation", veg_change, "var(--veg)"),
                (m3, "Built-up", built_change, "var(--built)"),
            ]
            for col, label, value, color in metrics:
                col.markdown(f"""
                <div style="background:var(--panel); border:1px solid var(--border); border-left:3px solid {color};
                            border-radius:10px; padding:1.1rem 1.2rem;">
                    <div style="color:var(--muted); font-size:0.85rem; margin-bottom:0.35rem;">{label}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:600; color:{color};">{value}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # Gemini API Call
            with st.spinner("Generating AI analysis..."):
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                
                prompt = f"""
                You are GeoChange AI. I am providing two satellite images (Baseline and Recent) and deterministically calculated pixel change metrics:
                - Water Coverage Change: {water_change}
                - Vegetation Coverage Change: {veg_change}
                - Built-up Area Change: {built_change}
                
                Analyze the images and the hard data. Output your response in exactly two sections separated by a "|||" delimiter.
                
                Section 1: Write a simple, engaging 3-sentence explanation of what happened here so a 10-year-old can easily understand it.
                Section 2: Write a dense, highly technical breakdown using geospatial terminology explaining the structural changes and impact.
                """
                
                response = client.models.generate_content(
                    model="gemini-3.6-flash",

                    contents=[img_base, img_recent, prompt]
                )
                
                try:
                    kid_friendly, technical = response.text.split("|||")
                except ValueError:
                    kid_friendly = response.text
                    technical = "Technical details unavailable. Please try again."

            # Display Kid-Friendly UI
            st.markdown(f"""
            <div style="background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--water);
                        border-radius:10px; padding:1.3rem 1.5rem; margin-top:0.4rem;">
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; margin-bottom:0.5rem;">AI explanation</div>
                <div style="line-height:1.6; font-size:0.98rem; color:var(--text);">{kid_friendly.strip()}</div>
            </div>
            """, unsafe_allow_html=True)

            # Display Expandable Technical Deep-Dive
            with st.expander("Show technical details"):
                st.write(technical.strip())