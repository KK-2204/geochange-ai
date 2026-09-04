# GeoChange AI 🌍

An automated, multi-sensor geospatial AI tool designed for environmental monitoring and flood impact assessment. GeoChange AI processes raw satellite data to provide deterministic change metrics alongside accessible, AI-driven explanations.

### Key Features
* **Universal Sensor Engine:** Natively processes Copernicus multi-spectral TIFFs (16/32-bit), standard optical imagery (PNG/JPG), and Sentinel-1 SAR (Radar) backscatter data.
* **Automated Band Stacking:** Dynamically extracts, identifies, and stacks individual spectral bands (B02, B03, B04, B08) from ZIP archives into unified arrays.
* **Deterministic Geospatial Math:** Computes strict NDWI (Water), NDVI (Vegetation), and Built-up proxies using OpenCV and Rasterio to prevent AI hallucination.
* **Dual-Tier Insights:** Powered by Gemini Flash, providing both accessible summaries and dense technical breakdowns.

### Tech Stack
* **Frontend:** Streamlit
* **Geospatial Processing:** Rasterio, OpenCV, NumPy
* **AI Engine:** Google Gemini API
