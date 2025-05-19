# GeoOCR

**GeoOCR** is a Python library for automating OCR on historical archive images and extracting geospatial data. It provides:

* **ArchiveImage**: image preprocessing, splitting and OCR.
* **ArchiveCollection**: batch operations on collections of images.
* **ArchiveGeolocator**: fuzzy-match CSV addresses against a GeoJSON reference to geolocate.
* **GeoExporter**: export results to CSV or GeoJSON.

We recommend using it with Google Colab or Jupyter Notebook.

*Note: This project was developed for a specific university research project and is not guaranteed to be maintained.*

---

## Installation

```bash
pip install geocr
```

Or install from source:

```bash
git clone https://github.com/joshuavachon25/geocr-py.git
cd geocr
pip install .
```

---

## Quickstart
Normal usage involves creating a collection, batch-processing what you can, then applying specific pipelines to individual items. After that, you perform OCR, extract and correct the data, and generate a CSV file that can be used with the Geolocator along with GIS data.
```python
from geocr.archive_image import ArchiveImage
from geocr.archive_collection import ArchiveCollection
from geocr.archive_geolocator import ArchiveGeolocator
from geocr.geo_exporter import GeoExporter
from shapely.geometry import Point
import cv2

# 1. Process an image
img = ArchiveImage(cv2.imread("/path/to/image.png"), name="example")
img.sharpen().binarize().ocr(language='fra', config='--psm 3')
print(img.ocr_text)

# 2. Batch process a folder 
col = ArchiveCollection("/path/to/images")
col.sharpen().binarize().ocr()
all_text = col.get_ocr_text()

# 3. Geocode addresses
locator = ArchiveGeolocator("rues.geojson", default_point=Point(0,0))
gdf = locator.geocode_csv(
    csv_path="adresses.csv",
    csv_fields=["No","Adresse"],
    ref_fields=["adresses"],
    similarity_threshold=85
)

# 4. Export results
GeoExporter.to_csv(gdf, "results.csv")
GeoExporter.to_geojson(gdf, "results.geojson")
```

---

## API Reference

### `ArchiveImage`

```python
ArchiveImage(image: np.ndarray, name: str)
```

Constructor: wraps an OpenCV image.

| Method                                                                                                                            | Description                                             | 
| --------------------------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------| 
| `show(grid=False, grid_step=100, grid_color='#FF0000', subgrid_step=20, subgrid_color='#999999', title='Preview', max_size=None)` | displays the image with optional grid.                  | 
| `save(output_dir: Path, format='png', prefix='', suffix='') -> Path`                                                              | save image to disk.                                     | 
| `split(template: str, orientation='vertical') -> list[ArchiveImage]`                                                              | split image by ratios, return new ArchiveImage objects. | 
| `rotate(angle: float)`                                                                                                            | rotate image by angle in degrees.                       | 
| `mask(bottom=0, right=0, top=0, left=0, gray=255, color=(255,255,255))`                                                           | mask borders with color.                                | 
| `invert()`                                                                                                                        | invert image colors.                                    | 
| `denoise(kernel=1, iterations=1)`<br>`erode(kernel=1, iterations=1)`<br>`dilate(kernel=1, iterations=1)`                          | morphological operations.                               | 
| `binarize(mode='auto', threshold_value=127)`                                                                                      | convert to black/white with Otsu or manual threshold.   | 
| `sharpen()`                                                                                                                       | apply unsharp mask.                                     | 
| `remove_borders()`                                                                                                                | crop to largest contour.                                | 
| `add_borders(top=20, bottom=20, left=20, right=20, color=(255,255,255))`                                                          | add uniform border.                                     | 
| `ocr(language='fra', config='--psm 3') -> ArchiveImage`                                                                           | run Tesseract OCR.                                      | 
| `get_ocr_text() -> str`                                                                                                           | display and return OCR text.                            | 
| `clean(rules=None, min_line_length=None)`                                                                                         | apply regex rules and merge short lines.                | 

### `ArchiveCollection`

```python
ArchiveCollection(input_path: str, is_natural_sort=True, split_template=None, split_orientation='vertical')
```

Constructor: loads images from folder.

| Method                                                                                                                                                                                                    | Description                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `get(name: str) -> ArchiveImage`                                                                                                                                                                          | return ArchiveImage with it's name.                           |
| `remove(image_name: str)`                                                                                                                                                                                 | remove image from collection.                                 |
| `update(original_name: str, new_images: list[ArchiveImage])`                                                                                                                                              | update changes from single ArchiveImage update                |
| `subset(imgs: list[str]) -> ArchiveCollection`                                                                                                                                                            | create a new ArchiveCollection from a list of image (by name) |
| `split_only(image_name: str, template: str, orientation='vertical')`                                                                                                                                      | split one image.                                              |
| `split_all(template: str, orientation='vertical')`                                                                                                                                                        | split every image.                                            |
| `show(title='Mosaic', max_cols=6, max_size=None)`                                                                                                                                                         | display grid of images.                                       |
| `copy() -> ArchiveCollection`                                                                                                                                                                             | deep copy collection.                                         |
| `save(output_dir: Path, format='png', prefix='', suffix='') -> list[Path]`                                                                                                                                | write images to disk.                                         |
| All image ops:<br>`sharpen()`<br>`rotate(angle)`<br>`mask(...)`<br>`invert()`<br>`denoise()`<br>`erode()`<br>`dilate()`<br>`binarize()`<br>`remove_borders()`<br>`add_borders(...)`<br>`ocr()`<br>`clean()` | shortcuts to apply on every image.                            |
| `get_ocr_text(separator='\n', show=True)`                                                                                                                                                             | concatenate ocr_text of all ArchiveImage and display OCR text |
                    

### `ArchiveGeolocator`

```python
ArchiveGeolocator(reference_geojson_path: str, default_point: Point = Point(0,0))
```

Constructor: loads GeoJSON reference.

```python
gdf = geolocator.geocode_csv(
    csv_path: str = None,
    gdf: GeoDataFrame = None,
    csv_fields: list[str] = ['Adresse'],
    ref_fields: list[str] = ['nom_voie'],
    output_geometry: str = 'geometry',
    match_label: str = 'matched_ref',
    score_label: str = 'match_score',
    similarity_threshold: float = 80,
    sep: str = ';',
    method: string = "difflib" #difflib or rapidfuzz
) -> GeoDataFrame
```

* If `gdf` provided, reuses it; otherwise loads CSV.
* Only geocodes rows where `geometry` is missing or equals `default_point`.
* Returns `GeoDataFrame` with original columns + `output_geometry`, `match_label`, `score_label`.

### `GeoExporter`

```python
GeoExporter.to_csv(gdf: GeoDataFrame, path: str, sep: str = ';')
```

Export attribute table to CSV (drops geometry).

```python
GeoExporter.to_geojson(gdf: GeoDataFrame, path: str)
```

Export full `GeoDataFrame` to GeoJSON.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
