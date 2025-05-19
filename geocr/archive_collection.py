# SPDX-License-Identifier: MIT
import cv2
import numpy as np
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from .archive_image import ArchiveImage

class ArchiveCollection:
  def __init__(self, input_path, is_natural_sort=True, split_template=None, split_orientation="vertical"):
    self.input_path = Path(input_path)
    self.is_natural_sort = is_natural_sort
    self._files = self._load()
    self.images = sorted([ArchiveImage(cv2.imread(str(f)), f"f_{f.stem}") for f in self._files], key=self._natural_sort_key)

    if split_template:
        self.split_all(split_template, split_orientation)

  def _natural_sort_key(self, image):
        match = re.search(r'\d+', image.name)
        return int(match.group()) if match else image.name

  def _apply_to_all(self, method_name: str, **kwargs):
    for img in self.images:
        getattr(img, method_name)(**kwargs)
    return self

  def _load(self):
    files = [f for f in self.input_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff"]]
    return sorted(files, key=self._natural_sort_key if self.is_natural_sort else lambda f: f.name)

  def get(self, name: str):
    for img in self.images:
        if img.name == name:
            return img
    raise ValueError(f"Image '{name}' not found")

  def remove(self, image_name: str):
    initial_len = len(self.images)
    self.images = [img for img in self.images if img.name != image_name]
    if len(self.images) < initial_len:
        print(f"Image '{image_name}' deleted from collection.")
    else:
        print(f"Image '{image_name}' not found")
    return self

  def update(self, original_name, new_images):
      self.images = [img for img in self.images if img.name != original_name]
      self.images.extend(new_images)
      self.images = sorted(self.images, key=self._natural_sort_key)

  def subset(self, imgs: list[str]):
    subset_collection = ArchiveCollection.__new__(ArchiveCollection)
    subset_collection.input_path = self.input_path
    subset_collection.is_natural_sort = self.is_natural_sort
    subset_collection._files = self._files.copy()
    subset_collection.images = [img.copy() for img in self.images if img.name in imgs]
    return subset_collection

  def split_only(self, image_name: str, template: str, orientation="vertical"):
    image = self.get(image_name)
    new_images = image.split(template, orientation)
    self.update(image_name, new_images)
    return self

  def split_all(self, template: str, orientation="vertical"):
    all_new_images = []

    for image in self.images:
        new_images = image.split(template, orientation)
        all_new_images.append((image.name, new_images))

    for original_name, new_imgs in all_new_images:
        self.update(original_name, new_imgs)

    return self

  def show(self, title="Mosaique", max_cols=6, max_size=None):
      if not self.images:
          print("No image to display")
          return

      n = len(self.images)
      cols = min(max_cols, n)
      rows = (n + cols - 1) // cols

      fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
      axes = np.array(axes).reshape(-1)
      for ax in axes[n:]:
          ax.axis("off")

      for ax, img_obj in zip(axes, self.images):
          rgb = cv2.cvtColor(img_obj.image, cv2.COLOR_BGR2RGB)
          if max_size:
              pil_img = Image.fromarray(rgb)
              pil_img.thumbnail(max_size)
              rgb = np.array(pil_img)

          ax.imshow(rgb)
          ax.set_title(img_obj.name, fontsize=8)
          ax.axis("off")

      plt.suptitle(title)
      plt.tight_layout()
      plt.show()

  def copy(self):
      clone = ArchiveCollection.__new__(ArchiveCollection)
      clone.input_path = self.input_path
      clone.is_natural_sort = self.is_natural_sort
      clone._files = self._files.copy()
      clone.images = [img.copy() for img in self.images]
      return clone

  def save(self, output_dir: Path, format="png", prefix="", suffix=""):
      output_dir.mkdir(parents=True, exist_ok=True)
      paths = []
      for img in self.images:
          filename = f"{prefix}{img.name}{suffix}.{format}"
          path = output_dir / filename
          cv2.imwrite(str(path), img.image)
          paths.append(path)
      return paths

  def sharpen(self):
    return self._apply_to_all("sharpen")

  def rotate(self, angle):
    return self._apply_to_all("rotate", angle=angle)

  def mask(self, bottom=0, right=0, top=0, left=0, gray=255, color=(255,255,255)):
      return self._apply_to_all("mask", bottom=bottom, right=right, top=top, left=left, gray=gray, color=color)

  def invert(self):
      return self._apply_to_all("invert")

  def denoise(self, kernel=1, iterations=1):
      return self._apply_to_all("denoise", kernel=kernel, iterations=iterations)

  def erode(self, kernel=1, iterations=1):
      return self._apply_to_all("erode", kernel=kernel, iterations=iterations)

  def dilate(self, kernel=1, iterations=1):
      return self._apply_to_all("dilate", kernel=kernel, iterations=iterations)

  def binarize(self, mode="auto", threshold_value=127):
      return self._apply_to_all("binarize",  mode=mode, threshold_value=threshold_value)

  def remove_borders(self):
      return self._apply_to_all("remove_borders")

  def add_borders(self, t=20, b=20, l=20, r=20, color=(255,255,255)):
      return self._apply_to_all("add_borders", t=t, b=b, l=l, r=r, color=color)

  def ocr(self, language="fra", config="--psm 3"):
      return self._apply_to_all("ocr", language=language, config=config)

  def clean(self,rules=None, min_line_length=None):
      return self._apply_to_all("clean", rules=rules, min_line_length=min_line_length)

  def get_ocr_text(self, separator='\n', show=True):
    result = separator.join(
        img.ocr_text.strip() for img in self.images if img.ocr_text
    )
    if show:
      text_md = result.strip().replace("\n", "  \n")
      display(Markdown(f"```{text_md}\n```"))

    return result

