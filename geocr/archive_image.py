# SPDX-License-Identifier: MIT
import cv2
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pytesseract as tess
from IPython.display import display, Markdown


class ArchiveImage:
  def __init__(self, image, name):
    self.name = name
    self.parent = ""
    self.image = image.copy()
    self.ocr_text = ""
    self.data = None

  def copy(self):
    return ArchiveImage(self.image.copy(), self.name)

  def show(self, grid=False, grid_step=100, grid_color="#FF0000", subgrid_step=20, subgrid_color="#999999", title="Preview", max_size=None):
    img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    h, w = self.image.shape[:2]

    scale = min(max_size[0] / w, max_size[1] / h) if max_size else 1
    figsize = (w * scale / 100, h * scale / 100)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(f"{title} [{self.name}] ({w}x{h})")

    if grid:
        ax.set_xticks(range(0, w, grid_step))
        ax.set_yticks(range(0, h, grid_step))
        ax.grid(color=grid_color, linestyle='--', linewidth=0.5)
        ax.tick_params(labelbottom=True, labelleft=True)

        ax.set_xticks(range(0, w, round(subgrid_step)), minor=True)
        ax.set_yticks(range(0, h, round(subgrid_step)), minor=True)
        ax.grid(which='minor',color=subgrid_color, linestyle=':', linewidth=0.3)

    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])

    plt.show()

  def save(self, output_dir: Path, format="png", prefix="", suffix="") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}{self.name}{suffix}.{format}"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), self.image)
    return output_path

  def split(self, template: str, orientation="vertical") -> list:
    ratios = [int(r) for r in template.split("|")]
    total = sum(ratios)
    img = self.image
    h, w = img.shape[:2]
    result = []
    pos = 0

    for i, ratio in enumerate(ratios):
        if orientation == "vertical":
            width = round((ratio / total) * w)
            part = img[:, pos:pos + width]
            pos += width
        elif orientation == "horizontal":
            height = round((ratio / total) * h)
            part = img[pos:pos + height, :]
            pos += height
        else:
            raise ValueError("Unsupported orientation.")

        name = f"{self.name}_{i}"
        result.append(ArchiveImage(part, name))

    return result

  def rotate(self, angle):
    h, w = self.image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    self.image = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return self

  def mask(self, bottom=0, right=0, top=0, left=0, gray=255, color=(255,255,255)):
      h, w = self.image.shape[:2]

      if len(self.image.shape) == 2:
          bg = gray
      else:
          bg = color

      if bottom > 0:
          self.image[h - bottom:h, :] = bg
      if top > 0:
          self.image[0:top, :] = bg
      if right > 0:
          self.image[:, w - right:w] = bg
      if left > 0:
          self.image[:, 0:left] = bg
      return self

  def invert(self):
    self.image = cv2.bitwise_not(self.image)
    return self

  def denoise(self, kernel=1, iterations=1):
    dilatedImage = cv2.dilate(self.image, np.ones((kernel,kernel), np.uint8), iterations=iterations)
    erodedImage = cv2.erode(dilatedImage, np.ones((kernel,kernel), np.uint8), iterations=iterations)
    morphImage = cv2.morphologyEx(erodedImage, cv2.MORPH_CLOSE, np.ones((kernel,kernel), np.uint8))
    self.image = cv2.medianBlur(morphImage, 3)
    return self

  def erode(self, kernel=1, iterations=1):
    inverted = cv2.bitwise_not(self.image)
    erodedImage = cv2.erode(inverted, np.ones((kernel,kernel), np.uint8), iterations=iterations)
    self.image = cv2.bitwise_not(erodedImage)
    return self

  def dilate(self, kernel=1, iterations=1):
    inverted = cv2.bitwise_not(self.image)
    dilatedImage = cv2.dilate(inverted, np.ones((kernel,kernel), np.uint8), iterations=iterations)
    self.image = cv2.bitwise_not(dilatedImage)
    return self

  def binarize(self, mode="auto", threshold_value=127):
      if len(self.image.shape) == 3 and self.image.shape[2] == 3:
          gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
      else:
          gray = self.image

      if mode == "auto":
          _, self.image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      elif mode == "manual":
          _, self.image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
      else:
          raise ValueError("Unsupported value (auto or manual)")

      return self

  def sharpen(self):
    if len(self.image.shape) == 3:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    else:
        gray = self.image

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    self.image = sharpened
    return self

  def remove_borders(self):
    grayscaleImage = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grayscaleImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return self
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    self.image = self.image[y:y+h, x:x+w]
    return self

  def add_borders(self, top=20, bottom=20, left=20, right=20, color=(255, 255, 255)):
    h, w = self.image.shape[:2]
    new_h, new_w = h + top + bottom, w + left + right

    if len(self.image.shape) == 2:
        new_image = np.full((new_h, new_w), color[0], dtype=self.image.dtype)
    else:
        new_image = np.full((new_h, new_w, 3), color, dtype=self.image.dtype)

    new_image[top:top+h, left:left+w] = self.image

    self.image = new_image
    return self

  def ocr(self, language='fra', config='--psm 3'):
    self.ocr_text = tess.image_to_string(self.image, lang=language, config=config)
    return self

  def get_ocr_text(self) -> str:
    if not self.ocr_text:
        display(Markdown(f"**{self.name}**\n\n_No OCR text available._"))
        return

    text_md = self.ocr_text.strip().replace("\n", "  \n")
    display(Markdown(f"**{self.name}**\n\n```\n{text_md}\n```"))

  def clean(self, rules=None, min_line_length=None):
      if not self.ocr_text:
          return self

      text = self.ocr_text

      if rules:
          for regex, replacement in rules:
              text = re.sub(regex, replacement, text)

      lines = [line.strip() for line in text.splitlines() if line.strip()]

      if min_line_length:
          merged = []
          for line in lines:
              if len(line) < min_line_length and merged:
                  merged[-1] += ' ' + line
              else:
                  merged.append(line)
          lines = merged

      self.ocr_text = "\n".join(lines)
      return self