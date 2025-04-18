import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import json

def crop_border(image, padding=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - binary
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    return image[y:y + h, x:x + w]

def is_colorful(image, threshold=10):
    stddev_bgr = np.std(image, axis=(0, 1))  # по каналам BGR
    return np.mean(stddev_bgr) > threshold

def is_dense(image, white_thresh=0.85):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary == 255) / binary.size
    return white_ratio < white_thresh

def has_detail(image, std_thresh=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray) > std_thresh

def extract_images_from_scanned_page(
        pdf_path,
        poppler_path=r"C:\Program Files\poppler\Library\bin",
        save_dir="images"
):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    for page_num, pil_image in enumerate(pages):
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_index = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / h if h != 0 else 0

            if area < 30000 or aspect_ratio < 0.3 or aspect_ratio > 4:
                continue
            if area > image.shape[0] * image.shape[1] * 0.9:
                continue

            roi = image[y:y + h, x:x + w]
            roi_cropped = crop_border(roi)

            # Проверка по признакам
            if not (is_colorful(roi_cropped) and is_dense(roi_cropped) and has_detail(roi_cropped)):
                continue  # скорее всего не изображение

            filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, roi_cropped)

            results.append({
                "file": filename,
                "page": page_num + 1,
                "position": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })

            img_index += 1

        print(f"Страница {page_num + 1}: сохранено {img_index} изображений.")

    json_path = os.path.join(save_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nJSON сохранён в: {json_path}")


# Пример запуска:
extract_images_from_scanned_page("Отчёт ЛР1 ОРБД.pdf")
