import cv2
import numpy as np
import pytesseract
import os
import time
from pdf2image import convert_from_path

# Пути
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + POPPLER_PATH

TESS_CONFIG = r'--oem 3 --psm 6 -l rus+eng'

def pdf_to_images(pdf_path, dpi=300):
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
    except Exception as e:
        print(f"Ошибка конвертации PDF в изображения: {e}")
        return None


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def detect_table_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )
    hor = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal = cv2.dilate(cv2.erode(binary, hor), hor)
    vertical = cv2.dilate(cv2.erode(binary, ver), ver)
    mask = cv2.add(horizontal, vertical)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:
            boxes.append((x, y, x + w, y + h))
    return boxes


def filter_table_boxes(boxes, min_area_ratio=0.5):
    if not boxes:
        return []
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    max_area = max(areas)
    return [box for box, area in zip(boxes, areas) if area >= max_area * min_area_ratio]


def mask_tables(img, boxes):
    res = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(res, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return res


def detect_cells(table_img):
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    hlines = cv2.dilate(cv2.erode(bw, horiz), horiz)
    vlines = cv2.dilate(cv2.erode(bw, vert), vert)
    joints = cv2.bitwise_and(hlines, vlines)
    ys, xs = np.where(joints > 0)

    def cluster(coords, tol=10):
        coords = sorted(coords)
        groups = []
        for c in coords:
            if not groups or abs(c - groups[-1][-1]) > tol:
                groups.append([c])
            else:
                groups[-1].append(c)
        return [int(sum(g) / len(g)) for g in groups]

    rows = cluster(ys)
    cols = cluster(xs)
    cells = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            y1, y2 = rows[i], rows[i + 1]
            x1, x2 = cols[j], cols[j + 1]
            cells.append((x1, y1, x2, y2, i + 1, j + 1))
    return cells


def extract_cells(table_img, table_box):
    # table_box содержит (x1, y1, x2, y2)
    x_off, y_off, _, _ = table_box  # только смещение
    cells = detect_cells(table_img)
    data = []
    for x1, y1, x2, y2, row, col in cells:
        cell_img = table_img[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cell_img, config=TESS_CONFIG).strip()
        abs_box = [x_off + x1, y_off + y1, x_off + x2, y_off + y2]
        data.append({"row": row, "col": col, "text": text, "box": abs_box})
    return data

# --- Функции обработки PDF ---
def pdf_to_images(pdf_path, dpi=300):
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
    except Exception as e:
        print(f"Ошибка конвертации PDF в изображения: {e}")
        return None

def expand_box(box, pad_x=5, pad_y=5, img_shape=None):
    x1, y1, x2, y2 = box
    if img_shape:
        max_x, max_y = img_shape[1], img_shape[0]
    else:
        max_x, max_y = None, None
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(max_x if max_x else x2 + pad_x, x2 + pad_x)
    y2 = min(max_y if max_y else y2 + pad_y, y2 + pad_y)
    return [x1, y1, x2, y2]

def extract_text_lines(img, y_tolerance=10):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=TESS_CONFIG)
    words = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        words.append({
            "text": text,
            "box": [x, y, x + w, y + h],
            "center_y": y + h // 2
        })

    # Группируем по Y (строкам)
    lines = []
    for word in sorted(words, key=lambda w: w['center_y']):
        found = False
        for line in lines:
            if abs(line['avg_y'] - word['center_y']) <= y_tolerance:
                line['words'].append(word)
                line['avg_y'] = np.mean([w['center_y'] for w in line['words']])
                found = True
                break
        if not found:
            lines.append({'words': [word], 'avg_y': word['center_y']})

    # Финальное объединение текста
    result = []
    for line in lines:
        line_words = sorted(line['words'], key=lambda w: w['box'][0])
        text = " ".join(w['text'] for w in line_words)
        x1 = min(w['box'][0] for w in line_words)
        y1 = min(w['box'][1] for w in line_words)
        x2 = max(w['box'][2] for w in line_words)
        y2 = max(w['box'][3] for w in line_words)
        result.append({
            "text": text,
            "box": expand_box([x1, y1, x2, y2], pad_x=5, pad_y=5, img_shape=img.shape)
        })

    return result

def generate_html_report(json_data, report_data, html_path='output.html'):
    """Генерирует HTML-отчет с результатами обработки PDF"""
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>PDF Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .header {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .stats {{ margin-bottom: 20px; }}
        .page {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
        .text-content {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        .table-content {{ margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .coords {{ font-size: 0.8em; color: #666; }}
        .errors {{ color: #e74c3c; background: #fde8e8; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PDF Processing Report</h1>
        <div class="stats">
            <p><strong>File:</strong> {report_data['file_name']}</p>
            <p><strong>Pages processed:</strong> {report_data['pages']}</p>
            <p><strong>Processing time:</strong> {report_data['duration']} seconds</p>
            <p><strong>Estimated accuracy:</strong> {report_data['accuracy'] * 100:.1f}%</p>
        </div>
    </div>
"""

    # Добавляем данные по страницам
    for p, page in enumerate(json_data['pages']):
        html += f"""
    <div class="page">
        <h2>Page {p + 1}</h2>

        <div class="text-content">
            <h3>Extracted Text</h3>
            <ul>"""

        for line in page['lines']:
            html += f"""
                <li>{line['text']} <span class="coords">[{line['box'][0]}, {line['box'][1]}, {line['box'][2]}, {line['box'][3]}]</span></li>"""

        html += """
            </ul>
        </div>"""

        # Добавляем таблицы
        for t, table in enumerate(page['tables']):
            html += f"""
        <div class="table-content">
            <h3>Table {t + 1}</h3>
            <p><strong>Location:</strong> {table['box']}</p>
            <table>"""

            # Организуем ячейки по строкам и столбцам
            rows = {}
            for cell in table['cells']:
                r, c = cell['row'], cell['col']
                rows.setdefault(r, {})[c] = cell['text']

            max_cols = max([max(r.keys()) for r in rows.values()]) if rows else 0

            # Заголовок таблицы
            html += "<tr>"
            for c in range(1, max_cols + 1):
                html += f"<th>Column {c}</th>"
            html += "</tr>"

            # Данные таблицы
            for r in sorted(rows.keys()):
                html += "<tr>"
                for c in range(1, max_cols + 1):
                    text = rows[r].get(c, "")
                    html += f"<td>{text}</td>"
                html += "</tr>"

            html += """
            </table>
        </div>"""

        html += """
    </div>"""

    # Добавляем ошибки, если есть
    if report_data['errors']:
        html += """
    <div class="errors">
        <h3>Processing Errors</h3>
        <ul>"""

        for error in report_data['errors']:
            html += f"""
            <li>{error}</li>"""

        html += """
        </ul>
    </div>"""

    html += """
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def process_pdf(pdf_path, output_html='output.html', tables_dir='tables', area_ratio=0.5):
    """
    Основная функция обработки PDF-файла
    :param pdf_path: путь к PDF-файлу
    :param output_html: путь для сохранения HTML-отчета
    :param tables_dir: директория для сохранения изображений таблиц
    :param area_ratio: минимальное отношение площади таблицы к площади страницы
    """
    os.makedirs(tables_dir, exist_ok=True)
    start_time = time.time()
    images = pdf_to_images(pdf_path)

    if not images:
        print("Failed to load PDF")
        return

    doc = {"pages": []}
    stats = {
        "file_name": os.path.basename(pdf_path),
        "pages": len(images),
        "tables": 0,
        "lines": 0,
        "errors": [],
        "accuracy": 0.0,
        "duration": 0.0
    }

    for page_num, img in enumerate(images, 1):
        boxes = filter_table_boxes(detect_table_boxes(img), area_ratio)
        page = {"lines": [], "tables": []}
        clean_img = mask_tables(img, boxes)

        try:
            lines = extract_text_lines(clean_img)
            page['lines'] = lines
            stats["lines"] += len(lines)
        except Exception as e:
            stats["errors"].append(f"Text extraction error on page {page_num}: {str(e)}")

        for table_num, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = box
            table_img = img[y1:y2, x1:x2]
            filename = f"{tables_dir}/tbl_p{page_num}_{table_num}.png"
            cv2.imwrite(filename, table_img)

            try:
                cells = extract_cells(table_img, box)
                page["tables"].append({
                    "image": filename,
                    "box": list(box),
                    "cells": cells
                })
                stats["tables"] += 1
            except Exception as e:
                stats["errors"].append(
                    f"Table processing error on page {page_num}, table {table_num}: {str(e)}"
                )

        doc["pages"].append(page)

    # Расчет статистики
    stats["duration"] = round(time.time() - start_time, 2)
    total_items = stats["lines"] + stats["tables"] * 5
    error_penalty = len(stats["errors"]) * 0.02
    stats["accuracy"] = max(0.0, min(1.0, 1.0 - error_penalty))

    # Генерация HTML-отчета (вместо сохранения JSON)
    generate_html_report(doc, stats, html_path=output_html)
    print(f"Processing complete. Report saved to {output_html}")


if __name__ == "__main__":
    process_pdf("TTimg.pdf")