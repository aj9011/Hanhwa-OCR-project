import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
from PIL import Image, ImageDraw
import numpy as np

import base64

def display_cell_and_connections2(cell, image, left_cells):
    ex = image.copy()
    r = 0
    g = 0
    b = 255
    ex = draw_box(cell, ex, r, g, b, draw_lines=True)
    start_point = (int(cell.x_min + cell.w / 2), int(cell.y_min + cell.h / 2))
    c = left_cells
    end_point = (int(c.x_min + c.w / 2), int(c.y_min + c.h / 2))
    r = 255
    g = 0
    b = 0
    ex = cv2.line(ex, start_point, end_point, color=(r, g, b),thickness=2)
    r = 0
    g = 0
    b = 255
    ex = draw_box(left_cells, ex, r, g, b, draw_lines=True)
    return ex

def draw_box(c, image, r, g, b, draw_lines=False):
    ex = image
    x = int(c.x_min)
    y = int(c.y_min)
    w = int(c.w)
    h = int(c.h)

    cv2.rectangle(ex, (x, y), (x + w, y + h), (r, g, b), 2)
    cv2.circle(ex, (int(x + w / 2), int(y + h / 2)), 1, (r, g, b), 3)  # center of the box
       
    return ex

def draw_rectangle(img, coords, color, thickness=2, text=None, text_origin="upper"):
    xmin, ymin, xmax, ymax = coords
    tmp_img = img.copy()
    draw = ImageDraw.Draw(tmp_img)
    # draw bounding box
    for i in range(thickness):
        a = max(0, xmin - i)
        b = max(0, ymin - i)
        c = min(img.size[0], xmax + i)
        d = min(img.size[1], ymax + i)
        draw.rectangle([a, b, c, d], outline=color)
    # draw text
    if text is not None:
        font = ImageFont.truetype(font="DejaVuSansMono.ttf", size=25)
        text_size = draw.textsize(text, font)
        if text_origin == "upper":
            text_origin = np.array([xmin, ymin - text_size[1]])
        else:
            text_origin = np.array([xmin, ymax])
        draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], 
                      fill=color)
        draw.text(tuple(text_origin), text, fill=(0, 0, 0), font=font)
    del draw
    return tmp_img


def connect_and_save(img_file, dict_cells, list_infos):
    result_image = np.array(Image.open(img_file))
    result_path = img_file.replace("original", 'result')
    for dict_cell in dict_cells.values():
        if dict_cell.text == '0':
            continue
        result_image = draw_box(dict_cell, result_image, 0, 0, 255, draw_lines=True)
    for idx in list_infos:
        cell = dict_cells[idx]
        if cell.left[1] != 0:
            result_image = display_cell_and_connections2(cell,result_image,dict_cells[cell.left[0]])

        if cell.top[1] != 0:
            result_image = display_cell_and_connections2(cell,result_image,dict_cells[cell.top[0]])
    Image.fromarray(result_image).save(result_path)
    return result_path