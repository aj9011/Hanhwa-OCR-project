import re
import pandas as pd
import numpy as np
import statistics
import math
import os

class PseCell():
    def __init__(self, x, y, w, h, x_min, y_min, x_max, y_max, text=''):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        #left = [cell_idx, distance, box size]
        self.left, self.top = [0, 0, self.h], [0, 0, self.w]
        self.text = text
        
def create_df(path):
    df = pd.read_csv(path, encoding='utf-8-sig', header=None, sep='\t')
    
    df.columns = ['left_top_x','left_top_y','left_bottom_x','left_bottom_y','right_top_x','right_top_y', 'right_bottom_x','right_bottom_y','text']
    df['w'] = abs(df['right_bottom_x'] - df['left_bottom_x'])
    df['h'] = abs(df['right_top_y'] - df['right_bottom_y'])
    df['x'] = abs((df['right_bottom_x'] - df['left_bottom_x'])/2 + df['left_bottom_x'])
    df['y'] = abs((df['right_top_y'] - df['right_bottom_y'])/2 + df['right_top_y'])
    df['text'] = df['text'].replace(" ","",regex=True).replace(",","",regex=True)
    
    left_bottom_list = ['상한액초과금', '상한액초과금(6)', '상환액초과금(6)']
    right_top_list = ['선택진료료', '선택진료', '선택진료료이외', '선택진료료이']
    left_bottom_cell = df[df['text'].isin(left_bottom_list)]
    left_bottom_x = max(left_bottom_cell['left_bottom_x'])
    left_bottom_y = max(left_bottom_cell['left_bottom_y'])
    right_top_cell = df[df['text'].isin(right_top_list)]
    right_top_x = max(right_top_cell['right_top_x'])
    right_top_y = max(right_top_cell['right_top_y'])

    topright_x_thresh = 50
    topright_y_thresh = 20

    bottomleft_y_thresh = 50
    
    outerIndex = df[(df['x']> topright_x_thresh + right_top_x) | (df['y']+ topright_y_thresh < right_top_y) | (df['y'] > left_bottom_y + bottomleft_y_thresh)].index
    df.drop(outerIndex , inplace=True)
    textException = df[(df['text'] == "일부본인부담금") | (df['text'] == "일부본인부담") | (df['text'] == 0)].index
    df.drop(textException, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for i in range(len(df)):
        if type(df['text'][i]) == str:
            new_text = re.sub(r'\(.*\)', '', df['text'][i])
            df['text'][i] = re.sub(r'\(.*\)', '', df['text'][i])
        if df['text'][i].isnumeric():
            df['text'][i] = int(df['text'][i])
    

    return df, right_top_x, right_top_y, left_bottom_y       

def create_cells(df):
    dict_cell = dict()
    list_info_idx = []
    for i in range(len(df)):
        series = df.loc[i]
        if series['text'] in ['']:
            continue
        new_cell = PseCell(series['x'], series['y'], series['w'], series['h'], 
                  series['left_top_x'], series['left_top_y'], 
                  series['right_bottom_x'], series['right_bottom_y'],
                  series['text'])
        dict_cell[i] = new_cell
        if type(series['text']) != str:
            list_info_idx.append(i)
    return dict_cell, list_info_idx

def connect_cells_iou(dict_cells, list_infos):
    for info in list_infos:
        iou_threshold = 0.2
        for idx, cell in dict_cells.items():
            if idx == info: # if idx is number or text us 0
                continue
            if dict_cells[info].y_min > cell.y_max and (dict_cells[info].text != 0 and cell.text != 0): 
                # connect to top
                x_min = np.maximum(dict_cells[info].x_min, cell.x_min)
                x_max = np.minimum(dict_cells[info].x_max, cell.x_max)
                x_intersection = np.maximum(0, x_max - x_min + 1)
                x_union = np.float(dict_cells[info].w + cell.w - x_intersection)
                x_iou = x_intersection / x_union

                if (x_iou > iou_threshold):
                    distance = math.sqrt((cell.x - dict_cells[info].x)**2 + (cell.y - dict_cells[info].y)**2)
                    if dict_cells[info].top[1] == 0 or dict_cells[info].top[1] > distance:
                        dict_cells[info].top[:2] = [idx, distance]
        
            # connect to left
            if dict_cells[info].x_min > cell.x_max:
                y_min = np.maximum(dict_cells[info].y_min, cell.y_min)
                y_max = np.minimum(dict_cells[info].y_max, cell.y_max)
                y_intersection = np.maximum(0, y_max - y_min + 1)
                y_union = np.float(dict_cells[info].h + cell.h - y_intersection)
                y_iou = y_intersection / y_union

                if (y_iou > iou_threshold):
                    #distance = math.sqrt((cell.x - dict_cells[info].x)**2 + (cell.y - dict_cells[info].y)**2)
                    distance =  abs(cell.x - dict_cells[info].x)
                    if dict_cells[info].left[1] == 0 or dict_cells[info].left[1] > distance:
                        dict_cells[info].left[:2] = [idx, distance] 

        # Update Cell Size
        dict_cells[info].left[2] = dict_cells[dict_cells[info].left[0]].h
        dict_cells[info].top[2] = dict_cells[dict_cells[info].top[0]].w
        
def upgrade_cell_size(dict_cells):
    sums_h=[]
    sums_w = []
    for cell in dict_cells.values():       
        sums_h.append(cell.h)
        sums_w.append(cell.w)
    h_mean = statistics.mean(sums_h)
    w_mean = statistics.mean(sums_w)
    for cell in dict_cells.values():
        if cell.h < h_mean:
            cell.h = h_mean
            cell.left[2] = h_mean
        if cell.w < w_mean:
            cell.w = w_mean
            cell.top[2] = w_mean

        
def create_DB(list_cells, list_infos):
    upgrade_cell_size(list_cells)
    connect_cells_iou(list_cells, list_infos)
    num_list = []
    idx_list = []
    left_top_f=[]

    for info_idx in list_infos:
        info = list_cells[info_idx]
        # Skip 0s which is not necessary
        if info.text == 0:
            continue
        num_list.append(info.text)
        idx_list.append(info_idx)
        # Go left until it meets string field
        left_box_idx = info.left[0]
        while type(list_cells[left_box_idx].text) != str:
            left_box_idx = list_cells[left_box_idx].left[0]
        # Go top until it meets string field
        top_box_idx = info.top[0]
        while type(list_cells[top_box_idx].text) != str:
            top_box_idx = list_cells[top_box_idx].top[0]
        left_top_f.append(list_cells[left_box_idx].text + '/' + \
                            list_cells[top_box_idx].text)
    
    num_list_df = pd.DataFrame(num_list)
    num_list_df.columns = ['info']
    idx_list_df = pd.DataFrame(idx_list)
    idx_list_df.columns = ['idx']
    left_top_f_df = pd.DataFrame(left_top_f)
    left_top_f_df.columns = ['field']
    doc_ans_df = pd.concat([num_list_df, idx_list_df, left_top_f_df], axis=1)
    
    return doc_ans_df 