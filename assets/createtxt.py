import os
import numpy as np

def CreateTxt(file_urls):
    for file_url in file_urls:
        pse_file = file_url.replace("original", "pse").replace(".jpg", ".txt")
        f = open(pse_file, encoding="utf-8-sig")
        lines = f.readlines()
        f.close()
        ocr_file = file_url.replace("original", "ocr").replace(".jpg", ".txt")
        f = open(ocr_file, encoding="utf-8-sig")
        ocr_lines = f.readlines()
        f.close()
        i = 0
        new_txt = ''
        for j, line in enumerate(lines):
            if len(ocr_lines) <= j:
                break
            ocr_data = ocr_lines[i].split('\t')
            if ocr_data[0].split('/')[-1].replace(".jpg", '') != str(j):
                continue
            i = i + 1
            label = str(ocr_data[1].replace(' ', '').replace('\n', ''))
            points = line.split(",")
            points = np.array([[float(points[2*x]), float(points[2*x+1])] for x in range(4)])
            
            xMax = str(int(np.max(points[:,0])))
            yMax = str(int(np.max(points[:,1])))
            xMin = str(int(np.min(points[:,0])))
            yMin = str(int(np.min(points[:,1])))
            new_txt = new_txt + xMin + "\t" + yMin + "\t" + \
                        xMin + "\t" + yMax + "\t" + xMax + "\t" + yMin + "\t" + \
                        xMax + "\t" + yMax + "\t" + label + "\n"

        
        txt_file = file_url.replace("original", "txt").replace(".jpg", ".txt")
        f = open(txt_file, 'w', encoding="utf-8-sig")
        f.write(new_txt)
        f.close()