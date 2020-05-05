import os
import cv2
from glob import glob

def CropPSE(file_urls):
    file_pathes = glob(file_urls + "*")
    label_str = ""
    for j, file_url in enumerate(file_pathes):
        print('cuurent img : ' + file_url)
        img = cv2.imread(file_url)
        txt_file = file_url.replace("original", "pse").replace(".jpg", ".txt")
        with open(txt_file, encoding="utf-8-sig") as f:
            lines = f.readlines()
        output_folder = file_url.replace("original", "ocr").replace(".jpg", "")
        try:
            os.makedirs(output_folder)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
        for i, line in enumerate(lines):
            points = line.split(",")
            points = [[points[2*x], points[2*x+1]]for x in range(4)]
            xmin, ymin, xmax, ymax = -1, -1, -1, -1

            for point in points:
                if xmax < float(point[0]):
                    xmax = float(point[0])
                elif xmin == -1 or xmin > float(point[0]):
                    xmin = float(point[0])
                if ymax < float(point[1]):
                    ymax = float(point[1])
                elif ymin == -1 or ymin > float(point[1]):
                    ymin = float(point[1])
            if int(ymin) < 3:
                ymin = 0
            else:
                ymin = int(ymin) - 3
            if int(xmin) < 3:
                xmin = 0
            else:
                xmin = int(xmin) - 3
            if int(ymax) + 3 >= img.shape[0]:
                ymax = int(img.shape[0])
            else:
                ymax = int(ymax) + 3
            if int(xmax) + 3 >= img.shape[1]:
                xmax = int(img.shape[1])
            else:
                xmax = int(xmax) + 3
            img_crop = img[ymin:ymax, xmin:xmax]
            try:
                cv2.imwrite(output_folder + "/" + str(i) + ".jpg", img_crop)
            except:
                print("ERrOR")
                exit()
                continue