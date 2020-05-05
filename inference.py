import yaml
from glob import glob
from numba import cuda

from assets.croppse import CropPSE
from assets.createtxt import CreateTxt
from assets.inferencepse import InferencePSE
from assets.inferenceocr import InferenceOCR
from assets.tableunderstanding import *
from assets.visualizer import connect_and_save

def inference(file_path, file_urls):
    result = dict()

    config = yaml.load(open("./assets/config.yaml", 'r'), Loader=yaml.FullLoader)
    pse = InferencePSE(config["pse_evaluation_parameter"], file_path)
    ocr = InferenceOCR(config["ocr_evaluation_parameter"], file_path)
    #Inference PSE
    pse_time = pse.run()
    #Crop image based on PSE output
    # Release the gpu memory
    cuda.select_device(int(config["pse_evaluation_parameter"]["gpu_list"]))
    cuda.close()
    print(file_path)
    CropPSE(file_path)
    #Inference OCR
    ocr_time = ocr.run()
    #Combining Result
    #CreateTxt(file_urls)
    for file_name in file_urls:
        result[file_name] = dict()
        txt_file = "./assets/demo/text/" + file_name.replace("jpg","txt")
        img_file = file_path + file_name
        df, _, _, _ = create_df(txt_file)
        dict_cells, list_infos = create_cells(df)
        result[file_name]['df'] = create_DB(dict_cells, list_infos).drop('idx', axis=1).to_html(header="true")
        # Visualizer
        result[file_name]['img'] = connect_and_save(img_file, dict_cells, list_infos)
    return result