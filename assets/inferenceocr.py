import string
import easydict
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from assets.utils.ocr.utils import CTCLabelConverter, AttnLabelConverter
from assets.utils.ocr.dataset import RawDataset, AlignCollate
from assets.nets.ocrmodel import Model
from glob import glob

class InferenceOCR:
    def __init__(self, parameters, file_urls):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f = open(parameters["dictionary"], encoding="utf-8-sig")
        chars = f.read()
        f.close()
        self.opt = easydict.EasyDict({
            "image_folder": file_urls.replace("original", "ocr"),
            "workers": 4,
            "batch_size": 192,
            "saved_model": parameters["saved_model"],
            "batch_max_length": 25,
            "imgH": 32,
            "imgW": 100,
            "character": chars,
            "Transformation": "TPS",
            "FeatureExtraction": "ResNet",
            "SequenceModeling": "BiLSTM",
            "Prediction": "Attn",
            "num_fiducial": 20,
            "input_channel": 1,
            "output_channel": 512,
            "hidden_size": 256,
            "PAD": False,
            "sensitive": False,
            "rgb": False
        })

    def demo(self):
        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            converter = CTCLabelConverter(self.opt.character)
        else:
            converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel, self.opt.output_channel,
              self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length, self.opt.Transformation, self.opt.FeatureExtraction,
              self.opt.SequenceModeling, self.opt.Prediction)
        model = torch.nn.DataParallel(model).to(self.device)

        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        model.load_state_dict(torch.load(self.opt.saved_model, map_location=self.device))
        sh_time1 = time.time()
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)

        image_folders = glob(self.opt.image_folder + "/*")
        for image_folder in image_folders:
            demo_data = RawDataset(root=image_folder, opt=self.opt)
            demo_loader = torch.utils.data.DataLoader(
                demo_data, batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=int(self.opt.workers),
                collate_fn=AlignCollate_demo, pin_memory=True)

            # predict
            model.eval()
            with torch.no_grad():
                for image_tensors, image_path_list in demo_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(self.device)
                    # For max length prediction
                    length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                    text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                    if 'CTC' in self.opt.Prediction:
                        preds = model(image, text_for_pred)
                        # Select max probabilty (greedy decoding) then decode index to character
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.view(-1)
                        preds_str = converter.decode(preds_index.data, preds_size.data)

                    else:
                        preds = model(image, text_for_pred, is_train=False)

                        # select max probabilty (greedy decoding) then decode index to character
                        _, preds_index = preds.max(2)
                        preds_str = converter.decode(preds_index, length_for_pred)

                    log = open(image_folder +'.txt', 'a')
                    dashed_line = '-' * 80
                    head = '{:25s}\t{:25s}\tconfidence score'.format("image_path", "predicted_labels")

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                        if 'Attn' in self.opt.Prediction:
                            pred_EOS = pred.find('[s]')
                            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                            pred_max_prob = pred_max_prob[:pred_EOS]

                        if len(pred_max_prob.cumprod(dim=0)) == 0:
                            continue

                        # calculate confidence score (= multiply of pred_max_prob)
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                        log.write('{:25s}\t{:25s}\t{:0.4f}\n'.format(img_name, pred, confidence_score))

                    log.close()
        return time.time() - sh_time1
        
    def run(self):
        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()
        return self.demo()