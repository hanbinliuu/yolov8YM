from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


# load model
# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改 lang参数进行切换
# lang参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`
ocr = PaddleOCR(lang="ch",
                use_gpu=False,
                det_model_dir="../paddleORC_model/ch_ppocr_server_v2.0_det_infer/",
                cls_model_dir="ch_ppocr_mobile_v2.0_cls_infer/",
                rec_model_dir="ch_ppocr_server_v2.0_rec_infer/")

# load dataset
img_path = r"C:\Users\Administrator\Desktop\to_rectangle.png"
result = ocr.ocr(img_path)
for line in result:
    print(line)
