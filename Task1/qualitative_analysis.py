from src.qualitative_analysis.att_cam_encoder import  detr_encoder_cam
from src.qualitative_analysis.faster_cam import faster_cam
from src.qualitative_analysis.yolo_cam import yolo_cam
from src.qualitative_analysis.faster_roi_cam import faster_cam_roi
from src.qualitative_analysis.draw_pred_boxes import draw_pred_boxes
from src.qualitative_analysis.att_cam_decoder import detr_decoder_cam
from pathlib import Path
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=str,nargs='+',   
                        help='List of sample names')
    parser.add_argument('--model_type', type=str, nargs='+', default=['detr'], choices=['yolo', 'detr', 'fasterrcnn', 'None'],
                        help='Model frameworks to use')
    parser.add_argument('--model_paths', type=str, nargs='+', default=[None],
                        help='Model paths to use, same order as model type')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_path', type=str, default="/home/group05/maiol/MCV-C5/outs")

    return parser.parse_args()

args = get_args()

output_folder = args.output_path

for sample in args.samples: #sample = 0000/000000.png
    result = Path(sample).with_suffix('')
    path = os.path.join(output_folder,result)

    if "detr" in args.model_type:
        decoder_path = os.path.join(path, "DEC")
        Path(decoder_path).mkdir(parents=True, exist_ok=True)
        detr_decoder_cam(sample, decoder_path, checkpoint_path=args.model_paths[0])

        enoder_path = os.path.join(path, "ENC")
        Path(enoder_path).mkdir(parents=True, exist_ok=True)
        detr_encoder_cam(sample, args.device, enoder_path, checkpoint_path=args.model_paths[0])
    
    if "fasterrcnn" in args.model_type:
        faster_path = os.path.join(path, "FRCNN")
        Path(faster_path).mkdir(parents=True, exist_ok=True)
        faster_cam(sample, faster_path, args.model_paths[0], class_id=1, pred_id=0, lora=False)
        #faster_cam_roi(sample, faster_path, args.model_paths[0], 1)
    
    if "yolo" in args.model_type:
        yolo_path = os.path.join(path, "YOLO")
        yolo_cam(sample,args.model_paths[0], args.device, yolo_path)

    else:
        boxes_path = os.path.join(path, "BBOXES")
        draw_pred_boxes([sample], 'detr', args.device, args.model_paths[0], boxes_path)
        break
    boxes_path = os.path.join(path, "BBOXES")
    draw_pred_boxes([sample], args.model_type, args.device, args.model_paths[0], boxes_path)
print('done')