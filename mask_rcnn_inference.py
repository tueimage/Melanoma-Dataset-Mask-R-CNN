import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.layers import nms
import argparse

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
        
def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.RPN.NMS_THRESH = 0.5
    
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    return cfg

def process_images(images_directory, output_directory, cfg):
    class_id_to_label = {
        0: ['cell_tumor', [255, 0, 0]],
        1: ['cell_stroma', [255, 165, 0]],
        2: ['cell_lymphocyte', [128, 0, 128]],
        3: ['cell_other', [0, 0, 255]]
    }

    predictor = DefaultPredictor(cfg)
        
    for root, dirs, files in os.walk(images_directory):
        for image_name in tqdm(files):
            if image_name.endswith('.png'): 
                base_name_without_extension = os.path.splitext(os.path.basename(image_name))[0]


                filenames_to_process = [ # only the files that are in the test set
                    "metastasis_image_181", "metastasis_image_182", "metastasis_image_183",
                    "metastasis_image_184", "metastasis_image_185", "metastasis_image_186",
                    "metastasis_image_187", "metastasis_image_188", "metastasis_image_189",
                    "metastasis_image_190", "metastasis_image_191", "metastasis_image_192",
                    "metastasis_image_193", "metastasis_image_194", "metastasis_image_195",
                    "metastasis_image_196", "metastasis_image_197", "metastasis_image_198",
                    "metastasis_image_199", "metastasis_image_200", "primary_image_081",
                    "primary_image_082", "primary_image_083", "primary_image_084",
                    "primary_image_085", "primary_image_086", "primary_image_087",
                    "primary_image_088", "primary_image_089", "primary_image_090",
                    "primary_image_091", "primary_image_092", "primary_image_093",
                    "primary_image_094", "primary_image_095", "primary_image_096",
                    "primary_image_097", "primary_image_098", "primary_image_099",
                    "primary_image_100"
                    ]
                
                # Check if the base name without extension is in the list of filenames to process
                if base_name_without_extension not in filenames_to_process:
                    continue

                image_path = os.path.join(root, image_name)
                im = cv2.imread(image_path)
                outputs = predictor(im)

                # Apply NMS
                boxes = outputs["instances"].pred_boxes.tensor
                scores = outputs["instances"].scores
                classes = outputs["instances"].pred_classes
                pred_masks = outputs["instances"].pred_masks.cpu().numpy()
                
                keep_indices = nms(boxes, scores, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                classes = classes[keep_indices]
                pred_masks = pred_masks[keep_indices.cpu().numpy()]

                # Convert segmentation masks to polygons and write to GeoJSON
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }

                for i in range(len(classes)):
                    mask = pred_masks[i].astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        polygon = [[int(pt[0]), int(pt[1])] for pt in contour[:, 0]]
                        if len(polygon) >= 3:
                            polygon.append(polygon[0])
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [polygon]
                                },
                                "properties": {
                                    "objectType": "annotation",
                                    "classification": {
                                        "name": class_id_to_label[classes[i].item()][0],
                                        "color": class_id_to_label[classes[i].item()][1],
                                        "score" : scores[i].item()
                                    }
                                },
                                "measurements": [
                                    {
                                        "name": "score",
                                        "value": scores[i].item(),
                                        "unit": "percentage"
                                    }
                                ]
                            }
                            geojson_data["features"].append(feature)
                        
                      
                geojson_folder_path = os.path.join(output_directory)
                if not os.path.exists(geojson_folder_path):
                    os.makedirs(geojson_folder_path)

                base_name = os.path.basename(image_path)
                file_name_without_extension = os.path.splitext(base_name)[0]
                geojson_filename = os.path.join(geojson_folder_path, f"{file_name_without_extension}_cell.geojson")
                with open(geojson_filename, 'w') as f:
                    json.dump(geojson_data, f, cls=NumpyEncoder)

            # Save the GeoJSON in the specified separate directory
            base_name = os.path.basename(image_path)
            file_name_without_extension = os.path.splitext(base_name)[0]
            geojson_filename = os.path.join(geojson_folder_path, (file_name_without_extension + '_cell.geojson'))
            with open(geojson_filename, 'w') as f:
                json.dump(geojson_data, f, cls=NumpyEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using Detectron2 and output GeoJSON.")
    parser.add_argument("images_directory", help="Directory containing images to process.")
    parser.add_argument("output_directory", help="Directory where GeoJSON files will be saved.")
    parser.add_argument("weights_path", help="Path to the model weights.")

    args = parser.parse_args()

    cfg = setup_cfg(args.weights_path)
    process_images(args.images_directory, args.output_directory, cfg)

# to use run python mask_rcnn_inference.py /path/to/images /path/to/output /path/to/model_weights.pth
