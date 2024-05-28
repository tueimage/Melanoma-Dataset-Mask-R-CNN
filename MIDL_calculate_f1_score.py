import os
import json
from shapely.geometry import Polygon, Point
import numpy as np
import argparse

def process_geojsons(folder, dict_classes, NN192 = False):
    centroid_dictionary = {}
    instance = 0
    filenames_to_process = [ # only the files that are in the test set
    "metastasis_image_181_cell", "metastasis_image_182_cell", "metastasis_image_183_cell",
    "metastasis_image_184_cell", "metastasis_image_185_cell", "metastasis_image_186_cell",
    "metastasis_image_187_cell", "metastasis_image_188_cell", "metastasis_image_189_cell",
    "metastasis_image_190_cell", "metastasis_image_191_cell", "metastasis_image_192_cell",
    "metastasis_image_193_cell", "metastasis_image_194_cell", "metastasis_image_195_cell",
    "metastasis_image_196_cell", "metastasis_image_197_cell", "metastasis_image_198_cell",
    "metastasis_image_199_cell", "metastasis_image_200_cell", "primary_image_081_cell",
    "primary_image_082_cell", "primary_image_083_cell", "primary_image_084_cell",
    "primary_image_085_cell", "primary_image_086_cell", "primary_image_087_cell",
    "primary_image_088_cell", "primary_image_089_cell", "primary_image_090_cell",
    "primary_image_091_cell", "primary_image_092_cell", "primary_image_093_cell",
    "primary_image_094_cell", "primary_image_095_cell", "primary_image_096_cell",
    "primary_image_097_cell", "primary_image_098_cell", "primary_image_099_cell",
    "primary_image_100_cell"
    ]

    for geojson in os.listdir(folder):
        geojson_basename =  os.path.basename(geojson).split('.')[0]
        if geojson_basename not in filenames_to_process:
            continue
        new_filepath = os.path.join(folder, geojson)
        if new_filepath.endswith('_cell.geojson'):  # Process only GeoJSON files
            with open(new_filepath) as f:
                data = json.load(f)
                geojson_name = geojson.split('.')[0]
                features_list = []

                if 'nucleosGeometry' in data['features'][0]:
                    NN192 = True
                else:
                    NN192 = False

                if NN192 == True:
                    features = data.get('features', [])
                    for feature in features:
                        if feature['properties']['objectType'] == 'cell':
                             # NN192 outputs a rectangle without a class in metastasis_image_187_cell, metastasis_image_190_cell, metastasis_image_192_cell, primary_image_081_cell, primary_image_083_cell, primary_image_089_cell, primary_image_091_cell, primary_image_096_cell, 
                            category = feature.get('properties', {}).get('classification', {}).get('name', 'cell_other')
                            category = dict_classes.get(category, 'cell_other')
                            segmentation = feature.get('nucleusGeometry', {}).get('coordinates', [])
                            polygon = Polygon(segmentation[0])
                            features_list.append({
                                'filename': geojson_name,
                                'category': category,
                                'centroid': polygon.centroid,
                                'score': 1.0
                                })
                            
                else: 
                    features = data.get('features', [])
                    for feature in features:
                        category = feature.get('properties', {}).get('classification', {}).get('name', 'cell_other')
                        category = dict_classes.get(category, 'cell_other')
                        geometry_type = feature["geometry"]["type"]
                        geometry = feature["geometry"]
                        properties = feature['properties']
                        if 'classification' in properties and 'score' in properties['classification']:
                                score = properties['classification']['score']
                            # If not, check if 'type_prob' key exists for hovernet
                        elif 'type_prob' in properties:
                                score = properties['type_prob']
                            # If neither condition is met, default the score to 1.0
                        else:
                                score = 1.0
                        if geometry_type == "Polygon":
                            polygons = [geometry["coordinates"]]
                        elif geometry_type == "MultiPolygon":
                            polygons = geometry["coordinates"]
                        else:
                            continue
                        for polygon_coords in polygons:
                            exterior_coords = polygon_coords[0]
                            interior_coords = polygon_coords[1:]
                            exterior_ring = [tuple(coord) for coord in exterior_coords]
                            interior_rings = [[tuple(coord) for coord in interior] for interior in interior_coords]

                            polygon = Polygon(exterior_ring, interior_rings)
                            centroid = polygon.centroid
                            features_list.append({
                                'filename': geojson_name, 
                                'category': category,
                                'centroid': centroid, 
                                'score': score})
                centroid_dictionary[geojson_basename] = features_list
    return centroid_dictionary

def calculate_centroid_distance(dict_ground_truth, dict_pred):
    results_dict = {}
    pred_structure = {}

    # Organize predictions by (filename, category) for faster access
    for pred_geojson, pred_features in dict_pred.items():
        for feature in pred_features:
            key = (feature['filename'], feature['category'])
            if key not in pred_structure:
                pred_structure[key] = []
            pred_structure[key].append(feature)
    
    # Process each ground truth feature individually
    for gt_geojson, gt_features in dict_ground_truth.items():
        print('Processing', gt_geojson)
        
        if gt_geojson not in results_dict:
            results_dict[gt_geojson] = []
        
        for gt_feature in gt_features:
            match_key = (gt_feature['filename'], gt_feature['category'])
            eligible_predictions = []
            
            # Check if there are any predictions matching the current ground truth feature's filename and category
            if match_key in pred_structure:
                for pred_feature in pred_structure[match_key]:
                    # Calculate the distance between centroids of the gt_feature and pred_feature
                    distance = gt_feature['centroid'].distance(pred_feature['centroid'])
                    # Filter predictions within the specified distance threshold
                    if distance < 15:
                        eligible_predictions.append({
                            'pred_geojson': pred_feature['filename'],
                            'gt_category': gt_feature['category'],
                            'pred_category': pred_feature['category'],
                            'distance': distance,
                            'pred_score': pred_feature['score'],
                            'pred_feature': pred_feature,
                        })
            
            # Sort the eligible predictions first by descending pred_score, then by ascending distance
            eligible_predictions.sort(key=lambda x: (-x['pred_score'], x['distance']))
            
            # If there are eligible predictions, take the best match based on the sorting criteria
            if eligible_predictions:
                best_match = eligible_predictions[0]  # The best match for this gt_feature
                results_dict[gt_geojson].append(best_match)

                # Remove the best match from the pred_structure to avoid duplicate matches
                pred_structure[match_key].remove(best_match['pred_feature'])

    
    return results_dict

def calculate_classification_metrics(results_mask_rcnn, dict_ground_truth, dict_mask_rcnn):
    # Extraction process remains the same
    pred_tp = [match['pred_category'] for matches in results_mask_rcnn.values() for match in matches]
    ground_truth = [match['category'] for matches in dict_ground_truth.values() for match in matches]
    pred_all = [match['category'] for matches in dict_mask_rcnn.values() for match in matches]

    # Calculation of counts remains the same
    gt_dict = dict(zip(*np.unique(ground_truth, return_counts=True)))
    pred_dict = dict(zip(*np.unique(pred_all, return_counts=True)))
    tp_dict = dict(zip(*np.unique(pred_tp, return_counts=True)))

    # Initialize variables for micro F1 calculation
    micro_TP, micro_FP, micro_FN = 0, 0, 0
    results = {}

    for category in np.unique(list(gt_dict.keys()) + list(pred_dict.keys())):
        TP = tp_dict.get(category, 0)
        FP = pred_dict.get(category, 0) - TP
        FN = gt_dict.get(category, 0) - TP

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        results[category] = {
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1_score': f1_score
        }

    # Micro F1 calculation
    micro_precision = micro_TP / (micro_TP + micro_FP) if micro_TP + micro_FP > 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if micro_TP + micro_FN > 0 else 0
    micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    # Macro F1 calculation
    macro_f1_score = np.mean([metrics['f1_score'] for metrics in results.values()])

    # Add micro and macro F1 scores to the results
    results['micro'] = {
        'precision': micro_precision, 'recall': micro_recall, 'f1_score': micro_f1_score
    }
    results['macro'] = {
        'f1_score': macro_f1_score
    }

    return results

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process GeoJSON files and calculate metrics.")
    
    # Adding arguments
    parser.add_argument("ground_truth_folder", type=str, help="Folder containing ground truth GeoJSON files.")
    parser.add_argument("prediction_folder", type=str, help="Folder containing prediction GeoJSON files.")

    # Parse the arguments
    args = parser.parse_args()

    # Dictionary for class mapping
    dict_classes = { # classes of the NN192 and HoverNet PanNuke models
        # NN192
        'Tumor' : 'cell_tumor',        
        "Stroma" : 'cell_stroma',
        'Immune cells' : 'cell_lymphocyte',
        'Other' : 'cell_other',

        # HoverNet PanNuke already code to geojson with the following classes
        'cell_tumor' : 'cell_tumor',
        'cell_stroma' : 'cell_stroma',
        'cell_lymphocyte' : 'cell_lymphocyte',
        'cell_other' : 'cell_other',

        # melanoma dataset extra classes
        'cell_plasma_cell' : 'cell_lymphocyte',
        "cell_histiocyte" : 'cell_other',
        "cell_melanophage" : 'cell_other',
        "cell_endothelium" : 'cell_other',
        "cell_epithelium" : 'cell_other',
        "cell_neutrophil" : 'cell_other',
        "cell_apoptosis" : 'cell_other', 
        }   
 
    # Process the GeoJSON files using the NN192 flag from arguments
    dict_ground_truth = process_geojsons(args.ground_truth_folder, dict_classes)
    dict_mask_rcnn = process_geojsons(args.prediction_folder, dict_classes)
    
    # Calculate matches and metrics
    results_mask_rcnn = calculate_centroid_distance(dict_ground_truth, dict_mask_rcnn)
    metrics_mask_rcnn = calculate_classification_metrics(results_mask_rcnn, dict_ground_truth, dict_mask_rcnn)
    
    # Print or otherwise output the metrics
    print(metrics_mask_rcnn)

# To run the script, use the following command:
# python MIDL_calculate_f1_score.py /path/to/ground_truth_folder /path/to/prediction_folder
