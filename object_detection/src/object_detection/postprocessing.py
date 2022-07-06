import numpy as np
import os
import csv
import cv2
import pandas as pd 

def get_distance(pred, target):
    N, M = pred.shape[0], target.shape[0]
    distance = np.zeros([N, M])
    for i, one_pred in enumerate(pred):
        for j, one_tgt in enumerate(target):
            distance[i, j] = np.linalg.norm(one_pred[:2] - one_tgt[:2])

    return distance

## class agnostic nms

def nms(map_pose, class_id, score,  distance_threshold=10, confidence_threshold=0.5):
    distance = get_distance(map_pose, map_pose)
    original_id = np.arange(score.shape[0])
    filtered_map_pose = []
    filtered_class_id = []
    filtered_score= []
    filtered_ori_id = []

    while score.shape[0] > 0 and np.amax(score)>=confidence_threshold:
        idx = np.argmax(score)
        filtered_map_pose.append(map_pose[idx])
        filtered_score.append(score[idx])
        filtered_class_id.append(class_id[idx])
        filtered_ori_id.append(original_id[idx])

        idx_filter = np.flatnonzero(np.logical_or(distance[idx, :] > distance_threshold, class_id != class_id[idx]))
        map_pose = map_pose[idx_filter]
        score = score[idx_filter]
        class_id = class_id[idx_filter]
        distance = distance[idx_filter]
        distance = distance[:, idx_filter]
        original_id = original_id[idx_filter]

    filtered_map_pose = np.array(filtered_map_pose) 
    filtered_score = np.array(filtered_score) 
    filtered_class_id = np.array(filtered_class_id)

    return filtered_map_pose, filtered_class_id, filtered_score, filtered_ori_id

det_result_path = "/home/liuzhi/det_result"
postprocess_result_path = os.path.join(det_result_path, "postprocess_result")

os.makedirs(postprocess_result_path, exist_ok=True)

det_result = np.load(os.path.join(det_result_path, "det.npz"), allow_pickle=True) ##hardcoded
map_pose, class_id, conf_core, ori_id = nms(det_result["map_pose"], det_result["class_id"], det_result["conf_score"])

i = 0

for class_id_, ori_id_ in zip(class_id, ori_id):
    ori_name = os.path.join(det_result_path, f"{ori_id_}_{class_id_}.png")
    new_name = os.path.join(postprocess_result_path, f"{i}_{class_id_}.png")
    os.system(f"cp {ori_name} {new_name}")
    i+=1

final_result = {"x": map_pose[:, 0], "y": map_pose[:, 1], "z": map_pose[:, 2], "class":class_id}
pd.DataFrame(final_result).to_csv(os.path.join(postprocess_result_path, f"final_result.csv"))