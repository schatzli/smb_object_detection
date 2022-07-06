import numpy as np

def get_distance(pred, target):
    N, M = pred.shape[0], target.shape[0]
    distance = np.zeros([N, M])
    for i, one_pred in enumerate(pred):
        for j, one_tgt in enumerate(target):
            distance[i, j] = np.linalg.norm(one_pred[:2] - one_tgt[:2])

    return distance

## class agnostic nms

def nms(map_pose, class_id, score, distance_threshold=10, confidence_threshold=0.5):
    distance = get_distance(map_pose, map_pose)
    filtered_map_pose = []
    filtered_class_id = []
    filtered_score= []

    while score.shape[0] > 0 and np.amax(score)>=confidence_threshold:
        idx = np.argmax(score)
        filtered_map_pose.append(map_pose[idx])
        filtered_score.append(score[idx])
        filtered_class_id.append(class_id[idx])

        idx_filter = np.flatnonzero(np.logical_or(distance[idx, :] > distance_threshold, class_id != class_id[idx]))
        map_pose = map_pose[idx_filter]
        score = score[idx_filter]
        class_id = class_id[idx_filter]
        distance = distance[idx_filter]
        distance = distance[:, idx_filter]

    filtered_map_pose = np.array(filtered_map_pose) 
    filtered_score = np.array(filtered_score) 
    filtered_class_id = np.array(filtered_class_id)

    return filtered_map_pose, filtered_class_id, filtered_score


det_result = np.load("/home/liuzhi/det.npz") ##hardcoded
map_pose, class_id, conf_core = nms(det_result["map_pose"], det_result["class_id"], det_result["conf_score"])

import pdb; pdb.set_trace()