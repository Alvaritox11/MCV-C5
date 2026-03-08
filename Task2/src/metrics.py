import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.coco_gt = self._build_coco_gt(dataset)
        self.results = []

    def _build_coco_gt(self, dataset):
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
        }

        anno_id = 1
        for i in range(len(dataset)):
            sample = dataset.samples[i]
            coco_dict["images"].append({
                "id": i, "width": 1242, "height": 375, "file_name": sample["path"]
            })

            for anno in sample["annos"]:
                cat_id = dataset.class_map[anno["label"]]
                rle = anno["rle"]
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                
                coco_dict["annotations"].append({
                    "id": anno_id,
                    "image_id": i,
                    "category_id": cat_id,
                    "bbox": anno["bbox"],
                    "segmentation": rle, # Required for 'segm' evaluation
                    "area": anno["bbox"][2] * anno["bbox"][3],
                    "iscrowd": 0
                })
                anno_id += 1

        coco_gt = COCO()
        coco_gt.dataset = coco_dict
        coco_gt.createIndex()
        return coco_gt

    def update(self, predictions):
        self.results.extend(predictions)

    def summarize(self):
        print(f"Evaluating segmentations on {len(self.results)} predictions...")
        if not self.results: return None, 0.0, 0.0

        coco_results = []
        for res in self.results:
            img_id = res["image_id"]
            
            for i in range(len(res["scores"])):
                # Convert predicted binary mask to COCO RLE format
                mask = res["masks"][i]
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                
                coco_results.append({
                    "image_id": img_id,
                    "category_id": res["labels"][i],
                    "segmentation": rle,
                    "score": res["scores"][i]
                })

        if not coco_results: return None, 0.0, 0.0

        coco_dt = self.coco_gt.loadRes(coco_results)
        
        # KEY CHANGE: iouType is now 'segm' instead of 'bbox'
        coco_eval = COCOeval(self.coco_gt, coco_dt, "segm") 
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        map_pedestrian = 0.0
        map_car = 0.0
        cat_ids = coco_eval.params.catIds

        if 1 in cat_ids:  # person
            idx = cat_ids.index(1)
            s = coco_eval.eval['precision'][:, :, idx, 0, 2]
            if len(s[s > -1]) > 0: map_pedestrian = float(np.mean(s[s > -1]))

        if 3 in cat_ids:  # car
            idx = cat_ids.index(3)
            s = coco_eval.eval['precision'][:, :, idx, 0, 2]
            if len(s[s > -1]) > 0: map_car = float(np.mean(s[s > -1]))

        return stats, map_car, map_pedestrian