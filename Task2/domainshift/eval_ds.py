import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluatorOxfordPet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.coco_gt = self._build_coco_gt(dataset)
        self.results = []
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset.categories}

    def _build_coco_gt(self, dataset):
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": dataset.categories
        }

        anno_id = 1
        for i in range(len(dataset)):
            image, target = dataset[i]
            image_id = int(target["image_id"].item())

            if hasattr(image, "size"):
                width, height = image.size
            else:
                height, width = target["masks"].shape[-2:]

            coco_dict["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": dataset.samples[i]["path"]
            })

            gt_masks = target["masks"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()

            for j in range(len(gt_masks)):
                mask = gt_masks[j].astype(np.uint8)
                if mask.sum() == 0:
                    continue

                rle = mask_utils.encode(np.asfortranarray(mask))
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")

                bbox = mask_utils.toBbox(rle).tolist()
                area = float(mask_utils.area(rle))

                coco_dict["annotations"].append({
                    "id": anno_id,
                    "image_id": image_id,
                    "category_id": int(gt_labels[j]),
                    "bbox": bbox,
                    "segmentation": rle,
                    "area": area,
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
        if not self.results:
            return None, {}

        coco_results = []
        for res in self.results:
            img_id = int(res["image_id"])

            for i in range(len(res["scores"])):
                mask = res["masks"][i].astype(np.uint8)
                if mask.sum() == 0:
                    continue

                rle = mask_utils.encode(np.asfortranarray(mask))
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.append({
                    "image_id": img_id,
                    "category_id": int(res["labels"][i]),
                    "segmentation": rle,
                    "score": float(res["scores"][i])
                })

        if not coco_results:
            return None, {}

        coco_dt = self.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        ap_by_class = self._compute_ap_by_class(coco_eval)
        return stats, ap_by_class

    def _compute_ap_by_class(self, coco_eval):
        ap_by_class = {}
        cat_ids = list(coco_eval.params.catIds)
        precisions = coco_eval.eval["precision"]  # [T, R, K, A, M]

        for cat_id in cat_ids:
            idx = cat_ids.index(cat_id)
            s = precisions[:, :, idx, 0, -1]
            valid = s[s > -1]
            ap_by_class[self.cat_id_to_name[cat_id]] = float(np.mean(valid)) if len(valid) > 0 else 0.0

        return ap_by_class