import albumentations as A

def transformations(aug_name: str):
    if aug_name == "none":
        return None

    if aug_name == "basic":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    if aug_name == "weather":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomFog(p=0.3),
                A.RandomRain(p=0.3),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(16, 64),
                    hole_width_range=(16, 64),
                    p=0.5
                ),
                A.ColorJitter(brightness=0.1, contrast=0.2, p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    raise ValueError(f"Unknown augmentation: {aug_name}")