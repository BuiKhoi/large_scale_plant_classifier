import os
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm
import cv2
import argparse
import numpy as np


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", help="Path to dataset folder")
    parser.add_argument("--min_set_size", help="Minimum image count for each class", default=3, type=int)
    parser.add_argument("--resize", help="Resize images for training", action="store_true")
    parser.add_argument("--image_size", help="Resize images to this size", default="224, 224")
    parser.add_argument("--make_val_data", help="Also create validation data from this dataset", action="store_true")
    parser.add_argument("--val_dir", help="Path to save validation data")
    parser.add_argument("--val_size", help="Number of validation images of each type (same, not same)", default=300)
    parser.add_argument("--remove_val_image", help="Remove val images after creating validation data", action="store_true")
    parser.add_argument("--image_suffix", help="Suffix of images in dataset", default="jpg")

    return parser.parse_args()


def process_pairs(pairs_file):
    with open(pairs_file) as fp:
        content = fp.read().split("\n")[:-1]

    images = []
    issame = []
    
    config = [int(c) for c in content[0].split(" ")]
    content = content[1:]
    for pack_count in range(config[0]):
        for img_idx in range(config[1]):
            issame.append(True)
            images.extend(content[img_idx].split("\t"))

        for img_idx in range(config[1]):
            issame.append(False)
            images.extend(content[img_idx+config[1]].split("\t"))

    return images, issame


def main(args):
    data_folder = Path(args.dataset_dir)
    data_classes = list(data_folder.glob("*"))
    assert len(data_classes) >= 50, f"You serious? {len(data_classes)} classes? Find more data to continue..."

    print("Filtering classes")
    removed_classes = 0
    new_size = [int(s) for s in args.image_size.replace(" ", "").split(",")]
    for dc in tqdm(data_classes):
        child_images = list(dc.glob(f"*.{args.image_suffix}"))
        if len(child_images) < args.min_set_size:
            rmtree(str(dc))
            removed_classes += 1
        elif args.resize:
            for ci in child_images:
                im = cv2.imread(str(ci))
                if im.shape != (*new_size, 3):
                    im = cv2.resize(im, tuple(new_size))
                    cv2.imwrite(str(ci), im)
    
    print(f"\nRemoved {removed_classes} classes")

    if args.make_val_data:
        print("Making data pairs")
        classes = [c.name for c in data_folder.glob("*")]
        same_count = args.val_size
        with open("./data/pairs.txt", "w") as fp:
            fp.write(f"1 {same_count}\n")
            for _ in range(same_count):
                cls_1 = np.random.choice(classes)

                while True:
                    img_1 = np.random.choice(list((data_folder/cls_1).glob("*.jpg")))
                    img_2 = np.random.choice(list((data_folder/cls_1).glob("*.jpg")))
                    if img_1 != img_2:
                        break
                
                fp.write(f"{str(img_1)}\t{str(img_2)}\n")
            
            for _ in range(same_count):
                while True:
                    cls_1 = np.random.choice(classes)
                    cls_2 = np.random.choice(classes)

                    if cls_1 != cls_2:
                        break

                img_1 = np.random.choice(list((data_folder/cls_1).glob("*.jpg")))
                img_2 = np.random.choice(list((data_folder/cls_2).glob("*.jpg")))
                
                fp.write(f"{str(img_1)}\t{str(img_2)}\n")

        print("Making validation data")
        from config import get_config
        from data.data_pipe import make_bin

        conf = get_config()
        rec_path = Path(args.val_dir)
        
        images_paths, issame = process_pairs("./data/pairs.txt")
        make_bin(images_paths, issame, rec_path, conf.test_transform, new_size)

        if args.remove_val_image:
            for img_path in images_paths:
                try:
                    os.remove(str(img_path))
                except FileNotFoundError:
                    pass
        
    print("Done. Bye")

if __name__ == "__main__":
    args = arguments()
    main(args)