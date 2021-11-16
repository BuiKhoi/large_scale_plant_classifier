import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from trainer import face_learner
from sklearn.metrics import classification_report

from config import get_config


def arguments():
    parser = ArgumentParser()

    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to images folder")
    parser.add_argument("--databank", required=True, help="Path to saved databank")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size to predict images")

    return parser.parse_args()


def main(args):
    images_path = Path(args.images_path)
    databank = np.load(args.databank, allow_pickle=True)
    embeddings = databank["embeddings"]
    embeddings = np.squeeze(embeddings, 1)
    labels = databank["labels"]

    conf = get_config(False)
    conf.use_mobilfacenet = True
    inferer = face_learner(conf, True)
    inferer.load_state(conf, args.model_path, False, True, absolute=True)

    files = list(images_path.rglob("*.jpg"))
    files = np.random.choice(files, 100, False)
    print(f"Doing evaluation on {len(files)} files")
    
    min_idx = []
    pred_images = []
    for f in tqdm(files):
        pred_images.append(f)
        if len(pred_images) == args.batch_size:
            pred_images = [Image.open(str(img)) for img in pred_images]
            min_pred, _ = inferer.infer(conf, pred_images, embeddings, True)
            min_idx.extend(min_pred)
            pred_images = []

    if len(pred_images) > 0:
        pred_images = [Image.open(str(img)) for img in pred_images]
        min_pred, _ = inferer.infer(conf, pred_images, embeddings, True)
        min_idx.extend(min_pred)

    gt = []
    pred = []
    for idx, f in zip(min_idx, files):
        parts = f.parts
        gt.append(parts[-2])
        pred.append(labels[idx][0])

    print("Classification report:")
    print(classification_report(gt, pred))

if __name__ == "__main__":
    args = arguments()
    main(args)
