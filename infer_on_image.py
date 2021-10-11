from pathlib import Path
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from trainer import face_learner

from config import get_config


def arguments():
    parser = ArgumentParser()

    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to images folder")
    parser.add_argument("--databank", required=True, help="Path to saved databank")

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
    images = [Image.open(str(img)) for img in files]
    min_idx, _ = inferer.infer(conf, images, embeddings, True)

    for idx, f in zip(min_idx, files):
        parts = f.parts
        print(f"Spiece: {parts[-2]}, File: {parts[-1]}, Predicted: {labels[idx]}")


if __name__ == "__main__":
    args = arguments()
    main(args)
