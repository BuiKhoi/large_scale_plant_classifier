from pathlib import Path
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from trainer import face_learner
import pandas as pd

from config import get_config


def arguments():
    parser = ArgumentParser()

    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--images_path", required=True, help="Path to images folder")
    parser.add_argument("--databank", required=True, help="Path to saved databank")

    return parser.parse_args()


def do_test_images_list(files, inferer, conf, embeddings, labels):
    report = []
    images = [Image.open(str(img)) for img in files]
    min_idx, _ = inferer.infer(conf, images, embeddings, True)

    for idx, f in zip(min_idx, files):
        parts = f.parts
        rp = [parts[-1], parts[-2], labels[idx][0]]
        if parts[-2] == labels[idx][0]:
            rp.append(1)
        else:
            rp.append(0)
        report.append(rp)
        # print(f"Spiece: {parts[-2]}, File: {parts[-1]}, Predicted: {labels[idx]}")

    return report


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
    preds = []

    test_pack = []
    for f in files:
        test_pack.append(f)
        if len(test_pack) == 10:
            preds.extend(do_test_images_list(test_pack, inferer, conf, embeddings, labels))
            test_pack = []
    
    if len(test_pack) > 0:
        preds.extend(do_test_images_list(test_pack, inferer, conf, embeddings, labels))

    columns = ["file_name", "spiece", "predicted", "correct"]
    report_df = pd.DataFrame(data=preds, columns=columns)
    report_df.to_csv("results/report.csv", index=False)

    print(sum(report_df.correct.values.tolist())/len(report_df))


if __name__ == "__main__":
    args = arguments()
    main(args)
