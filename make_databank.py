from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from argparse import ArgumentParser

from config import get_config
from model import MobileFaceNet


def arguments():
    parser = ArgumentParser()

    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to data folder")
    parser.add_argument("--save_path", required=True, help="Path to save folder")
    parser.add_argument("--dataset_name", required=True, help="Name of your dataset")

    return parser.parse_args()


def main(args):
    print("Loading model")
    conf = get_config()
    model = MobileFaceNet(conf.embedding_size)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    data_folder = Path(args.data_path)
    save_folder = Path(args.save_path)
    context = args.dataset_name
    data_images = list(data_folder.rglob("*.jpg"))

    with torch.no_grad():
        embds = []
        labels = []
        print("Processing images")
        for di in tqdm(data_images):
            img = Image.open(str(di))
            embd = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
            parts = di.parts
            save_img_path = save_folder / context / parts[-2] / di.stem
            if not save_img_path.parent.exists():
                save_img_path.parent.mkdir(parents=True)
            np.save(str(save_img_path), embd.cpu())
            embds.append(embd)
            labels.append(di.stem)
        np.savez_compressed(str(save_folder/"databank"), embeddings=embds, labels=labels)
    
    print("Done. Bye")


if __name__ == "__main__":
    args = arguments()
    main(args)