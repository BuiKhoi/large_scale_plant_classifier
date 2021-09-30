from pathlib import Path
from config import get_config
from data.data_pipe import make_bin
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-d", "--data_path", help="Path to data folder",required=True, type=str)
    parser.add_argument("-p", "--pairs_path", help="txt file for pairs",default='/content/training_data/pairs.txt', type=str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = Path(args.data_path)

    images_paths, issame = process_pairs(args.pairs_path)
    make_bin(images_paths, issame, rec_path/"val", conf.test_transform)