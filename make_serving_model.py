from pathlib import Path
import torch
from torch.autograd import Variable
from config import get_config
from model import MobileFaceNet
import onnx
from onnx_tf.backend import prepare
from argparse import ArgumentParser


def arguments():
    parser = ArgumentParser()

    parser.add_argument("--model_file", help="Path to your model file", default="")
    parser.add_argument("--mode", help="(best|last) Mode to find your model if model file is not specified", default="best")
    parser.add_argument("--image_size", help="Size of images to be predicted by your model", default="224,224")
    parser.add_argument("--save_folder", help="Path to folder to save the transfered model", default="work_space/extracted/")
    parser.add_argument("--model_name", help="Name of your model", default="plant_classifier")

    return parser.parse_args()


def find_model_file(folder, mode):
    model_names = [m.name for m in folder.glob("model_*.pth")]
    assert len(model_names) > 0, "No model found"
    if mode == "best":
        accuracy = 0.0
        selected_model = ""
        for ss in model_names:
            i = ss.index("acc")
            j = ss.index("_", i)
            acc = float(ss[i+4:j])
            if acc > accuracy:
                accuracy = acc
                selected_model = ss
        
        assert selected_model != "", "No model found"
        return str(folder / selected_model)
    elif mode == "last":
        overall_step = 0
        selected_model = ""
        for ss in model_names:
            i = ss.index("step:")
            j = ss.index(".", i)
            step = int(ss[i+5:j])
            if step > overall_step:
                overall_step = step
                selected_model = ss
        
        assert selected_model != "", "No model found"
        return str(folder / selected_model)
    else:
        raise NotImplementedError(f"Mode '{mode}' is currently not supported")


def main(args):
    print("Parsing config")
    conf = get_config()
    model = MobileFaceNet(conf.embedding_size)
    if args.model_file != "":
        model_path = args.model_file
    else:
        model_path = find_model_file(conf.model_path, args.mode)
    
    print("Loading model", model_path)
    model.load_state_dict(torch.load(model_path))

    print("Extracting onnx model")
    img_size = [int(s) for s in args.image_size.split(",")]
    dummy_input = Variable(torch.randn(1, 3, *img_size))
    torch.onnx.export(model, dummy_input, "work_space/temp_model_onnx.onnx")

    print("Extracting tensorflow model")
    model = onnx.load("work_space/temp_model_onnx.onnx")
    tf_rep = prepare(model)
    tf_rep.export_graph(str(Path(args.save_folder)/args.model_name))

    print("Done. Bye")

if __name__ == "__main__":
    args = arguments()
    main(args)
