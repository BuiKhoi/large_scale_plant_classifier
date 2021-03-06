from pathlib import Path
from config import get_config
from trainer import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_path", help="Path to training dataset", required=True)
    parser.add_argument("-v", "--val_path", help="Path to validation dataset", required=True)
    parser.add_argument("-load", "--load_checkpoint", help="Load checkpoint", default="")
    parser.add_argument("-best", "--save_best_only", help="The name says it all", action="store_true")
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_path = args.data_path
    conf.val_path = Path(args.val_path)
    conf.save_best_only = args.save_best_only
    learner = face_learner(conf)

    if args.load_checkpoint != "":
        print("Loading checkpoint", args.load_checkpoint)
        learner.load_state(conf, args.load_checkpoint)

    learner.train(conf, args.epochs)