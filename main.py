import argparse

from configs import get_cfg
from obstructors import get_obstructor
from utils.random_helper import set_random_seed, set_cudnn

def main(args):
    cfg = get_cfg(args)
    set_random_seed(cfg.RANDOM.SEED)
    set_cudnn(cfg.RANDOM.DETERMINISTIC)
    
    obstructor = get_obstructor(cfg, world_rank=0)
    if args.evaluate_only:
        obstructor.test()
        obstructor.load_model(ckpt_path=args.load_ckpt)
        obstructor.test()
    elif args.fewshot_only:
        obstructor.test_fewshot()
    else:
        obstructor.test()
        obstructor.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning to Obstruct')
    parser.add_argument('--config-obstructor', default=None, type=str,
                        help='config yaml file for the obstructor')
    parser.add_argument('--config-learner-def', default=None, type=str,
                        help='config yaml file for the learner in the obstrcutor inner loop')
    parser.add_argument('--config-learner-atk', default='configs/learner/base.yaml',
                        help='config yaml file for the learner for evaluation')
    parser.add_argument('--config-dataset', default='configs/data/imagenet.yaml',
                        help='config yaml file for the dataset')
    parser.add_argument('--fewshot-only', action='store_true',
                        help='Only run the few shot algorithm without obstruct learning')
    parser.add_argument('--load-ckpt', type=str,
                        help='')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    
    args = parser.parse_args()
    main(args)
