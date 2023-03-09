import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        HQ_path = os.path.join(args.save_path, 'HQ')
        SHQ_path = os.path.join(args.save_path, 'SHQ')
        LQ_path = os.path.join(args.save_path, 'LQ')
        pred_path = os.path.join(args.save_path, 'pred')

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        if not os.path.exists(HQ_path):
            os.makedirs(HQ_path)
            print('Create path : {}'.format(HQ_path))
        if not os.path.exists(SHQ_path):
            os.makedirs(SHQ_path)
            print('Create path : {}'.format(SHQ_path))
        if not os.path.exists(LQ_path):
            os.makedirs(LQ_path)
            print('Create path : {}'.format(LQ_path))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
            print('Create path : {}'.format(pred_path))


    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.train_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0, help="0 | 1")
    parser.add_argument('--train_path', type=str, default='train_path')
    parser.add_argument('--validate_path', type=str, default='valitate_path')
    parser.add_argument('--test_path', type=str, default='test_path')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--validate_patient', type=str, default='LDCT')
    parser.add_argument('--test_patient', type=str, default='LDCT')
    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-1024.0)
    parser.add_argument('--trunc_max', type=float, default=3072.0)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--patch_n', type=int, default=20)
    parser.add_argument('--patch_size', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=4000)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=48000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    # print(args)
    main(args)
