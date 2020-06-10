import torch, os
import numpy as np
from mex.personalised_maml.dc_data import DCData
import argparse

from mex.personalised_maml.meta_mlp import Meta


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    write_results(str(args))

    config = [
        ('linear', [1200, 960]),
        ('relu', [True]),
        ('bn', [1200]),
        ('linear', [args.n_way, 1200])
    ]
    if torch.cuda.is_available():
        dev = "cuda:x"
    else:
        dev = "cpu"
    device = torch.device(dev)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = DCData('',
                           batchsz=args.task_num,
                           n_way=args.n_way,
                           k_shot=args.k_spt,
                           k_query=args.k_qry,
                           test_index=args.test_index)

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        if step % 5 == 0:
            print('step:', step, '\ttraining acc:', accs)
            write_results('step:'+str(step)+'\ttraining acc:'+','.join([str(f) for f in accs]))

        if step % 10 == 0:
            accs = []
            for _ in range(1000 // args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            write_results('Test acc:'+','.join([str(f) for f in accs]))


def write_results(text):
    file_path = 'p_mlp.csv'
    if os.path.isfile(file_path):
        f = open(file_path, 'a')
        f.write(text + '\n')
    else:
        f = open(file_path, 'w')
        f.write(text + '\n')
    f.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=7)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=9)
    argparser.add_argument('--test_index', type=int, help='test_index', default=0)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
