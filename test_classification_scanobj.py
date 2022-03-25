"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""
import torch
from data_utils.ScanObjectNNLoader import ScanObjectNN
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_category', default=15, type=int, help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pretrained', help='log root')
    parser.add_argument('--data_type', type=str, default='OBJ_NOBG', help='data type')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    return parser.parse_args()

def test(model, loader, num_class=15):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        pred, feat = classifier(points.cuda())
        if len(pred.shape) == 3:
            pred = pred.mean(dim=1)

        pred_choice = pred.data.max(1)[1].cpu()

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification_scanobj/' + args.log_dir + '/' + args.data_type

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    if args.data_type == 'OBJ_NOBG':
        data_path = '../data/scanobjectnn/main_split_nobg/'
    elif args.data_type == 'hardest' or 'OBJ_BG': 
        data_path = '../data/scanobjectnn/main_split/'
    else:
        raise NotImplementedError()
    test_dataset = ScanObjectNN(root=data_path, args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module('riconv2_cls')

    classifier = model.get_model(num_class, 1)
    classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
