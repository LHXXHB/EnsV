import argparse
import os
import random
import math
import os.path as osp
import copy
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import pre_process as prep
from data_list import ImageList
from classifier import ImageClassifier, ImageClassifierMDD, ImageClassifierAFN
from backbone import get_model

def entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 

def init_model(args):        
    backbone = get_model(args.net, pretrain=True)
    #pool_layer = nn.Identity() if args.no_pool else None
    if args.method == 'MDD':
        classifier = ImageClassifierMDD(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim, width=args.width)
    elif args.method == 'SAFN':
        classifier = ImageClassifierAFN(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim)
    else:
        classifier = ImageClassifier(backbone, args.class_num, bottleneck_dim=args.bottleneck_dim)
    return classifier

def load_ckpt(args):
    ckpt_path = args.ckpt_path
    ckpt_dict = torch.load(ckpt_path)
    filtered_state_dict = OrderedDict()
    for k in ckpt_dict:
        if 'backbone.fc' in k:
            pass
        else:
            filtered_state_dict[k] = ckpt_dict[k]
    return filtered_state_dict

def load_ckpt_mdd(args):
    ckpt_path = args.ckpt_path
    ckpt_dict = torch.load(ckpt_path)
    filtered_state_dict = OrderedDict()
    for k in ckpt_dict:
        if 'fc' in k or 'adv' in k:
            pass
        elif 'bottleneck.1' in k:
            newk = k.replace('bottleneck.1', 'bottleneck.0')
            filtered_state_dict[newk] = ckpt_dict[k]
        elif 'bottleneck.2' in k:
            newk = k.replace('bottleneck.2', 'bottleneck.1')
            filtered_state_dict[newk] = ckpt_dict[k]
        else:
            filtered_state_dict[k] = ckpt_dict[k]
    return filtered_state_dict

def test(config, args, net, dset_path):
    prep_dict = {}
    prep_dict["test_tgt"] = prep.image_test(**config["prep"]["params"])
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test_tgt"]["batch_size"]
    dsets["test_tgt"] = ImageList(open(dset_path).readlines(), transform=prep_dict["test_tgt"])
    dset_loaders["test_tgt"] = DataLoader(dsets["test_tgt"], batch_size=test_bs, shuffle=False, num_workers=4, drop_last=False)

    net = net.cuda()
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["test_tgt"])
        for i in range(len(dset_loaders["test_tgt"])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])    
    all_label = all_label.long()
    all_sfmx = nn.Softmax(dim=-1)(all_output)
    mean_sfmx = all_sfmx.mean(dim=0)

    # entropy: min is best
    mean_ent = torch.mean(entropy(all_sfmx)).item()

    # im: max is best
    mean_div = -torch.sum(mean_sfmx * torch.log(mean_sfmx + 1e-5)).item()
    mean_im = mean_div - mean_ent

    # corr-c: min is best
    ori_corr = torch.mm(all_sfmx.t(), all_sfmx)
    sfmxcorr = ori_corr.diag().sum().item() / ((ori_corr**2).sum()**0.5).item()

    # snd: max is best
    ori_normalized = F.normalize(all_output)
    ori_mat = torch.matmul(ori_normalized, ori_normalized.t()) / 0.05
    ori_mask = torch.eye(ori_mat.size(0), ori_mat.size(0)).bool()
    ori_mat.masked_fill_(ori_mask, -1 / 0.05)
    snd = entropy(ori_mat.softmax(dim=-1)).mean().item()

    log_str = "Testing accuracy: {:.4f}, entropy is {:.4f}, im is {:.4f}, corr is {:.4f}, snd is {:.4f}.\n".format(accuracy, mean_ent, mean_im, sfmxcorr, snd)
    acc_score = accuracy

    if config["dataset"] == "visda":
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)
        cls_acc_str = ' '.join(['{:.4f}'.format(x) for x in acc.tolist()])
        log_str = "Testing per-class accuracy: {}, mean-class accuracy: {:.4f}, mean accuracy: {:.4f}, mean entropy is {:.4f}.\n".format(cls_acc_str, np.mean(acc), accuracy, mean_ent)
        acc_score = np.mean(acc)

    config["task_hyper_log"].write(log_str)
    config["task_hyper_log"].flush()
    print(log_str)
    if config["task_pred_file"] is not None:
        torch.save(all_output, config["task_pred_file"])
    return mean_ent, all_output, all_label, acc_score

def load_candidate(config, args, hyper_str):
    args.ckpt_path = os.path.join('./ckpts/'+args.da, args.method, config['name'], hyper_str+'_final.pt')
    if args.da == '2d_uda':
        args.bottleneck_dim = int(hyper_str.split('_')[1])
        args.width = args.bottleneck_dim
    net = init_model(args)
    if args.method == 'MDD':
        net.load_state_dict(load_ckpt_mdd(args))
    else:
        net.load_state_dict(load_ckpt(args))
    return net

def ensv(config, args):
    if args.da == '2d_uda':
        hyper_list = config['2d_hyper_list'][args.method]
    else:
        hyper_list = config['hyper_list'][args.method]
    
    config["task_path"] = os.path.join(config["save_dir"], args.method, config['name'])
    if not osp.exists(config["task_path"]):
        os.system('mkdir -p '+config["task_path"])
   
    acc_list = []
    for idx in range(len(hyper_list)):
        hyper_str = hyper_list[idx]
        if not osp.exists(osp.join(config["task_path"], hyper_str)):
            os.system('mkdir -p '+osp.join(config["task_path"], hyper_str))
        config["task_hyper_log"] = open(osp.join(config["task_path"], hyper_str, hyper_str+"_log.txt"), "a+")
        config["task_pred_file"] = osp.join(config["task_path"], hyper_str, hyper_str+"_pred.pt")
        setting_str = "dset: {}, src: {}, tgt: {}, method: {}, net: {}, hyperparameter: {}.\n"\
        .format(args.dset, names[args.s], names[args.t], args.method, args.net, hyper_str)
        config["task_hyper_log"].write(setting_str)
        config["task_hyper_log"].flush()
        net = load_candidate(config, args, hyper_str)
        ent, logits, labels, acc_score = test(config, args, net, args.t_dset_path)
        acc_list.append(acc_score)
        if idx == 0:
            ensem_pred = nn.Softmax(dim=-1)(logits)
        else:
            ensem_pred += nn.Softmax(dim=-1)(logits)

    print('********************Model Ensembling***************')
    ensem_pred /= len(hyper_list)
    _, ensem_pl = torch.max(ensem_pred, 1)
    ensem_acc = torch.sum(torch.squeeze(ensem_pl).float() == labels.float()).item() / float(labels.size()[0])    
    log_str = "Ensemble accuracy: {:.4f}.\n".format(ensem_acc)
    ensem_acc_score = ensem_acc

    if config["dataset"] == "visda":
        matrix = confusion_matrix(labels.long(), torch.squeeze(ensem_pl).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)
        cls_acc_str = ' '.join(['{:.4f}'.format(x) for x in acc.tolist()])
        log_str = "Ensemble per-class accuracy: {}, mean-class accuracy: {:.4f}, mean accuracy: {:.4f}.\n".format(cls_acc_str, np.mean(acc), ensem_acc)
        ensem_acc_score = np.mean(acc)
    print(log_str)

    print('********************Model Selection***************')
    score_list = []
    for idx in range(len(hyper_list)):
        hyper_str = hyper_list[idx]
        config["task_pred_file"] = osp.join(config["task_path"], hyper_str, hyper_str+"_pred.pt")
        idx_pred = torch.load(config["task_pred_file"])
        idx_pl = torch.argmax(idx_pred, dim=-1)
        idx_ensem_acc = (ensem_pl == idx_pl).sum()/idx_pred.shape[0]
        score_list.append(idx_ensem_acc.item())
        setting_str = "dset: {}, src: {}, tgt: {}, method: {}, net: {}, hyperpara: {}, tgtAcc: {:.4f}, ensAcc: {:.4f}, plAcc: {:.4f}.\n"\
        .format(args.dset, names[args.s], names[args.t], args.method, args.net, hyper_str, acc_list[idx], ensem_acc_score, idx_ensem_acc.item())
        print(setting_str)
        config["task_avg_log"].write(setting_str)
        config["task_avg_log"].flush()
    best_index = score_list.index(max(score_list))
    worst_index = score_list.index(min(score_list))
    best_acc = acc_list[best_index]
    worst_acc = acc_list[worst_index]
    log_str = "dset: {}, src: {}, tgt: {}, method: {}, net: {}, numOfHyper: {}, bestAcc: {:.4f}, bestIdx: {}, worstAcc: {:.4f}, worstIdx: {}.\n"\
    .format(args.dset, names[args.s], names[args.t], args.method, args.net, len(hyper_list), best_acc, best_index, worst_acc, worst_index)
    print(log_str)
    config["task_avg_log"].write(setting_str)
    config["task_avg_log"].flush()
    return best_index, worst_index, best_acc, worst_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model selection for unsupervised domain adaptation')

    # task parameters
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home', 'DomainNet126'], help="dataset")
    parser.add_argument('--seed', type=int, default=123, help="seed")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', '2d_uda', 'opda'])

    # model parameters
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--width', type=int, default=2048, help="for mdd")
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'MCC', 'BNM', 'MDD', 'ATDOC', 'SAFN', 'PADA'])

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    s = args.s
    t = args.t
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        if args.da in {'uda', '2d_uda'}:
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list.txt'
        elif args.da == 'pda':
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_25_list.txt'
        args.class_num = 65

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        if args.da in {'uda', '2d_uda'}:
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list.txt'
        elif args.da == 'pda':
            args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_10_list.txt'
        args.class_num = 31

    if args.dset == 'visda':
        names = ['training', 'validation']
        args.s_dset_path = './data/visda17/train_list.txt'
        args.t_dset_path = './data/visda17/validation_list.txt'
        args.class_num = 12
        
    config = {}
    config['hyper_list'] = {
                            'ATDOC': ['0.02', '0.05', '0.1', '0.2', '0.5', '1.0', '2.0'],
                            'BNM': ['0.02', '0.05', '0.1', '0.2', '0.5', '1.0', '2.0'],
                            'CDAN': ['0.05', '0.1', '0.2', '0.5', '1.0', '2.0', '5.0'],
                            'MCC': ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0'],
                            'MDD': ['0.5', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0'],
                            'SAFN': ['0.002', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2'],
                            'PADA': ['0.05', '0.1', '0.2', '0.5', '1.0', '2.0', '5.0']
                            }

    config['2d_hyper_list'] = {
                            'MCC': ['1.0_256', '1.0_512', '1.0_1024', '1.0_2048', '1.5_256', '1.5_512', '1.5_1024', '1.5_2048', 
                                    '2.0_256', '2.0_512', '2.0_1024', '2.0_2048', '2.5_256', '2.5_512', '2.5_1024', '2.5_2048',
                                    '3.0_256', '3.0_512', '3.0_1024', '3.0_2048', '3.5_256', '3.5_512', '3.5_1024', '3.5_2048',
                                    '4.0_256', '4.0_512', '4.0_1024', '4.0_2048'],
                            'MDD': ['1.0_256', '1.0_512', '1.0_1024', '1.0_2048', '2.0_256', '2.0_512', '2.0_1024', '2.0_2048',
                                    '3.0_256', '3.0_512', '3.0_1024', '3.0_2048', '4.0_256', '4.0_512', '4.0_1024', '4.0_2048',
                                    '5.0_256', '5.0_512', '5.0_1024', '5.0_2048', '6.0_256', '6.0_512', '6.0_1024', '6.0_2048',
                                    '7.0_256', '7.0_512', '7.0_1024', '7.0_2048']
                            }

    config['visda'] = (args.dset == 'visda')
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config['name'] = args.dset + '/' + names[s][0].upper() + names[t][0].upper()
    config["save_dir"] = os.path.join('./logs/'+args.da, args.da)
    if not osp.exists(config["save_dir"]):
        os.system('mkdir -p '+config["save_dir"])
    config["task_avg_log"] = open(osp.join(config["save_dir"], args.da+"_avglog.txt"), "a+")
    config["prep"] = {'params':{"resize_size":256, "crop_size":224, "alexnet":False}}
    config["dataset"] = args.dset
    # 36 100 96
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.bs}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test_tgt":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test_src":{"list_path":args.s_dset_path, "batch_size":args.bs}}
                      
    setting_str = "dset: {}, src: {}, tgt: {}, method: {}, net: {}.\n"\
                .format(args.dset, names[args.s], names[args.t], args.method, args.net)
    print(setting_str)
    ensv(config, args)


