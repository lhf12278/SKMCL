from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net, Cross_modal_Attn, modal_Classifier
from utils import *
from loss import OriTripletLoss, Average_loss, CrossEntropyLabelSmooth
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math

import setproctitle
setproctitle.setproctitle("xukaixiong")

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')

parser.add_argument('--aa', default=0.5, type=float,
                    metavar='aa', help='lamuda3')
parser.add_argument('--bb', default=0.01, type=float,
                    metavar='bb', help='lamuda4')
parser.add_argument('--cc', default=0.4, type=float,
                    metavar='cc', help='lamuda2')
parser.add_argument('--dd', default=1.0, type=float,
                    metavar='dd', help='lamuda1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)
loader_batch = args.batch_size * args.num_pos
dataset = args.dataset
if dataset == 'sysu':
    # TODO: define your data path
    data_path = '/home/l/data/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log_ddag/'
    test_mode = [1, 2] # infrared to visible
elif dataset =='regdb':
    # TODO: define your data path for RegDB dataset
    data_path = 'YOUR DATA PATH'
    log_path = args.log_path + 'regdb_log_ddag/'
    test_mode = [2, 1] # visible to infrared

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = dataset
if args.graph:
    suffix = suffix + '_G'
if args.wpa:
    suffix = suffix + '_P_{}'.format(args.part)

suffix = suffix + '_n_{}_b_{}_lr_{}_aa_{}_bb_{}_cc_{}_dd_{}'.format(args.num_pos, args.batch_size, args.lr, args.aa, args.bb, args.cc, args.dd)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
feature_dim = args.low_dim
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])


if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa, load_batch=loader_batch)
net.to(device)
net_cma = Cross_modal_Attn(in_dim=2048, class_num=n_class, load_batch=loader_batch)
net_cma.to(device)

net_modal_classifier1 = modal_Classifier(embed_dim=2048, modal_class=3)
net_modal_classifier1.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion1 = nn.CrossEntropyLoss()
# criterion1 = CrossEntropyLabelSmooth(num_classes=n_class)
loader_batch = args.batch_size * args.num_pos
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion1.to(device)
criterion2.to(device)

criterion3 = nn.MSELoss()
#criterion3 = nn.L1Loss()
criterion3.to(device)

criterion4 = Average_loss(num_classes=3)
criterion4.to(device)

# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},

        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

net_cma_optimizer = torch.optim.SGD(net_cma.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
modal_classifier_optimizer_1 = torch.optim.SGD(net_modal_classifier1.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer_P, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch, wG):
    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer_P, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0).long()

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda()).long()
        label1 = Variable(label1.cuda()).long()

        modal_v_labels = Variable(torch.ones(loader_batch).long().cuda())
        modal_t_labels = Variable(torch.zeros(loader_batch).long().cuda())
        # modal_3_labels = Variable(2 * torch.ones(2 * loader_batch).long().cuda())
        modal_3_labels = Variable(2 * torch.ones(loader_batch).long().cuda())

        data_time.update(time.time() - end)

        # Forward into the network
        feat, out0, feat_map, out_mix, vt_mix_pool = net(input1, input2)

        loss_id = criterion1(out0, labels)
        loss_tri, batch_acc = criterion2(feat, labels)

        loss_id_mix = args.aa *  criterion1(out_mix, label1)

        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        feat_modal = feat.detach()
        out_modal = net_modal_classifier1(feat_modal)

        feature_map = feat_map.detach()
        att_map, out1, att_feat = net_cma(feature_map)
        loss_att = criterion1(out1, labels)
        loss_att_tri, _ = criterion2(att_feat, labels)

        if epoch < 30:
            loss = loss_id + loss_tri + loss_id_mix
            loss_total = loss

            # optimization
            optimizer_P.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer_P.step()

            net_cma_optimizer.zero_grad()
            (loss_att + loss_att_tri).backward(retain_graph=True)
            net_cma_optimizer.step()

            modal_feat_vt_fushion = vt_mix_pool.detach()
            modal_feat_vt_fushion_cls = net_modal_classifier1(modal_feat_vt_fushion)
            modal_loss1 = criterion4(modal_feat_vt_fushion_cls)

            modal_loss2_1 = criterion1(out_modal[:loader_batch], modal_v_labels)
            modal_loss2_2 = criterion1(out_modal[loader_batch:], modal_t_labels)
            modal_loss2 = modal_loss2_1 + modal_loss2_2
            modal_loss = modal_loss1 + modal_loss2

            modal_classifier_optimizer_1.zero_grad()
            modal_loss.backward()
            modal_classifier_optimizer_1.step()
            # if batch_idx % 10 == 0:
            #     print('modal_loss: ' + str(modal_loss.cpu().detach().numpy()))
        else:
            modal_loss2 = criterion1(out_modal[:loader_batch], modal_v_labels) + criterion1(out_modal[loader_batch:],
                                                                                           modal_t_labels)
            modal_loss = modal_loss2

            modal_classifier_optimizer_1.zero_grad()
            modal_loss.backward()
            modal_classifier_optimizer_1.step()

            out2 = net_modal_classifier1(feat)
            loss2 = args.cc * (criterion1(out2[:loader_batch], modal_3_labels) + criterion1(out2[loader_batch:],
                                                                                           modal_3_labels))
            out3 = net_modal_classifier1(vt_mix_pool)
            loss3 = args.bb * criterion1(out3, modal_3_labels)

            teach_loss = args.dd * criterion3(feat, att_feat)

            loss = loss_id + loss_tri + loss_id_mix
            loss_total = loss + teach_loss + loss2 + loss3

            # optimization
            optimizer_P.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer_P.step()

            net_cma_optimizer.zero_grad()
            (loss_att + loss_att_tri).backward(retain_graph=True)
            net_cma_optimizer.step()
            if batch_idx % 10 == 0:
                print('teach_loss: ' + str(teach_loss.cpu().detach().numpy()))
                print('loss2: ' + str(loss2.cpu().detach().numpy()))
                print('loss3: ' + str(loss3.cpu().detach().numpy()))


        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        # graph_loss.update(loss_G.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print('att_loss: ' + str(loss_att.cpu().detach().numpy()))
            print('loss_att_tri: ' + str(loss_att_tri.cpu().detach().numpy()))
            print('modal_loss: ' + str(modal_loss.cpu().detach().numpy()))
            print('loss_id_mix: ' + str(loss_id_mix.cpu().detach().numpy()))

            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '

                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    # computer wG
    return 1. / (1. + train_loss.avg)

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    # gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1,5,10,20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP, mINP


# training
print('==> Start Training...')
for epoch in range(start_epoch, 101 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler: 
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # infrared index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    wG = train(epoch, wG)

    if epoch >= 0 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP, mINP = test(epoch)
        # log output
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP), file=test_log_file)

        test_log_file.flush()
        
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            best_mAP = mAP
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
        print('Best Epoch [{}]'.format(best_epoch))
        print('Best Epoch [{}]: Rank-1: {:.2%} |  mAP: {:.2%}'.format(best_epoch, best_acc, best_mAP))
