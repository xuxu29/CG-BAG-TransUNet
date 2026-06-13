import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    warmup_epochs = 20
    warmup_iterations = warmup_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    logging.info("Warmup iterations: {}".format(warmup_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            aux_preds = None
            if isinstance(outputs, tuple):
                outputs, aux_preds = outputs
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_main = 0.5 * loss_ce + 0.5 * loss_dice
            loss_aux = outputs.new_tensor(0.0)
            if aux_preds is not None:
                aux_weights = [0.1, 0.1]
                for aux_pred, aux_w in zip(aux_preds, aux_weights):
                    aux_pred_up = F.interpolate(
                        aux_pred,
                        size=label_batch.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    aux_ce = ce_loss(aux_pred_up, label_batch[:].long())
                    aux_dice = dice_loss(aux_pred_up, label_batch, softmax=True)
                    loss_aux = loss_aux + aux_w * (0.5 * aux_ce + 0.5 * aux_dice)
            loss = loss_main + loss_aux
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if iter_num < warmup_iterations:
                lr_ = base_lr * (iter_num + 1) / warmup_iterations
            else:
                progress = (iter_num - warmup_iterations) / (max_iterations - warmup_iterations)
                lr_ = base_lr * (1.0 - progress) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_aux', loss_aux, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_aux: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_aux.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
