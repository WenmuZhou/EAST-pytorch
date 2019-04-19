import os
import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import shutil
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from model.model import East
from model.loss import EastLoss
from dataset.data_utils import custom_dset, collate_fn
import config
from utils import *
from eval import eval


def train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    model.train()
    train_loss = 0.
    start = time.time()
    lr = scheduler.get_lr()[0]

    for i, (img, score_map, geo_map, training_mask) in enumerate(train_loader):
        cur_batch = img.size()[0]
        img, score_map, geo_map, training_mask = img.to(device), score_map.to(device), geo_map.to(
                device), training_mask.to(device)

        f_score, f_geometry = model(img)
        loss = criterion(score_map, f_score, geo_map, f_geometry, training_mask)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss += loss
        cur_step = epoch * all_step + i
        writer.add_scalar(tag='Train/loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f} time:{:.4f}, lr:{}'.format(
                    epoch, config.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, batch_time, lr))
            start = time.time()

    return train_loss / all_step, lr



def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")
    writer = SummaryWriter(config.output_dir)
    # Model
    model = East()
    if not config.pretrained and not config.restart_training:
        init_weights(model, init_type=config.init_type)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    train_data = custom_dset(config.trainroot)
    train_loader = DataLoader(train_data, batch_size=config.train_batch_size_per_gpu * num_gpus,
                              shuffle=True, collate_fn=collate_fn, num_workers=config.workers)
    criterion = EastLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                    last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    best_model = {'recall': 0, 'precision': 0, 'f1': 0, 'model': ''}

    try:
        for epoch in range(start_epoch, config.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))
            if epoch % 4 == 0 or train_loss < 0.005:
                recall, precision, f1 = eval(model, os.path.join(config.output_dir, 'output'), config.testroot, device)
                logger.info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, f1))

                net_save_path = '{}/PSENet_{}_loss{:.6f}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, epoch,
                                                                                            0.1,
                                                                                            recall,
                                                                                            precision,
                                                                                            f1)
                save_checkpoint(net_save_path, model, optimizer, epoch, logger)
                if f1 > best_model['f1']:
                    best_model['recall'] = recall
                    best_model['precision'] = precision
                    best_model['f1'] = f1
                    best_model['model'] = net_save_path
                writer.add_scalar(tag='Test/recall', scalar_value=recall, global_step=epoch)
                writer.add_scalar(tag='Test/precision', scalar_value=precision, global_step=epoch)
                writer.add_scalar(tag='Test/f1', scalar_value=f1, global_step=epoch)
        writer.close()
    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, logger)
    finally:
        if best_model['model']:
            shutil.copy(best_model['model'],
                        '{}/best_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(config.output_dir, best_model['recall'],
                                                                      best_model['precision'], best_model['f1']))
            logger.info(best_model)

    # for epoch in range(start_epoch, config.max_epochs):
    #
    #     train(train_loader, model, criterion, scheduler, optimizer, epoch)
    #
    #     if epoch % config.eval_iteration == 0:
    #
    #         # create res_file and img_with_box
    #         output_txt_dir_path = predict(model, criterion, epoch)
    #
    #         # Zip file
    #         submit_path = MyZip(output_txt_dir_path, epoch)
    #
    #         # submit and compute Hmean
    #         hmean_ = compute_hmean(submit_path)
    #
    #         if hmean_ > hmean:
    #             is_best = True
    #
    #         state = {
    #                 'epoch'      : epoch,
    #                 'state_dict' : model.state_dict(),
    #                 'optimizer'  : optimizer.state_dict(),
    #                 'is_best'    : is_best,
    #                 }
    #         save_checkpoint(state, epoch)


if __name__ == "__main__":
    main()
