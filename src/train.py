import logging
import torch
import numpy as np
import json
import os
from utils import common
import model
from DataLoad import load_data, load_test
import argparse
import time


logging.basicConfig(level=logging.DEBUG)
common.set_random_seed(12321)


@torch.no_grad()
def validation(m, ds, lam):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(ds):
        images,  asc, labels = data
        logits_fuse_img, logits_fuse_asc, logits_img, logits_asc, confusion_mat, _ = m.inference(
            images, asc)

        predictions = _softmax(
            lam*logits_fuse_img + (1-lam)*logits_fuse_asc)
        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


def run(epochs, data_dir, asc_dir, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        sum_weight, lam1, lam2, lam3, experiments_path=None):
    train_dir_img = os.path.join(data_dir, dataset, 'SAR_img/train')
    train_dir_asc = os.path.join(asc_dir, dataset, 'ASC/train')
    test_dir_img = os.path.join(data_dir, dataset, 'SAR_img/test')
    test_dir_asc = os.path.join(asc_dir, dataset, 'ASC/test')
    train_set, _ = load_data(file_dir=train_dir_img, asc_dir=train_dir_asc, id=0,
                             picture_size=88, setting=dataset)
    valid_set, _ = load_test(file_dir=test_dir_img, asc_dir=test_dir_asc, id=0,
                             picture_size=88, setting=dataset)
    train_set = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_set = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, drop_last=False)
    m = model.Model(
        classes=classes, dropout_rate=dropout_rate, channels=channels,
        lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        weight_decay=weight_decay, lam1=lam1, lam2=lam2, lam3=lam3
    )

    history_path = os.path.join('results', 'history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        'lam1': [],
        'lam2': [],
        'lam3': [],
        'sum_weight': [],
        'accuracy': [],
    }
    history['lam1'].append(lam1)
    history['lam2'].append(lam2)
    history['lam3'].append(lam3)
    history['sum_weight'].append(sum_weight)
    accuracy_list = []
    for epoch in range(epochs):
        _loss = []

        m.net.train()
        for i, data in enumerate(train_set):
            images, asc, labels = data
            _loss.append(m.optimize(images, asc, labels))

        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        accuracy = validation(m, valid_set, sum_weight)
        accuracy_list.append(accuracy)

        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss_cls={np.mean(_loss[0]):.4f},loss_aln={np.mean(_loss[1]):.4f}, loss_co={np.mean(_loss[2]):.4f}| lr={lr} | accuracy={accuracy:.2f} | MAX accuracy={max(accuracy_list):.2f} achieved on epoch {accuracy_list.index(max(accuracy_list))+1}'
        )

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    with open(os.path.join(history_path, f'history-{dataset}-{localtime}.json'),
              mode='w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=True, indent=2)


def main():
    logging.info('Start')

    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str,
                        default='experiments/config/EOC-2.json')

    args = parser.parse_args()
    config = common.load_config(args.configfile)

    dataset = config['dataset']
    print(dataset)
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']
    data_dir = config['data_dir']
    asc_dir = config['asc_dir']
    sum_weight = config['sum_weight']

    lam1 = config['lam1']
    lam2 = config['lam2']
    lam3 = config['lam3']
    run(epochs, data_dir, asc_dir, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        sum_weight, lam1, lam2, lam3, experiments_path=True)


if __name__ == '__main__':
    main()
