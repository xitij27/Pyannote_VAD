import os
import sys

import time
import torch
import logging
import numpy as np
from pathlib import Path
from importlib import import_module
import json
import torch.nn as nn

from torch.utils.data import DataLoader
from itertools import permutations, combinations

egs_path = os.path.dirname(__file__)
sys.path.append(egs_path)

from dataloader.data_loader_aug import Dataset
from utils_vad.utils import downsample

def collate_fn(batches):
    feat_batches = [item['feat'] for item in batches]
    label_batches = [item['label'] for item in batches]
    
    feat_batches = torch.stack(feat_batches)
    label_batches = torch.stack(label_batches)
    
    egs = {
        'feat': feat_batches,
        'label': label_batches,
    }
    
    return egs

def train(train_config): 
    # Initial
    output_directory     = train_config.get('output_directory', '')
    max_epoch            = train_config.get('max_epoch', 50)
    batch_size           = train_config.get('batch_size', 64)
    chunk_size           = train_config.get('chunk_size', 40)
    chunk_step           = train_config.get('chunk_step', 20)
    rate                 = train_config.get('rate', 16000)
    frame_len            = train_config.get('frame_len', 0.025)
    frame_shift          = train_config.get('frame_shift', 0.010)
    iters_per_log        = train_config.get('iters_per_log', 1)
    seed                 = train_config.get('seed', 1234)
    checkpoint_path      = train_config.get('checkpoint_path', '')
    epochs_per_eval      = train_config.get('epochs_per_eval', 5)
    model_config         = train_config.get('model_config')

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   

    # Initial trainer
    module = import_module('trainer.{}'.format('trainer'), package=None)
    TRAINER = getattr( module, 'Trainer')
    trainer = TRAINER( train_config, model_config)

    # Load checkpoint if the path is given 
    iteration = 1
    epoch = 0
    if checkpoint_path != "":
        iteration = trainer.load_checkpoint( checkpoint_path)
        iteration += 1  # next iteration is iteration + 1

    # Load training data
    trainset = Dataset(
        data_path=os.path.join(egs_path, train_config['train_path']) if train_config['train_path'].startswith("/") else train_config['train_path'],
        noise_path=train_config['noise_path'],
        snr=train_config['snr'],
        rttm_path=train_config['train_rttm_path'],
        chunk_size=chunk_size, 
        chunk_step=chunk_step,
        rate=rate,
    )    
    train_loader = DataLoader(
        trainset, 
        num_workers=train_config['num_workers'], 
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Load evaluation data
    evalset = Dataset(
        data_path=os.path.join(egs_path, train_config['eval_path']) if train_config['eval_path'].startswith("/") else train_config['eval_path'],
        noise_path=train_config['noise_path'],
        snr=train_config['snr'],
        rttm_path=train_config['eval_rttm_path'],
        chunk_size=chunk_size, 
        chunk_step=chunk_step, 
        rate=rate,
    )    
    eval_loader = DataLoader(
        evalset, 
        num_workers=train_config['num_workers'], 
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Get shared output_directory ready
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_directory/'Stat'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("Output directory: {}".format(output_directory))
    logger.info("Training utterances: {}".format(len(trainset)))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("# of seconds per sample: {}".format(chunk_size))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start traininig...")

    loss_log = dict()
    while epoch < max_epoch:
        trainer.model.train()
        for i, batch in enumerate(train_loader):
            
            iteration, loss_detail, lr = trainer.step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)

            # Show log per M iterations
            if iteration % iters_per_log == 0 and len(loss_log.keys()) > 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in loss_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                mseg += '  lr: {:.6f}'.format(lr)
                logger.info(mseg)
                loss_log = dict()
            
        epoch += 1

        if epoch % epochs_per_eval == 0:
            eval_loss = []
            trainer.model.eval()
            for i, batch in enumerate(eval_loader):
                with torch.no_grad():
                    for key in batch.keys():
                        batch[key] = batch[key].to(trainer.device)
                            
                    # batch['feat'] = batch['feat'][:, None, None, ...]

                    # [batch_size, num_sample, 2]
                    preds = torch.nn.parallel.data_parallel(
                        trainer.model,
                        (batch['feat']),
                        trainer.gpus,
                        trainer.gpus[0],
                    )
                    batch["label"] = downsample(batch["label"], preds).float()
                    loss = torch.nn.BCELoss(reduction='mean')(preds, batch["label"])
                    eval_loss.append(loss.item())
            mseg = 'Epoch {}:'.format( epoch)
            mseg += "Eval loss: {}".format(np.mean(eval_loss))
            logger.info(mseg)

        if epoch % 5 == 0:
            checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),epoch)
            trainer.save_checkpoint(checkpoint_path)

        if epoch > max_epoch:
            break
        

    print('Finished')
        

if __name__ == "__main__":

    import argparse
    import yaml
    
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=os.path.join(egs_path, 'config/config.yaml'),
                        help='Yaml file for configuration')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_yaml = yaml.safe_load(stream)

    train(config_yaml)
