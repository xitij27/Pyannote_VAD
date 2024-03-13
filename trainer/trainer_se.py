import torch

from scipy import signal
from importlib import import_module

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils_vad.utils import downsample

class Trainer(object):
    def __init__(self, train_config, model_config):
        model_name     = train_config.get('model_name', 'PyanNet_se')
        self.opt_param = train_config.get('optimizer_param', {
                                            'optim_type': 'Adam',
                                            'learning_rate': 1e-4,
                                            'max_grad_norm': 10,
                                        })    

        self.gpus = train_config.get('gpus', [-1])
        if self.gpus == [-1]:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.gpus[0]))

        module = import_module('model.{}'.format(model_name), package=None)
        MODEL = getattr(module, model_name)
        model = MODEL(**model_config).to(self.device)

        print(model)

        self.model = model.to(self.device)
        self.learning_rate = self.opt_param['learning_rate']

        if self.opt_param['optim_type'].upper() == 'RADAM':
            self.optimizer = torch.optim.RAdam( self.model.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.9,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=self.opt_param['learning_rate'],
                                               betas=(0.9,0.999),
                                               weight_decay=0.0)

        if 'lr_scheduler' in self.opt_param.keys():
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
                                **self.opt_param['lr_scheduler']
                            )
        else:
            self.scheduler = None


        self.iteration = 0
        self.model.train()

    def step(self, input, iteration=None):
        assert self.model.training
        self.model.zero_grad()
        
        for key in input.keys():
            input[key] = input[key].to(self.device)

        # [batch_size, num_sample, 2]
        if self.gpus == [-1]:
            preds = self.model(input['feat'])
        else:
            preds = torch.nn.parallel.data_parallel(
                self.model,
                (input['feat']),
                self.gpus,
                self.gpus[0],
            )
            
        input["label"] = downsample(input["label"], preds).float()
        loss = torch.nn.BCELoss(reduction='mean')(preds, input["label"])
        
        loss_detail = {"VAD loss": loss.item()}

        loss.backward()
        if self.opt_param['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.opt_param['max_grad_norm'])
        self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        if self.scheduler is not None:
            self.scheduler.step()

        if iteration is not None:
            self.iteration = iteration + 1
        else:
            self.iteration += 1

        return self.iteration, loss_detail, learning_rate


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        print(f'load pretrained model from {checkpoint_path}')
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'], strict=False)
        # self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        return checkpoint_data['iteration']
    