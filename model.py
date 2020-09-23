# making R-GNN Network, please refer to tests/ folder where I have done several 
# experiments with the smaller networks and how different parts of it can be
# combined to create the final model
# 26.08.2020 - @yashbonde
# Trying to code Karpathy style

import os
import time
import math
import random
import logging
import platform
import subprocess
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, MultiStepLR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
from torch_scatter.scatter import scatter_max

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for node_edge <- node + edge_attr
        self.edge_lin = nn.Sequential(
            nn.Linear(int(config.edim * 2), config.edim),
            nn.ReLU()
        )
        self.edge_drop = nn.Dropout(config.drop_p)

        # for node_2 <- node + global + node_edge
        self.node_lin = nn.Sequential(
            nn.Linear(int(config.edim * 3), int(config.edim * 4)),
            nn.ReLU(),
            nn.Linear(int(config.edim * 4), config.edim)
        )
        self.node_drop = nn.Dropout(config.drop_p)

        # for global <- node_2 + global
        self.glob_lin = nn.Sequential(
            nn.Linear(int(config.edim * 2), config.edim),
            nn.ReLU()
        )
        self.glob_drop = nn.Dropout(config.drop_p)

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = self.edge_lin(torch.cat((x[row], edge_attr), dim = -1))
        out = self.edge_drop(scatter_max(out, col, dim = 0, dim_size = x.size(0)))
        out = self.node_drop(self.node_lin(torch.cat((out, x, u), dim = -1)).sigmoid())
        x = x + out

        x_u = scatter_max(x, batch, dim=0, dim_size=x.size(0))
        out = self.glob_lin(torch.cat((x_u, u), dim = -1))
        u = u + out

        return x, edge_index, edge_attr, u, batch

class GraphTemporalNetwork(nn.Module):
    """
    This is a very weird network that uses both the graph neural network and
    LSTM with residual connections
    """
    def __init__(self, config) -> None:
        super().__init__()
        
    def forward(self, graphs, hidden_state):
        """
        first perform the embeddings and get three things:
        1. node embeddings (x_i)
        2. edge_embeddings (v_ij)
        3. glob_embeddings (u)

        :param graph: Namespace(
            
            edge_index: []
        )
        """
        

# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################
# TRAINER ##########################################################################

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, tokenizer):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.tokenizer = tokenizer

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"Saving Model at {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, verbose = False):
        model, config, tokenizer = self.model, self.config, self.tokenizer
        optimizer = AdamW(
            model.parameters(),
            lr = self.config.lr,
            betas = self.config.betas
        )
#         lrscheduler = CosineAnnealingLR(optimizer, config.num_batch*config.max_epochs)
#         lrscheduler = CosineAnnealingWarmRestarts(optimizer, config.num_batch * 3, T_mult = 2, eta_min=0)
        lrscheduler = OneCycleLR(
            optimizer,
            max_lr = config.lr,
            total_steps = config.num_batch*config.max_epochs
        )

#         lrscheduler = MultiStepLR(optimizer, [25, 75, 175], gamma = 0.5)
        
        with SummaryWriter(log_dir=config.tb_path, flush_secs=20) as tb:
            
            def run_epoch(split, epoch, _gs = None):
                is_train = split == "train"
                model.train(is_train)
                data = self.train_dataset if is_train else self.test_dataset
                dl = DataLoader(
                    data,
                    shuffle = True,
                    pin_memory = True,
                    batch_size = config.batch_size,
                    num_workers = config.num_workers
                )

                losses = []
                pbar = tqdm(enumerate(dl))
                for it, d in pbar:
                    _l = -1 if not losses else losses[-1]
                    if is_train:
                        pbar.set_description(f"[TRAIN] GS: {_gs}, Epoch: {epoch}, Loss: {round(_l, 5)}")
                    else:
                        pbar.set_description(f"[VAL] Epoch: {epoch}")

                    with torch.set_grad_enabled(is_train):
                        out = model(
                            input_ids = d["input_ids"],
                            labels=d["input_ids"],
                            output_attentions = True,
                            return_dict = True 
                        )
                        losses.append(out.loss.item())

                    if is_train:
                        # add things to tb, loss and attention images
                        tb.add_scalar("loss", out.loss.item(), global_step=_gs, walltime=time.time())
                        tb.add_scalar("lr", lrscheduler.get_lr()[0], global_step=_gs, walltime=time.time())
                        for l, att in enumerate(out.attentions):
                            tb.add_image(
                                f"attention/layer_{l}", att[0][0],
                                global_step=gs, walltime=time.time(),
                                dataformats= "HW"
                            )

                        out.loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                        
                        # when using CosineAnnealing
                        # lrscheduler.step(epoch + it / len(data))
                        lrscheduler.step()
                        _gs += 1

                
                if not is_train:
                    if epoch % config.sample_every == 0:
                        with open(config.tb_path + f'/samples_{gs}.txt', "w") as f:
                            samples_st = time.time()
                            print("Creating Samples ... might take some time", end = "")
                            out = model.generate(
                                input_ids = d["input_ids"][:30, :30],
                                max_length = 150,
                                min_length = 60,
                                do_sample = True,
                                early_stopping = True,
                                num_beams = 20,
                                temperature = 0.9,
                                top_k = 10,
                                top_p = 0.9,
                                bos_token_id = tokenizer.bos_id(),
                                pad_token_id = tokenizer.pad_id(),
                                eos_token_id = tokenizer.eos_id(),
                            )
                            delim = "\n" + "="*60 + "\n"
                            strings = delim.join([tokenizer.decode_ids(x) for x in out.tolist()])
                            print(f"... took {time.time() - samples_st:.3}s")
                            print(f"Saving samples at: {config.tb_path + f'/samples_{gs}.txt'}")
                            f.write(strings)

                    test_loss = float(np.mean(losses))
                    return test_loss

                return _gs

            # now write wrapper for each epoch
            best_loss = float("inf")
            gs = 1
            test_no_improve = 0
            for e in range(config.max_epochs):
                gs = run_epoch("train", e, gs)
                if self.test_dataset is not None:
                    test_loss = run_epoch("test", e)
                    print(f"Test loss: {test_loss}")

                # early stopping based on the test loss of just save always if no test set is provided
                good_model = self.test_dataset is None or test_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    self.save_checkpoint()
                    test_no_improve = 0
                else:
                    test_no_improve += 1
                
                if test_no_improve == config.patience:
                    print(f"Stop Training after [patience = {config.patience}]: {e} epochs")
                    break


class TrainerConfig:
    max_epochs = 10
    batch_size = 128
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    num_workers = 0 # for DataLoader

    len_data = None # required for CosineAnnealing
    sample_every = 5 # after how many epochs to log
    num_batch = None
    
    patience = 5 # training stops after patience runs out

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "max_epochs",
                "batch_size",
                "betas",
                "grad_norm_clip",
                "num_workers",
                "sample_every",
                "num_batch",
                "len_data",
                "patience"
            ] + self.attrs))
        ]) + "\n"

# funcs
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)