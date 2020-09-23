"""going by o2f format and using huggingface library
15.09.2020 - @yashbonde"""

import os
from types import SimpleNamespace
from argparse import ArgumentParser
from data import StdzDataset, DatasetConfig, load_tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

# --- arguments
args = ArgumentParser(description="GPT based standardisation methods")

# --- paths
args.add_argument("--save_folder", default = "models", type = str, help = "folder to save all models")
args.add_argument("--name", type = str, help = "name of this particular model")
args.add_argument("--tokenizer", default = "data/cleaned/s3k3", type = str, help = "path to spm model file")
args.add_argument("--datatrain", default = "data/cleaned/train.xlsx", type = str, help = "path to Train Excel file to train")
args.add_argument("--datatest", default = "data/cleaned/test.xlsx", type = str, help = "path to Test Excel file to train")
args.add_argument("--seed", default = None, type = int, help = "seed value for training")
sheets = ["D1"] # add to parser later

# --- arch
args.add_argument("--n_embd", default = 144, type = int, help = "Embedding Dim")
args.add_argument("--n_layer", default = 12, type = int, help = "Num Layers")
args.add_argument("--n_head", default = 6, type = int, help = "Num Heads")
args.add_argument("--maxlen", default = 200, type = int, help = "Maximum length of decoder")

# --- data
args.add_argument("--mult", default = 3, type = int, help = "Size of dataset")
args.add_argument("--pf", default = 0.6, type = float, help = "Probability of using fields in training sequence")
args.add_argument("--fmax", default = 0.8, type = float, help = "Max fields probability")
args.add_argument("--fmin", default = 0.1, type = float, help = "Min fields probability")

# --- trainer
args.add_argument("--n_epochs", default = 200, type = int, help = "Number of epochs to train")
args.add_argument("--batch_size", default = 200, type = int, help = "Mini-Batch Size")
args.add_argument("--lr", default = 1e-3, type = float, help = "Learning Rate")
args.add_argument("--sample_every", default = 5, type = int, help = "After t")
args.add_argument("--train_ratio", default = 0.9, type = float, help = "Ratio of train data, rest is testing")
args.add_argument("--beta1", default = 0.9, type = float, help = "Adam.beta1")
args.add_argument("--beta2", default = 0.95, type = float, help = "Adam.beta2")
args.add_argument("--grad_norm_clip", default = 1.0, type = float, help = "Adam.beta2")

args.add_argument("--patience", default = 6, type = int, help = "training stops after patience runs out")

# --- parse and add more
args = args.parse_args()
tb_path = os.path.join(args.save_folder, args.name)
ckpt_path = os.path.join(tb_path, f"{args.name}.pt")
args = SimpleNamespace(**vars(args), ckpt_path = ckpt_path, tb_path = tb_path)

# make folders
os.makedirs(args.save_folder, exist_ok=True)
os.makedirs(args.tb_path, exist_ok=False)

# get/set things
t = load_tokenizer(args.tokenizer)

# Model
modelConfig = GPT2Config(
    vocab_size=t.vocab_size(),
    n_positions=args.maxlen,
    n_ctx=args.maxlen,
    n_embd=args.n_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    bos_token_id=t.bos_id(),
    eos_token_id=t.eos_id()
)
model = GPT2LMHeadModel(modelConfig)
print(f"Number of parameters: {model.num_parameters()}")

# DataSet 
dataTrainConfig = DatasetConfig(
    path = args.datatrain,
    sheets = sheets,
    mult = args.mult,
    pf = args.pf,
    fmax = args.fmax,
    fmin = args.fmin,
    maxlen = args.maxlen
)

dataTestConfig = DatasetConfig(**vars(dataTrainConfig))
dataTestConfig.path = args.datatest
dtrain = StdzDataset(dataTrainConfig, t, mode = "train")
dtest = StdzDataset(dataTestConfig, t, mode = "test")

# Trainer
trainConfig = TrainerConfig(
    max_epochs = args.n_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    betas = (args.beta1, args.beta2),
    grad_norm_clip = args.grad_norm_clip,
    tb_path = args.tb_path,
    ckpt_path = args.ckpt_path,
    num_batch = (len(dtrain) // args.batch_size) + int(len(dtrain) % args.batch_size != 0),
    len_data = len(dtrain)
)

print(modelConfig, dataTrainConfig, dataTestConfig, trainConfig)
trainer = Trainer(model, dtrain, dtest, trainConfig, t)
trainer.train()
