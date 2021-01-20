import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from dataset_utils import InsertionTransformerDataset, concat_fn
from utils import get_linear_schedule_with_warmup, get_lr

device = torch.device("cuda")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train a transformer")
# Basic config
parser.add_argument("--epoch", type=int, default=10, help="epoch")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--save_dir", type=str, default="checkpoint", help="save directory")
parser.add_argument("--save_epoch", type=int, default=5, help="save per how many epoch")
# Optimizer
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--lr_override", action="store_true", help="ignore optimizer checkpoint and override learning rate")
parser.add_argument("--weight_decay", type=float, default=1, help="lr weight decay factor")
parser.add_argument("--decay_step", type=int, default=1, help="lr weight decay step size")
parser.add_argument("--warmup", action="store_true", help="Learning rate warmup")
parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup step")
# Dataset
parser.add_argument("--workers", type=int, default=8, help="number of workers for dataset loader")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--dataset_version", type=str, help="dataset version")
# model
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Choose between bert_initialized or original")
parser.add_argument("--tokenizer", type=str, default='bert-base-cased', help="Using customized tokenizer")
# Debug
parser.add_argument("--no_shuffle", action="store_false", help="No shuffle")
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--debug_dataset_size", type=int, default=1)

args = parser.parse_args()

if not args.debug:
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
logger.info("Args...", vars(args))

# Tokenizer
tokenizer_path = os.path.join(args.save_dir, args.tokenizer)
if os.path.exists(tokenizer_path):
    logger.info("Loading saved tokenizer in {}...".format(tokenizer_path))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
elif os.path.exists(args.tokenizer):
    logger.info("Loading saved tokenizer in {}...".format(args.tokenizer))
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
else:
    logger.info("Using {} tokenizer...".format(args.tokenizer))
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    if args.dataset_version == "CoNLL":
        tokenizer.add_special_tokens({"additional_special_tokens": ["[NOI]", "\n"]})
    else:
        raise ValueError("dataset/tokenizer config error!")
    os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)

# Building model
padding_token = tokenizer.pad_token_id
logger.info("Building model...")
model = BertForMaskedLM.from_pretrained("bert-base-cased")
model.resize_token_embeddings(len(tokenizer))
model = torch.nn.DataParallel(model)
model = model.to(device)

# Read model counter which records the training epoch of the current model
counter = 0
counter_path = os.path.join(os.getcwd(), args.save_dir, "counter.txt")
if not args.debug:
    if os.path.exists(counter_path):
        with open(counter_path, "r") as counter_file:
            counter = int(counter_file.read())
    else:
        with open(counter_path, "w") as counter_file:
            counter_file.write(str(counter))

# Loss history
loss_history_path = os.path.join(os.getcwd(), args.save_dir, "loss_history.npy")
if os.path.exists(loss_history_path):
    with open(loss_history_path, "r") as counter_file:
        loss_history = np.load(loss_history_path)
else:
    loss_history = np.zeros(shape=0)

# Load check points and set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
model_path = os.path.join(os.getcwd(), args.save_dir, "model_" + str(counter - 1) + ".ckpt")
optim_path = os.path.join(os.getcwd(), args.save_dir, "optim_" + str(counter - 1) + ".ckpt")

if counter > 0 and not args.debug:
    if os.path.exists(model_path):
        logger.info("Loading weight from %s", model_path)
        model.module.load_state_dict(torch.load(model_path))
    else:
        logger.info("Model check point not exist!")
    if args.lr_override:
        logger.info("Learning rate OVERRIDE!")
    elif os.path.exists(optim_path):
        logger.info("Loading optim from %s", optim_path)
        optimizer.load_state_dict(torch.load(optim_path))
    else:
        logger.info("Optimizer check point not exist!")
optimizer.param_groups[0]['initial_lr'] = optimizer.param_groups[0]['lr']

training_dataset = InsertionTransformerDataset(tokenizer, os.path.join(os.getcwd(), "dataset", args.dataset))
if args.debug or args.workers == 1:
    loader = data.DataLoader(training_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=concat_fn)
else:
    loader = data.DataLoader(training_dataset,
                             batch_size=args.batch_size,
                             shuffle=args.no_shuffle,
                             collate_fn=concat_fn,
                             num_workers=args.workers)

# Setup scheduler
if len(training_dataset) % args.batch_size == 0:
    total_step = len(training_dataset) // args.batch_size
else:
    total_step = len(training_dataset) // args.batch_size + 1
step = counter * len(loader)
if args.warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_step * args.epoch,
                                                step if step != 0 else -1)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()
logger.info("Start training...")
epoch_loss = np.zeros(0)
for e in range(counter, args.epoch):
    pbar = tqdm(total=total_step)
    avg_loss = np.zeros(shape=(1))
    for batch_num, batch_data in enumerate(loader):
        model.train()
        pbar.update(1)
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        attn_mask = torch.tensor(inputs != tokenizer.pad_token_id, dtype=torch.float32, device=device)
        output = model(inputs, attn_mask, labels=labels)
        loss = output[0]
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if args.warmup:
            scheduler.step()
        logger.info(f"Epoch: {e} lr: {get_lr(optimizer)} Avg NLLLoss: {avg_loss / (batch_num + 1)}")
        if args.debug and batch_num and batch_num % args.debug_dataset_size == 0:
            break
    if not args.warmup:
        scheduler.step()
    pbar.close()
    loss_history = np.concatenate((loss_history, avg_loss / len(loader)))
    np.save(os.path.join(os.getcwd(), args.save_dir, "loss_history"), loss_history)
    plt.plot(loss_history)
    plt.title("loss history")
    plt.savefig(os.path.join(args.save_dir, "loss_history.png"))
    if not args.debug and (e % args.save_epoch == 0 or e == args.epoch - 1):
        torch.save(model.module.state_dict(), os.path.join(os.getcwd(), args.save_dir, "model_" + str(e) + ".ckpt"))
        torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), args.save_dir, "optim_" + str(e) + ".ckpt"))
        with open(counter_path, "w") as counter_file:
            counter_file.write(str(e + 1))
