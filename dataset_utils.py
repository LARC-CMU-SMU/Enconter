import logging
import os
import pickle as pk

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InsertionTransformerDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512, add_special_tkn=False):
        self.examples = []
        self.cond = []
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        tokenizer_name = tokenizer.__class__.__name__
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}_{}_{}_{}'.format(block_size,
                                                                                         filename.replace(".txt", ""),
                                                                                         tokenizer_name,
                                                                                         len(tokenizer),
                                                                                         self.__class__.__name__))
        cached_cond_file = os.path.join(directory, 'cached_cond_lm_{}_{}_{}_{}_{}'.format(block_size,
                                                                                          filename.replace(".txt", ""),
                                                                                          tokenizer_name,
                                                                                          len(tokenizer),
                                                                                          self.__class__.__name__))
        if os.path.exists(cached_features_file) and os.path.exists(cached_cond_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            logger.info("Loading conditions from cached file %s", cached_cond_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pk.load(handle)
            with open(cached_cond_file, 'rb') as handle:
                self.cond = pk.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            with open(file_path, "rb") as f:
                txt_splt = pk.load(f)
            # Must have padding token
            padding_token = tokenizer.pad_token
            assert padding_token is not None
            pbar = tqdm(total=len(txt_splt))
            for jbpst in txt_splt:
                cond_tknize, tknize = jbpst[0], jbpst[1]
                cond_tknize = cond_tknize + [padding_token] * (block_size - len(cond_tknize))
                if add_special_tkn:
                    tknize = [tokenizer.cls_token] + tknize + [tokenizer.sep_token] + \
                             [padding_token] * (block_size - len(tknize))
                else:
                    tknize = tknize + [padding_token] * (block_size - len(tknize))
                self.cond.append(tokenizer.convert_tokens_to_ids(cond_tknize))
                self.examples.append(tokenizer.convert_tokens_to_ids(tknize))
                pbar.update(1)
            pbar.close()
            with open(cached_features_file, "wb") as handle:
                pk.dump(self.examples, handle)
            with open(cached_cond_file, "wb") as handle:
                pk.dump(self.cond, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.cond[item]), torch.tensor(self.examples[item])


def concat_fn(batch):
    """ Aggregate batch """
    cond, inp = zip(*batch)
    cond = torch.stack(cond, 0)
    inp = torch.stack(inp, 0)
    return cond, inp
