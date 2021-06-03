import os
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset

from src.task_utils import label_text_map

from src.processors_eprompt import processors

class CachedDataset(Dataset):
    def get_default_config():
        config = {
            'data_path': './data',
            'seq_size': 2048,
            'n_select': 0,
            'select_start_index': 0,
            'randomize': False,
            'random_seed': 0,
            'balanced': False,
            'mode': 'test',
            'parametrize_answer_sep': False,
            'sep_init': 'newline',
        }
        return config

    def __init__(self, task_name, tokenizer, split, config):
        self.task_name = task_name
        if 'fact-retrieval' in task_name:
            task_name_generic, rel = task_name.split('_R_')
            self.processor = processors[task_name_generic](rel)
        else:
            self.processor = processors[task_name.lower()]()

        print ('CachedDataset config {}'.format(config))
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_mask_token = tokenizer.mask_token
        self.tokenizer_mask_token_id = tokenizer.mask_token_id

        if config["parametrize_answer_sep"]:
            self.answer_sep = len(tokenizer) #one larger than the current vocab size
        else:
            #self.answer_sep = tokenizer.convert_tokens_to_ids(
            #                        tokenizer.tokenize("a\n"))[-1]
            if config['sep_init'] == 'newline':
                self.answer_sep = tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("a\n"))[-1]
            elif config['sep_init'] == 'eos':
                print ("data: Setting sep to eos")
                self.answer_sep = tokenizer.eos_token_id

        
        cached_features_file = os.path.join(config['data_path'],
                                'pero_{}_{}_{}{}'.format(task_name,
                                    tokenizer.__class__.__name__,
                                    split,
                                    tokenizer.name_or_path if tokenizer.name_or_path in [
                                        'bert-large-cased', 'bert-large-uncased', 'bert-base-cased', 'bert-base-uncased'] else ''))
        fetch_fn = {
            'train': self.processor.get_train_examples,
            'dev': self.processor.get_dev_examples,
            'test': self.processor.get_test_examples
        }
        
        self.examples = fetch_fn[split](config['data_path'])
        if os.path.exists(cached_features_file):
            print ("Loading features from {}".format(cached_features_file))
            self.features = torch.load(cached_features_file)
        else:
            self.features = self.convert_examples_to_features(self.examples, config)
            print ("Saving features to {}".format(cached_features_file))
            torch.save(self.features, cached_features_file)

        indices = self.select_examples(self.examples, config)
        if indices is not None:
            self.examples = [self.examples[idx] for idx in indices]
            self.features = [self.features[idx] for idx in indices]

        self.prompt_sequence = None

    def select_examples(self, examples, config):
        if config['n_select'] == 0: return None
        elif config['n_select'] > len(examples):
            #raise ValueError("Number of elements {} less than selected size {}".format(len(examples), config['n_select']))
            print("Number of elements {} less than selected size {}".format(len(examples), config['n_select']))
            return None
    
        n_examples = len(examples)
        indices = np.arange(len(examples))
        if config['randomize']:
            r = np.random.RandomState(seed=config['random_seed'])
            indices = r.choice(indices, size=len(indices), replace=False)

        label_map = label_text_map.get(self.task_name.lower(), None)
        if config['balanced'] and label_map is None:
            config['balanced'] = False
        if not config['balanced']:
            indices = indices[config['select_start_index']: config['select_start_index'] + config['n_select']]
            return indices

        n_labels = len(label_map)
        per_label_count = config['n_select']//n_labels
        print ('Per label count', per_label_count)
        counts = {k:per_label_count for k in label_map}
        curr_idx = config['select_start_index']
        new_indices = []
        while len(counts) > 0:
            l = self.examples[indices[curr_idx]].label
            if l in counts:
                new_indices.append(indices[curr_idx])
                counts[l] = counts[l] - 1
                if counts[l] == 0: del counts[l]
            curr_idx = curr_idx + 1
            if curr_idx % config['n_select'] == 0:
                print ("Selecting balanced examples", curr_idx) 
        return new_indices

    def convert_examples_to_features(self, examples, config):
        #Assume all tasks to contain only one sequence
        tokenizer = self.tokenizer
        mask_token = self.tokenizer_mask_token
        label_map = label_text_map.get(self.task_name.lower(), None)
        features = []
        print_count = 3
        for e in examples:
            text = e.text_a.replace("[ANSWER]", mask_token)
            token_info = tokenizer(text)
            tokenized_text = token_info.input_ids

            label_text = label_map[e.label] if label_map is not None else e.label
            label_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + label_text))
            features.append((tokenized_text, label_ids))

            if print_count > 0:
                print (tokenizer.decode(tokenized_text).encode('utf-8').strip())
                print_count = print_count - 1
        return features

    def get_sequence(self, item, labeled=True, pad=False):
        #item could be index or tuple, currently assuming tuple
        config = self.config
        features = []
        for pos_idx, idx in enumerate(item):
            feat = self.features[idx][0]
            if not labeled and pos_idx == (len(item) - 1):
                feat = feat[1:-1]
            else:
                mask_idx = feat.index(self.tokenizer_mask_token_id)
                feat = feat[1:mask_idx] + self.features[idx][1] + feat[mask_idx+1:-1]
                feat.append(self.answer_sep)
            features.extend(feat)
        if labeled: return features
        features = [self.tokenizer.cls_token_id] + features + [self.tokenizer.sep_token_id]
        if not pad: return features
       
        feat = features
        len_tokens = len(feat)
        pad_token_id = self.tokenizer.pad_token_id
        if len(feat) > config["seq_size"]:
            print ("need to truncate, actual {} need to reduce to {}".format(len(feat), config["seq_size"]))
            feat = feat[:config["seq_size"]]
            len_tokens = len(feat)
        elif len(feat) < config["seq_size"]:
            feat = feat + [pad_token_id] * (config["seq_size"] - len(feat))
        return torch.tensor(feat), len_tokens, self.features[item[-1]][1][0]
        

    def set_prompt_sequence(self, item, drop_match=False, indices=None): #drop_match to be used only when ref data is the same set
        self.prompt_sequence_indices = indices
        self.drop_match = drop_match
        self.prompt_sequence = item

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        config = self.config
        pad_token_id = self.tokenizer.pad_token_id

        label_id = self.features[item][1]
        #if len(label_id) > 1: print ("Label ids more than one token: {}".format(self.examples[item].label))
        label_id = label_id[0]
        example_feat = self.features[item][0]
        if self.prompt_sequence is None:
            feat = example_feat
        else:
            if self.drop_match:
                indices = list(self.prompt_sequence_indices)
                if item in indices:
                    indices.remove(item)
                prompt_sequence = self.get_sequence(indices)
                feat = [example_feat[0]] + prompt_sequence + example_feat[1:]
            else:
                feat = [example_feat[0]] + self.prompt_sequence + example_feat[1:]
        len_tokens = len(feat)

        if len(feat) > config["seq_size"]:
            print ("need to truncate, actual {} need to reduce to {}".format(len(feat), config["seq_size"]))
            #feat = feat[:config["seq_size"]]
            feat = feat[:config["seq_size"]-4]+example_feat[-4:]
            len_tokens = len(feat)
        elif len(feat) < config["seq_size"]:
            feat = feat + [pad_token_id] * (config["seq_size"] - len(feat))
        return torch.tensor(feat), len_tokens, label_id
