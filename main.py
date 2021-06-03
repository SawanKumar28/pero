import sys
import os
import numpy as np
import argparse
from tqdm import tqdm
import random
import torch
import pickle
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import SequentialSampler, RandomSampler

from copy import deepcopy
from collections import defaultdict, Counter

from src.data_utils import CachedDataset
from src.lm_utils import LMPredictor
from src.misc_utils import PermutationHandler
from src.opt_utils import optimizers

from transformers import AdamW

import pdb

cached_data = {}

def get_parser():
    #args list
    #name, type, default, help
    args_list = [
        ("cache_dir", str, "./cache", "Directory to cache models"),
        ("output_dir", str, None, "Directory to save model to"),

        ("task_name", str, "sst2", "Task name"),
        ("model_name", str, "roberta-large", "Pretrained LM"),
        ("train_global_sep", bool, None, "Whether to train sep"),
        ("sep_init", str, "newline", ""),

        ("metric_name", str, "acc", "metric for evaluation"),

        ("data_dir", str, "./data", "Data directory"),
        ("n_select", int, 0, "Number of training examples to use"),
        ("select_start_index", int, 0, "Start index for selecting training examples"),
        ("randomize", bool, False, "Whether to randomize training data"),
        ("balanced", bool, None, "Whether to select balanced training set"), 
        ("balanced_dev", bool, None, "Whether to select balanced dev set"), 

        ("select_using_dev", bool, False, "Whether to use dev set for selection"),
        ("restrict_dev_set", bool, False, "Whether to restrict selection dev set to the same size as train set"),
        ("fitness_using_metric", bool, False, ""),

        ("prompt_size", int, 10, "Number of examples in the prompt"),
        ("drop_match", bool, None, "Whether to drop matching examples during training"),

        ("seq_size", int, 512, "Maximum sequence length"),
    
        ("optim_name", str, "GA", "Optimizer to use"),

        ("search_train_size", int, 10, "Train size during search, full if -1"),
        ("sep_train_epochs", int, 5, "Number of epochs for training separator"),
        
        ("do_train", bool, None, ""),
        ("do_eval", bool, None, ""),
        ("do_test", bool, None, ""),
        ("eval_freq_steps", int, 1, "Frequency of evaluation"),
        ("train_batch_size", int, 5, "Batch size for training"),
        ("eval_batch_size", int, 10, "Batch size of eval"),
        ("num_train_epochs", int, 1000, "Number of epochs to train"),

        ("reverse_objective", bool, None, "Whether to reverse the objective fn"),
        ("seed", int, 0, "Seed for random number generation")
    ]
    parser = argparse.ArgumentParser()
    for item in args_list:
        name, dtype, default, helpstr = item
        if dtype is bool:
            parser.add_argument("--{}".format(name), action='store_true', help=helpstr)
        else:
            parser.add_argument("--{}".format(name), type=dtype, default=default, help=helpstr)
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metric(args, labels, preds):
    results = {}
    labels = [item.lower() for item in labels]
    preds = [item.lower() for item in preds]
    labels = np.array(labels)
    preds = np.array(preds)
    n_total = len(labels)
    if args.metric_name == "acc":
        n_correct = np.sum(labels==preds)
        results['acc'] = 100.0 * n_correct / n_total
    elif args.metric_name == "mcc":
        results['mcc'] = matthews_corrcoef(labels, preds)

    n_none = np.sum(preds == 'none')
    results['n'] = n_total
    results['none'] = 100.0 * n_none / n_total
    return results
    

def evaluate(args, model, dataset, verbose=False, max_len=-1):
    if max_len == -1 or max_len >= len(dataset):
        eval_sampler = SequentialSampler(dataset)
        eval_indices = None
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    else:
        eval_indices = np.random.choice(len(dataset), size=max_len, replace=False)
        eval_dataloader = DataLoader(dataset, sampler=eval_indices, batch_size=args.eval_batch_size)
    if verbose: eval_dataloader = tqdm(eval_dataloader)
    preds = []
    avg_loss = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = model.predict(batch, get_loss=True)
            preds.extend(output['pred'])
            avg_loss = avg_loss + output['loss'].item()
    avg_loss = avg_loss / len(eval_dataloader)
    labels = [e.label for e in dataset.examples]
    if eval_indices is not None:
        labels = [labels[idx] for idx in eval_indices]

    if 0:
        print (labels[:10], preds[:10])

    eval_out = compute_metric(args, labels, preds)
    results = {
        'loss': avg_loss,
        #'acc': eval_out['acc']
    }
    results[args.metric_name] = eval_out[args.metric_name]
    return results

def validate(args, model_to_test, val_dataset):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size)
    avg_loss = 0
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            output = model_to_test.predict(batch, get_loss=True)
            avg_loss = avg_loss + output['loss'].item()
    avg_loss = avg_loss / len(val_dataloader)
    return avg_loss

def train(args, model, train_dataset, val_dataset, verbose=False, sep=None):
    if sep is not None: model.set_priming_embeddings(sep)

    best_sep = model.get_priming_embeddings()
    best_val_metric = validate(args, model, val_dataset)

    optimizer = AdamW([model.priming_embeddings], lr=1e-4) #lr 1e-6
    optimizer.zero_grad()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    num_epochs = args.sep_train_epochs
    max_patience = 5
    patience = max_patience
    for e in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            output = model(batch, train=True, get_loss=True)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        val_loss = validate(args, model, val_dataset)
        if val_loss < best_val_metric:
            #print ("Older val {:.2f} Newer val {:.2f}".format(best_val_metric, val_loss))
            best_val_metric = val_loss
            best_sep = model.get_priming_embeddings()
            patience = max_patience
        else:
            patience = patience - 1
            if patience == 0:
                print ("Patience over")
                break
    return best_sep

class SearchHandler():  
    def __init__(self, args, model, data):
        self.args = args
        self.model = model
        self.data = data
    
    def create_dataset(self, items=None, split="train"):
        if split == "train":
            features = [self.data[split].get_sequence(item, labeled=False, pad=True) for item in items]
        else:
            #PH = PermutationHandler(self.args.n_select, self.args.prompt_size, random_seed=np.random.randint(10000))
            PH = PermutationHandler(len(self.data[split]), self.args.prompt_size, random_seed=np.random.randint(10000))
            n_perms = 100
            items = [PH.generate_random_permutation() for _ in range(n_perms)]
            features = [self.data[split].get_sequence(item, labeled=False, pad=True) for item in items]
        inputs = torch.cat([feat[0].unsqueeze(0) for feat in features])
        lengths = torch.tensor([feat[1] for feat in features])
        labels = torch.tensor([feat[2] for feat in features])
        dataset = TensorDataset(inputs, lengths, labels)
            
        return dataset

    def backprop_for_sep(self, population):
        if not self.args.select_using_dev:
            len_train = int(len(population)/2)
            train_p = population[:len_train]
            val_p = population[len_train:]
            train_dataset = self.create_dataset([item.permutation for item in train_p]) 
            val_dataset = self.create_dataset([item.permutation for item in val_p])
        else:
            train_dataset = self.create_dataset([item.permutation for item in population])
            val_dataset = self.create_dataset(split='dev_selection')

        current_sep = self.model.get_priming_embeddings()
        new_sep = train(self.args, self.model, train_dataset, val_dataset, sep=current_sep)
        self.model.set_priming_embeddings(new_sep)
        return new_sep

    def fitness(self, item, sep=None, split='train', metric='loss'): #metric: loss/acc
        if self.args.fitness_using_metric: #override
            metric = self.args.metric_name
        prompt_seq = self.data['train'].get_sequence(item)
        if split == "train":
            self.data[split].set_prompt_sequence(prompt_seq, drop_match=self.args.drop_match, indices=item)
        else:
            self.data[split].set_prompt_sequence(prompt_seq)
        if sep is not None:
            self.model.set_priming_embeddings(sep)
        if split == "train":
            results = evaluate(self.args, self.model, self.data[split], max_len=self.args.search_train_size)
        else:
            results = evaluate(self.args, self.model, self.data[split])
        output = results[metric]
        if metric != 'loss':
            output = 100 - output
        if self.args.reverse_objective: output = -output
        return output

def main():
    parser = get_parser()
    args = parser.parse_args()
    print (args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif len(os.listdir(args.output_dir)) > 0 and args.do_train:
        print ("Output dir already exists and is not empty {}".format(args.output_dir))
        exit()
    set_seed(args.seed)

    config = LMPredictor.get_default_config()
    config['task_name'] = args.task_name
    config['cache_dir'] = args.cache_dir
    config['model_name'] = args.model_name
    config['train_sep'] = args.train_global_sep
    config['sep_init'] = args.sep_init
    model = LMPredictor(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dconfig = CachedDataset.get_default_config()
    dconfig['data_path'] = args.data_dir
    dconfig['seq_size'] = args.seq_size
    dconfig['n_select'], dconfig['select_start_index'] = args.n_select, args.select_start_index
    dconfig['randomize'], dconfig['random_seed'] = args.randomize, args.seed
    dconfig['balanced'] = args.balanced
    dconfig['parametrize_answer_sep'] = args.train_global_sep
    dconfig['sep_init'] = args.sep_init
    cached_data['train'] = CachedDataset(args.task_name, model.tokenizer, "train", dconfig)
    if dconfig['n_select'] == 0: args.n_select = len(cached_data['train'])


    dconfig = CachedDataset.get_default_config()
    dconfig['data_path'] = args.data_dir
    dconfig['seq_size'] = args.seq_size
    dconfig['parametrize_answer_sep'] = args.train_global_sep
    dconfig['sep_init'] = args.sep_init
    if args.restrict_dev_set:
        dconfig['n_select'] =  args.n_select
        dconfig['balanced'] = args.balanced_dev
    cached_data['dev_selection'] =  CachedDataset(args.task_name, model.tokenizer, "dev", dconfig)

    if args.train_global_sep:
        dconfig['n_select'] =  10
        dconfig['balanced'] = True #args.balanced_dev
        cached_data['dev_selection_tgs'] =  CachedDataset(args.task_name, model.tokenizer, "dev", dconfig)

    if args.do_eval:
        dconfig = CachedDataset.get_default_config()
        dconfig['data_path'] = args.data_dir
        dconfig['seq_size'] = args.seq_size
        dconfig['parametrize_answer_sep'] = args.train_global_sep
        dconfig['sep_init'] = args.sep_init
        cached_data['dev'] = CachedDataset(args.task_name, model.tokenizer, "dev", dconfig)
    if args.do_test:
        dconfig = CachedDataset.get_default_config()
        dconfig['data_path'] = args.data_dir
        dconfig['seq_size'] = args.seq_size
        dconfig['parametrize_answer_sep'] = args.train_global_sep
        dconfig['sep_init'] = args.sep_init
        cached_data['test'] = CachedDataset(args.task_name, model.tokenizer, "test", dconfig)

    S = SearchHandler(args, model, cached_data)

    sfname = os.path.join(args.output_dir, "search_state.p")
    if args.do_train:
        gconfig = optimizers[args.optim_name].get_default_config()
        gconfig["n_examples"] = args.n_select
        gconfig["permutation_size"] = args.prompt_size

        callbacks = {
            'fitness_fn': S.fitness
        }
        opt = optimizers[args.optim_name](gconfig, callbacks)
        display_idx_list = [0,1,10, -1]

        best_sel_metric = 1000000000
        prev_entry = None
        best_entry = None
        best_sep = None
        best_epoch = -1
        sep = None
        for e in tqdm(range(args.num_train_epochs), desc="Running training"):
            opt.step()
            if args.train_global_sep:
                sep = S.backprop_for_sep(opt.population)

            if args.select_using_dev:
                if not args.train_global_sep:
                    top_entry = opt.get_best_entry()
                    dev_fitness = S.fitness(top_entry.permutation, split="dev_selection")
                else:
                    updated_fitness = [S.fitness(item.permutation, split="dev_selection_tgs") for item in opt.population]
                    updated_fitness = np.array(updated_fitness)
                    best_index = np.argmin(updated_fitness)
                    top_entry = opt.population[best_index]
                    dev_fitness = S.fitness(top_entry.permutation, split="dev_selection")

                if top_entry is not best_entry:
                    if dev_fitness < best_sel_metric:
                        best_sel_metric = dev_fitness
                        best_epoch = e
                        best_entry = top_entry
                        best_sep = sep
                        dev_fitness_unrestricted = S.fitness(top_entry.permutation, split="dev", metric="acc")
                        print ("Updating the best entry, dev {}".format(dev_fitness_unrestricted))
                        with open(sfname, 'wb') as f: pickle.dump((best_entry, best_sep), f)
                    else:
                        print ("*"*30  + "Best dev [{}] {}".format(best_epoch, dev_fitness_unrestricted))
                else:
                    print ("No change, best metric [{}] {}".format(best_epoch, dev_fitness_unrestricted))
            else:
                with open(sfname, 'wb') as f: pickle.dump((opt.get_best_entry(), sep), f)
                
    #with open(sfname, 'rb') as f: opt = pickle.load(f)
    with open(sfname, 'rb') as f: best_entry, sep = pickle.load(f)
    #best_entry = opt.best_entry()
    if args.do_eval:
        fitness = S.fitness(best_entry.permutation, sep=sep, split="dev", metric="acc")
        print ("Dev fitness {:.2f}".format(fitness))

    if args.do_test:
        fitness = S.fitness(best_entry.permutation, sep=sep, split="test", metric="acc")
        print ("Test fitness {:.2f}".format(fitness))

if __name__ == "__main__":
    main()
