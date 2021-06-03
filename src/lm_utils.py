import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

from src.task_utils import label_text_map

class LMPredictor(torch.nn.Module):
    def get_default_config():
        config = {
            "task_name": "SST-2",
            "model_name": "roberta-large",
            "cache_dir": "./cache",
            "train_sep": False,
            "sep_init": "newline",
        }
        return config

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.model_config = AutoConfig.from_pretrained(config["model_name"], cache_dir=config["cache_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["cache_dir"])
        self.lm = AutoModelForMaskedLM.from_pretrained(config["model_name"], from_tf = False,
                            config=self.model_config, cache_dir=config["cache_dir"])
        if config["model_name"] in ["bert-large-cased"]:
            self.tokenizer.eos_token = self.tokenizer.sep_token

        for p in self.lm.parameters():
            p.requires_grad = False

        if config["train_sep"]:
            wte = self.lm.get_input_embeddings()
            if config['sep_init'] == 'newline':
                embed = wte(torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("a\n"))[-1]))
            elif config['sep_init'] == 'eos':
                print ("Setting sep to eos")
                embed = wte(torch.tensor(self.tokenizer.eos_token_id))
            self.priming_embeddings = torch.nn.Parameter(embed)
   
        self.label_map = label_text_map.get(config["task_name"].lower(), None)

        if self.label_map is not None:
            self.label_tokens = {}
            for l in self.label_map:
                self.label_tokens[l] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" " + self.label_map[l]))[0]
            self.label_tokens_inv = {v:k for k,v in self.label_tokens.items()}
            self.label_indices_list = list(self.label_tokens.values())
            self.label_text_list = [self.label_tokens_inv[item] for item in self.label_indices_list]
            self.label_softmax_map = {k:idx for idx,k in enumerate(self.label_indices_list)}

        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v:k for k,v in self.vocab.items()}


    def set_priming_embeddings(self, vec):
        self.priming_embeddings = torch.nn.Parameter(torch.tensor(vec).to(self.lm.device))
    
    def get_priming_embeddings(self):
        return self.priming_embeddings.cpu().data.numpy()

    def predict(self, batch, get_loss=False):
        config = self.config
        outputs = self(batch, get_loss=get_loss)
        indices = torch.argmax(outputs["logits"], dim=1).tolist()
        if self.label_map is not None:
            pred = [self.label_text_list[idx] for idx in indices]
        else:
            if 'roberta' in config['model_name']:
                pred = [self.tokenizer.decode(item, clean_up_tokenization_spaces=True).strip() for item in indices]
            else:
                pred = [self.inv_vocab[item] for item in indices]
        results  = {'pred':  pred}
        results.update(outputs)
        return results 

    def forward(self, batch, get_loss=False, train=False):
        config = self.config
        if train: self.train()
        else: self.eval()

        inputs, input_lengths, labels = batch
        max_length = torch.max(input_lengths).item()
        inputs = inputs[:, :max_length]
        inputs = inputs.to(self.lm.device)
        attention_mask = torch.zeros_like(inputs)
        for idx in range(len(input_lengths)):
            attention_mask[idx, :input_lengths[idx]] = 1

        wte = self.lm.get_input_embeddings()
        if config["train_sep"]:
            weight = torch.cat([wte.weight, self.priming_embeddings.unsqueeze(0)], dim=0)
            inputs_embeds = F.embedding(inputs, weight, wte.padding_idx, wte.max_norm,
                                wte.norm_type, wte.scale_grad_by_freq, wte.sparse)
        else:
            inputs_embeds = wte(inputs)

        input_args = {'inputs_embeds': inputs_embeds, 'return_dict': True, 'attention_mask': attention_mask, 'output_attentions': True, 'output_hidden_states': True}
        outputs = self.lm(**input_args)

        logits = outputs['logits'][inputs==self.tokenizer.mask_token_id]
        if self.label_map is not None:
            logits = logits[:, self.label_indices_list]
        results = {'logits': logits}
        results['mask_embeddings'] = outputs['hidden_states'][-1][inputs==self.tokenizer.mask_token_id]

        if not get_loss:
            return results

        if self.label_map is not None:
            labels = [self.label_softmax_map[l.item()] for l in labels]
        labels = torch.LongTensor(labels).to(self.lm.device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        results['label_ids'] = labels
        results['loss'] = loss

        return results

    def save(self, output_dir, args=None):
        if args is not None:
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        state_dict = self.state_dict()
        state_dict = {k:v.cpu() for k,v in state_dict.items() if 'priming' in k}
        fname = os.path.join(output_dir, "priming_params.bin")
        torch.save(state_dict, fname)

    def load(self, output_dir):
        model_state_dict = self.state_dict()
        fname = os.path.join(output_dir, "priming_params.bin")
        loaded_state_dict = torch.load(fname)
        model_state_dict.update(loaded_state_dict)
        self.load_state_dict(model_state_dict)

    def get_params(self):
        params = [self.priming_embeddings]
        return params
