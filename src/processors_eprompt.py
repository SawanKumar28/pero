import os
import jsonlines

from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.glue import *

class SST2LabeledProcessor(Sst2Processor):
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test-labeled.tsv")), "dev")

    def _create_examples(self, lines, set_type):
        examples = super()._create_examples(lines, set_type)
        for e in examples:
            e.text_a = "{} Answer: [ANSWER]".format(e.text_a)
        return examples


class SICKE2balancedProcessor(DataProcessor):
    filenames = {
        "train": "SICK_TRAIN_US.tsv",
        "dev": "SICK_DEV_US.tsv",
        "test": "SICK_TEST_US.tsv"
    }
    s1_idx = 0
    s2_idx = 1
    label_idx = 2

    def get_labels(self):
        return ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, self.filenames["train"])))
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, self.filenames["dev"])))
    
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, self.filenames["test"])))

    def _create_examples(self, lines):
        examples = []
        for (idx, line) in enumerate(lines):
            if idx == 0: continue
            examples.append(InputExample(
                guid=idx,
                text_a="\"{}\" implies \"{}\" Answer: [ANSWER]".format(line[self.s1_idx],line[self.s2_idx]),
                text_b=None,
                label=line[self.label_idx]
                ))
        return examples


class FactRetrievalProcessor(DataProcessor):
    all_relations = ["P19","P20","P279","P37","P413","P166","P449","P69","P47","P138","P364","P54","P463","P101","P1923","P106","P527","P102","P530","P176","P27","P407","P30","P178","P1376","P131","P1412","P108","P136","P17","P39","P264","P276","P937","P140","P1303","P127","P103","P190","P1001","P31","P495","P159","P36","P740","P361"]

    def __init__(self, relation=None):
        self.relation = relation

    def get_relation_info(self, data_dir):
        fname = os.path.join(data_dir, "relations.jsonl")
        info = self._read_json(fname)
        info = {line['relation']:line for line in info}
        return info

    def get_labels(self):
        return []

    def _read_json(self, fname):
        lines = []
        with jsonlines.open(fname) as reader:
            for line in reader:
                lines.append(line)
        return lines

    filenames = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "test": "test.jsonl"
    }

    def get_examples(self, data_dir, split):
        relation_info = self.get_relation_info(data_dir)
        if self.relation == 'all':
            lines = []
            subdirs = os.listdir(data_dir)
            for r in subdirs:
                if not os.path.isdir(os.path.join(data_dir, r)): continue
                #print ("Adding files from {}".format(r))
                r_lines = self._create_examples(self._read_json(os.path.join(data_dir, r, self.filenames[split])), relation_info)
                lines.extend(r_lines)
            return lines
        else:
            return self._create_examples(self._read_json(os.path.join(data_dir, self.relation, self.filenames[split])), relation_info)
    
    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, "dev")
    
    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, "test")
    
    def get_text(self, subj, template=None, context=None):
       return template.replace("[X]", subj).replace("[Y]", "[ANSWER]")

    def _create_examples(self, lines, relation_info=None):
        examples = []
        #template = relation_info[self.relation]["template"]
        for (idx, line) in enumerate(lines):
            subj = line['sub_label']
            obj = line['obj_label']
            template = relation_info[line["predicate_id"]]["template"]
            text_a = self.get_text(subj, template)
            examples.append(InputExample(
                guid=idx,
                text_a=text_a,
                label=obj
                ))
        return examples

processors = {
    "sst2": SST2LabeledProcessor,
    "sicke2b": SICKE2balancedProcessor,
    "fact-retrieval": FactRetrievalProcessor,
}
