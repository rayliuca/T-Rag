import json
import typing

import datasets
import hanlp
import torch
from elasticsearch import client as elastic_client
from jinja2 import Template as JinjaTemplate

DEFAULT_PROMPT_TEMPLATE = """[INST] <|system|> Translate the given text to {{target_language}}:
<|user|> 
{{source_text}}
[/INST]
<|context|> [CON]
Glossary: {{glossary_data}}
Reference translations: {{ref_trans_data}}
[/CON]
<|assistant|>"""


def serialize_str(s):
    return json.dumps(s, ensure_ascii=False)


class InputProcessor:
    def __init__(self,
                 device=None,
                 hanlp_model=None,
                 hanlp_model_id=hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE,
                 prompt_template: typing.Optional[str, JinjaTemplate] = None
                 ):
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.hanlp_model = hanlp_model
        if hanlp_model is None:
            self.hanlp_model = hanlp.load(hanlp_model_id).to(self.device)

        if prompt_template is None:
            self.prompt_template = JinjaTemplate(DEFAULT_PROMPT_TEMPLATE)
        else:
            if isinstance(prompt_template, str):
                self.prompt_template = JinjaTemplate(prompt_template)
            elif isinstance(prompt_template, JinjaTemplate):
                self.prompt_template = prompt_template
            else:
                raise ValueError()

        self.es_client = None
        self.general_memory_index_key = None
        self.general_memory: typing.Optional[datasets.Dataset] = None
        self.task_memory: dict = {}

        self.general_glossary: dict = {}
        self.task_glossary: dict = {}

    def load_general_translation(self,
                                 dataset_disk_path,
                                 index_key='ja',
                                 elasticsearch_host: str = "localhost",
                                 elasticsearch_port: int = 9200,
                                 es_client: typing.Optional[elastic_client] = None,
                                 dataset_args={},
                                 elastic_args={},
                                 ):
        """
        Load the general translation examples
        """

        self.general_memory = datasets.load_from_disk(dataset_disk_path, **dataset_args)

        # initiate the elastic index
        if es_client is None:
            self.general_memory.add_elasticsearch_index(index_key, host=elasticsearch_host, port=elasticsearch_port,
                                                        **elastic_args)
        else:
            self.es_client = es_client
            self.general_memory.add_elasticsearch_index(index_key, es_client=es_client, **elastic_args)

        self.general_memory_index_key = index_key

        return self.general_memory

    def load_task_translation(self):
        """
        Load the general translation examples
        """
        raise NotImplementedError()
        pass

    def search_general_memory(self, text, search_index: str = None, k=4, max_item_len=500, **search_kwargs):
        """
        search general translation examples using elasticsearch
        """
        if search_index is None:
            search_index = self.general_memory_index_key

        mem_scores, mem_indices = self.general_memory.search_batch(search_index, text, k=k, **search_kwargs)

        ref_trans_data = [self.general_memory[midx] for midx in mem_indices]

        # truncate in case the example translations are too long
        for i in range(len(ref_trans_data)):
            for k in ref_trans_data[i]:
                ref_trans_data[i][k] = ref_trans_data[i][k][:max_item_len]
        return ref_trans_data

    def get_task_memory(self, client=None):
        """
        search in-task translation examples using elasticsearch
        """
        raise NotImplementedError()

    def load_general_glossary(self, glossary_json_path, encoding="utf8"):
        """
        Load the general glossary (i.e. wikidata title pair/ dictionary entries )
        format: {
            "original text" : ["translation 1", "translation 2"],
            "original text2" : ["translation 3", "translation 4"],
        }
        """
        with open(glossary_json_path, "r", encoding=encoding) as f:
            self.general_glossary = json.load(f)

        # sort by descending original text length
        self.general_glossary = {
            k: self.general_glossary[k] for k in sorted(self.general_glossary, key=len, reverse=True)
        }

        return self.general_glossary

    def load_task_glossary(self, glossary_json_path, glossary_index, encoding="utf8"):
        # raise NotImplementedError()

        with open(glossary_json_path, "r", encoding=encoding) as f:
            glossary_dict = json.load(f)

        # sort by descending original text length
        glossary_dict = {
            k: glossary_dict[k] for k in sorted(glossary_dict, key=len, reverse=True)
        }
        self.task_glossary[glossary_index] = glossary_dict

        return self.task_glossary[glossary_index]

    def search_glossary(self, text, max_k=10, task_index=None, search_general_glossary=True):
        found_glossary = []
        task_glossary = {}

        if task_index is not None:
            task_glossary = self.task_glossary[task_index]

        for word in self.hanlp_model(text)['tok']:
            # search in task glossary first
            if word in task_glossary:
                found_glossary.append((word, self.general_glossary[word]))
            elif word in self.general_glossary and search_general_glossary:
                found_glossary.append((word, self.general_glossary[word]))

        return found_glossary[:max_k]

    def search_general_glossary(self):
        raise NotImplementedError()

    def search_task_glossary(self, glossary_index):
        raise NotImplementedError()

    def render_prompt(self,
                      source_text,
                      glossary_data: list = [],
                      ref_trans_data: list = [],
                      target_language='English',
                      **kwargs):

        return self.prompt_template.render(
            source_text=source_text,
            glossary_data=serialize_str(glossary_data),
            ref_trans_data=serialize_str(ref_trans_data),
            target_language=target_language,
            **kwargs
        )

    def build_prompt(self,
                     source_text,
                     target_language="English",
                     memory_search_args={},
                     glossary_search_args={},
                     prompt_args={}
                     ):
        glossary_data = self.search_glossary(source_text, **glossary_search_args)
        ref_trans_data = self.search_general_memory(source_text, **memory_search_args)

        return self.render_prompt(source_text, glossary_data=glossary_data, ref_trans_data=ref_trans_data,
                                  target_language=target_language, **prompt_args)
