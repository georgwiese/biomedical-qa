import os
import json

from biomedical_qa.data.bioasq_squad_builder import BioAsqSquadBuilder
from biomedical_qa.sampling.squad import SQuADSampler


class BioAsqSampler(SQuADSampler):


    def __init__(self, dir, filenames, batch_size, vocab,
                 instances_per_epoch=None, shuffle=True, dataset_json=None,
                 types=None, split_contexts_on_newline=False,
                 context_token_limit=-1, include_synonyms=False,
                 tagger=None, include_answer_spans=True):

        if dataset_json is None:
            # load json
            with open(os.path.join(dir, filenames[0])) as dataset_file:
                dataset_json = json.load(dataset_file)

        squad_builder = BioAsqSquadBuilder(dataset_json,
                                           types=types,
                                           context_token_limit=context_token_limit,
                                           include_synonyms=include_synonyms,
                                           include_answer_spans=include_answer_spans)
        squad_json = squad_builder.build().get_result_object()

        SQuADSampler.__init__(self, None, None, batch_size, vocab,
                              instances_per_epoch=instances_per_epoch,
                              shuffle=shuffle, types=types,
                              split_contexts_on_newline=split_contexts_on_newline,
                              dataset_json=squad_json, tagger=tagger)
