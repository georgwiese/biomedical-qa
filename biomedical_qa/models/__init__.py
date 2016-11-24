import tensorflow as tf

from biomedical_qa.models.embedder import Embedder,WordEmbedder,CharWordEmbedder, ConcatEmbedder, \
    ConstantWordEmbedder
from biomedical_qa.models.context_embedder import ContextEmbedder,RNNContextEmbedder,AttentionMemoryContextEmbedder

""" Largest answer in wikireading
A History of the Clan MacLean from Its First Settlement at Duard Castle, in the Isle of Mull, to the Present Period: Including a Genealogical Account of Some of the Principal Families Together with Their Heraldry, Legends, Superstitions, etc.
"""
wikireading_max_answer_length = 46 + 2 # including <S> and </S>
wikireading_max_question_length = 10

class QASetting:
    def __init__(self, question, answers, context,
                 answer_spans=None,
                 answer_candidates=None,
                 answer_candidate_spans=None,
                 id=None):
        """
        :param question: list of indices
        :param answers:  list of list of indices
        :param context: list of indices
        :return:
        """
        self.question = question
        self.answers = answers
        self.context = context
        self.answer_spans = answer_spans
        self.answer_candidates = answer_candidates
        self.answer_candidate_spans = answer_candidate_spans
        self.id = id

    def translate(self, vocab, unk_id):
        self.question = [vocab.get(w, unk_id) for w in self.question]
        self.context = [vocab.get(w, unk_id) for w in self.context]
        self.answers = [[vocab.get(w, unk_id) for w in a] for a in self.answers]

    @staticmethod
    def from_dict(d):
        return QASetting(d["question"], d["answers"], d["context"], d.get("answer_spans", None),
                         d.get("answer_candidates", None), d.get("answer_candidate_spans", None))

from biomedical_qa.models.qa_pointer import QAPointerModel


def model_from_config(config, devices=None, dropout=0.0, inputs=None, seq_lengths=None, reuse=False):
    devices = ["/cpu:0"] if devices is None else devices
    type = config.get("type")
    if type is None and "embedders" in config:
        return ConcatEmbedder.create_from_config(config, devices, dropout, inputs, seq_lengths, reuse)
    elif type == "rnn_context":
        return RNNContextEmbedder.create_from_config(config, devices, dropout, reuse=reuse)
    elif type == "attention_context":
        return AttentionMemoryContextEmbedder.create_from_config(config, devices, dropout, reuse=reuse)
    elif type == "word":
        return WordEmbedder.create_from_config(config, inputs=inputs, seq_lengths=seq_lengths, reuse=reuse)
    elif type == "constant_word":
        return ConstantWordEmbedder.create_from_config(config, inputs=inputs, seq_lengths=seq_lengths, reuse=reuse)
    elif type == "charword":
        return CharWordEmbedder.create_from_config(config, device=devices[0], inputs=inputs, seq_lengths=seq_lengths,
                                                   reuse=reuse)
    elif type == "pointer":
        return QAPointerModel.create_from_config(config, devices, dropout)
    else:
        raise NotImplementedError("")
