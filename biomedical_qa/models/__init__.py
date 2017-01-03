import tensorflow as tf

from biomedical_qa.models.embedder import Embedder,WordEmbedder,CharWordEmbedder, ConcatEmbedder, \
    ConstantWordEmbedder
from biomedical_qa.models.context_embedder import ContextEmbedder,RNNContextEmbedder,AttentionMemoryContextEmbedder


class QASetting:
    def __init__(self, question, answers, contexts,
                 answers_spans=None,
                 answer_candidates=None,
                 answer_candidate_spans=None,
                 id=None,
                 q_type=None,
                 is_yes=None,
                 paragraph_json=None,
                 question_json=None):
        """
        :param question: list of indices
        :param answers:  list of list of list of indices:
                         (answers -> alternatives -> token ids)
        :param contexts: list of list indices
        :param answer_spans: list of list of (context_index, start, end) tuples
        :return:
        """
        self.question = question
        self.answers = answers
        self.contexts = contexts
        self.answers_spans = answers_spans
        self.answer_candidates = answer_candidates
        self.answer_candidate_spans = answer_candidate_spans
        self.id = id
        self.q_type = q_type
        self.is_yes = is_yes
        self.paragraph_json = paragraph_json
        self.question_json = question_json

    def translate(self, vocab, unk_id):
        self.question = [vocab.get(w, unk_id) for w in self.question]
        self.contexts = [[vocab.get(w, unk_id) for w in c] for c in self.contexts]
        self.answers = [[vocab.get(w, unk_id) for w in a] for a in self.answers]

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
    elif type == "simple_pointer":
        from genie_qa.models.qa_pointer import QASimplePointerModel
        return QASimplePointerModel.create_from_config(config, devices, dropout)
    else:
        raise NotImplementedError("Unknown model type: %s" % type)
