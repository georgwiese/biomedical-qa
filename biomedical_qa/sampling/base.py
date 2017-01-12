import random
import abc

from nltk.tokenize import RegexpTokenizer


class BaseSampler:

    def __init__(self, batch_size, vocab, instances_per_epoch=None, shuffle=True):
        self.__batch_size = batch_size
        self.unk_id = vocab["<UNK>"]
        self.start_id = vocab["<S>"]
        self.end_id = vocab["</S>"]
        self.vocab = vocab
        self.instances_per_epoch = instances_per_epoch
        self.num_batches = 0
        self.epoch = 0
        self._rng = random.Random(28739)
        self.tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        self._qas, self.char_offsets = self.build_questions()

        assert len(self._qas) > 0

        if shuffle:
            self._rng.shuffle(self._qas)
        if instances_per_epoch is not None:
            self._qas = self._qas[:instances_per_epoch]
        self._idx = 0


    @abc.abstractmethod
    def build_questions(self):
        """Should return an Array of Questions and a char_offsets map."""
        pass


    def get_ids_and_offsets(self, s):
        idxs = []
        offsets = []
        offset = 0
        for t in self.tokenizer.tokenize(s):
            offset = s.index(t, offset)
            offsets.append(offset)
            i = self.vocab.get(t, self.unk_id)
            idxs.append(i)
            offset += len(t)
        return idxs, offsets


    def get_batch(self):
        qa_settings = [self._qas[i+self._idx] for i in range(min(self.__batch_size, len(self._qas) - self._idx))]
        self._idx += len(qa_settings)

        if self._idx == len(self._qas):
            self.epoch += 1
            self._rng.shuffle(self._qas)
            self._idx = 0

        return qa_settings


    def reset(self):
        self._idx = 0


    def get_questions(self):

        return self._qas
