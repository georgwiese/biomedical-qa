# Neural Biomedical Question Answering

This repository collects the source code of the work published in [Neural Domain Adaptation for Biomedical Question Answering](https://arxiv.org/abs/1706.03610).
It has been written by Dirk Weissenborn and Georg Wiese at the German Research Center for Artificial Intelligence ([DFKI](https://www.dfki.de/web?set_language=en&cl=en)).

## Inference Server via Docker

To merely try out our pre-trained models, the simplest way is to use the pre-built [Docker](https://www.docker.com/) image.
To start server with single model on port 5000:

```bash
$ docker run -it -p 127.0.0.1:5000:5000 \
  georgwiese/biomedical-qa \
  ./start_server.sh single
```

The server accepts POST requests to `localhost:5000/answer` which contain a payload in [BioASQ JSON format](http://participants-area.bioasq.org/general_information/Task5b/).

## Training a Model

The code structure in this repository is as follows:

- `notebooks`: Contains Jupyter Notebooks for analysis.
- `biomedical_qa`: Root directory for all python code.
- `biomedical_qa/tools`: All command line tools.

### Data Sources

Download the datasets from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [BioASQ](http://participants-area.bioasq.org/general_information/Task5b/) (Task B training dataset).
The BioASQ dataset needs to be split into dev and train datasets, e.g. via `biomedical_qa/tools/split_bioasq.py`.

### Build Embedders

Download the [GloVe](https://nlp.stanford.edu/projects/glove/) and [PubMed](http://bioasq.org/news/bioasq-releases-continuous-space-word-vectors-obtained-applying-word2vec-pubmed-abstracts) Embeddings.
Then, use `biomedical_qa/tools/embedder_from_glove_and_pubmed.py` to create *transfer models* that are used by the QA model.

### Example Training

Here is an example to train a model from scratch on SQuAD:

```bash
python3 biomedical_qa/training/train_qa.py \
    --with_question_type_features \
    --size 100 \
    --max_epochs 40 \
    --model_type simple_pointer \
    --data data/squad \
    --trainset_prefix train \
    --validset_prefix dev \
    --start_output_unit sigmoid \
    --ckpt_its 1000 \
    --batch_size 64 \
    --save_dir $SAVE_DIR \
    --dropout 0.5 \
    --dataset squad \
    --transfer_model_config model_checkpoints/glove_pubmed_embedder/config.pickle \
    --transfer_model_path model_checkpoints/glove_pubmed_embedder/model.tf \
    --composition LSTM
```

It assumes the SQuAD `train` and `dev` JSONs are located at `data/SQuAD` and the embedder is stored at `model_checkpoints/glove_pubmed_embedder`.

Fine-tuning on BioASQ works analogously by passing the `--model_config` and `--init_model_path` flags.
