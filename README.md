# IMDb Reviews - NLP Model

An end-to-end Natural Language Processing (NLP) project that classifies IMDb movie reviews as positive or negative using TensorFlow and Keras.

## Dataset

- **Source:** [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 labeled movie reviews
- **Classes:** Positive / Negative
- **Task:** Binary text classification (sentiment analysis)

## Project Structure

```
├── NLP_fundumentals.ipynb   # Original exploratory notebook
├── config.py                # Shared pipeline constants
├── requirements.txt
└── src/
    ├── data.py               # Dataset download, loading, train/test split
    ├── pipeline.py            # tf.data pipeline + TextVectorization
    ├── models.py               # Model architectures (dense, dense_v2, LSTM)
    ├── evaluate.py            # Metrics, loss/accuracy plots, confusion matrix
    └── train.py                # CLI entrypoint that wires everything together
```

## Usage

Install dependencies:

```
pip install -r requirements.txt
```

Train a model (run from inside `src/`):

```
cd src
python train.py --model dense_v2
```

Available `--model` choices:
- `dense` — Embedding + GlobalAveragePooling1D baseline (expects pre-vectorized input)
- `dense_v2` — same baseline with the vectorizer and dropout layers baked into the model, so it accepts raw text directly (default)
- `lstm` — Embedding + Bidirectional LSTM sequence model

## Workflow

1. Load and explore the IMDb dataset with pandas.
2. Split the data into training and test sets while preserving class balance.
3. Build an efficient `tf.data` pipeline for batching, shuffling, and prefetching.
4. Convert raw text into numerical sequences with `TextVectorization`.
5. Learn dense word representations with an `Embedding` layer.
6. Train and compare multiple architectures:
   - Embedding + GlobalAveragePooling1D (baseline)
   - Embedding + Bidirectional LSTM (sequence model)
7. Evaluate models with loss/accuracy curves, a confusion matrix, and precision/recall/F1.
8. Predict sentiment on unseen movie reviews.

## Technologies

- Python
- TensorFlow / Keras
- tf.data API
- TextVectorization
- Embedding layers
- Bidirectional LSTM
- Pandas
- Scikit-learn

## Concepts Practiced

- NLP preprocessing
- Vocabulary construction
- Tokenization
- Sequence padding and truncation
- Word embeddings
- Sequence modeling with LSTMs
- Binary text classification
- TensorFlow input pipelines
- Model evaluation and prediction

## Skills Demonstrated

- End-to-end NLP pipeline development
- Deep learning with TensorFlow
- Efficient data pipeline creation using tf.data
- Building and training recurrent neural networks
- Text classification and sentiment analysis
git 