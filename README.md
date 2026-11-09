# Character-level text generator

An early experiment in generating text one character at a time with Keras.
I worked on this in 2017 and uploaded this snapshot in August 2020.

## What it does

[`model.py`](model.py) trains a single-layer, 128-unit LSTM to predict the next
character from a 40-character window. It lowercases the text, builds one-hot
training examples with a stride of three, and trains for 60 epochs.

After each epoch, it prints 400-character continuations at sampling temperatures
of 0.2, 0.5, 1.0, and 1.2. Changing the temperature changes how adventurous the
sampling is.

The current script reads [`war_and_peace.txt`](war_and_peace.txt). The repository
also contains Shakespeare, William Blake, and *Capital* text files used for
experimentation.

## Files

- [`model.py`](model.py): training and text generation.
- [`first_utils.py`](first_utils.py): additional Keras/data-preparation helpers.
- [`utils.py`](utils.py): NumPy RNN forward/backward-pass helpers.
- `*.txt`: text corpora.

The main script does not import either helper file.

## Status

Preserved learning code from my early work with generative models. The original
dependency versions were not recorded. It uses legacy Keras and NumPy APIs,
including `np.bool`, and has not been tested with current versions. There are no
saved model weights or recorded evaluation results in this repository.

To revisit it, start with `model.py` and the corpus selection, then restore a
compatible environment or update the legacy APIs before training.

