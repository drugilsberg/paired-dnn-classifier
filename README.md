# paired-dnn-classifier

Create a DNN classifier for word pairs starting from pretrained word vectors.

## Set-up the environment

Create a `venv`:

```sh
python -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Install `jupiter` playground:

```sh
pip install jupyter
ipython kernel install --user --name=paired-dnn-classifier
```

## Generate synthetic data

Run the `generate_syntheric_data.py` adjusting parameters as needed.

## Train the model for word pairs classification

Edit and run the `paired_dnn_classifier.py` and play with different architectures and parameters.
THe script can be easily adapted into a jupyter notebook.