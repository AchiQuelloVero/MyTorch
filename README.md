# myTorch

*Team project (2 members) at Epitech Barcelona, Oct–Nov 2026.*

Neural networks coded from scratch (Epitech, Computational Learning Systems).
No neural network library is used: [NumPy](https://numpy.org) only handles the linear algebra.

The project has two parts:

- **the NN module** (`src/nn/`): create, train, save and load a neural network;
- **the analyzer** (`my_torch_analyzer`): reads chessboards in
  [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and tells if the position is
  `Checkmate`, `Check` or `Nothing`.

## Requirements

- Python 3
- NumPy
- `make`
- A POSIX shell (`sh`, `bash` or `zsh`) for the scripts

The Epitech test image (`epitechcontent/epitest-docker`) has Python 3.14.4 and NumPy 2.3.5, so the
analyzer runs there without a virtual environment. `requirements.txt` pins the same NumPy version.

Check that the machine has everything:

```sh
sh scripts/check_env.sh
```

## Installation

```sh
git clone https://github.com/AchiQuelloVero/mytorch.git myTorch
cd myTorch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make
```

> Run `source .venv/bin/activate` again in every new terminal. Run `deactivate` to leave it.

| Rule | Effect |
|---|---|
| `make` | builds `my_torch_analyzer` and `my_torch_generator` |
| `make clean` | removes Python caches |
| `make fclean` | `clean` + removes `my_torch_analyzer` and `my_torch_generator` |
| `make re` | `fclean` + `make` |
| `make tests_run` | runs the unit tests |

## Usage

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Argument | Meaning |
|---|---|
| `--train` | train the network on CHESSFILE (each line: FEN + expected output) |
| `--save SAVEFILE` | where to save the trained network (default: LOADFILE) |
| `--predict` | print one prediction per chessboard, in the same order as CHESSFILE |
| `LOADFILE` | network file (`.nn`) |
| `CHESSFILE` | chessboards, one FEN per line |

`--predict` prints one line per chessboard on the standard output, and nothing else:

```
$> ./my_torch_analyzer --predict my_torch_network_full.nn chessboards.txt
Checkmate Black
Nothing
Check White
```

- The network's configuration chooses the format: `Check` or `Check White`, see the `output` key in
  [Configuration](#configuration-conf).
- When every line of CHESSFILE also has its expected output (`Check` or `Check White`), the accuracy is
  printed on the error output: `accuracy 89.31% on 71322 boards`.

`--train` needs the expected output on every line of CHESSFILE:

- A part of CHESSFILE is held back to measure the network on boards it doesn't train on (10% by
  default, see the `validation_split` key in [Configuration](#configuration-conf)). It is stratified: it
  keeps the proportion of every label.
- The progress is printed on the error output, one line per epoch:
  `epoch 3/30  loss 0.2519  accuracy 89.88%  val_loss 0.3011  val_accuracy 87.92%`.
- With `early_stopping`, the training stops when the validation loss stops improving, and the network
  that is saved is the one of the best epoch, not the last one: `best network: epoch 3 of 8 (val_loss 0.7057)`.
- The network is saved only once the training is over, to a temporary file that then replaces SAVEFILE:
  if anything fails, LOADFILE is left unchanged.

On any error the program prints a message on the error output and exits with code `84`.

### Generator

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

Creates `nb_i` new random networks from each configuration file, in the current folder:

```sh
./my_torch_generator config/basic.conf 3      # basic_1.nn, basic_2.nn, basic_3.nn
```

- Every network gets its own seed, so they all start from different weights: `seed`, `seed + 1`, …
  when the configuration has a `seed`, a random one otherwise.
- The seed used is written in the `config` of the `.nn` file, so any network can be generated again.
- Every configuration is checked before the first file is written: on an error nothing is created
  and the program exits with code `84`.
- Generated `.nn` files are ignored by git, except the pre-trained `my_torch_network*.nn`.

## Dataset

The dataset (`dataset/<class>/<pieces>.txt`, 723,019 labeled boards) is not versioned. Put it in
`dataset/`, then build the training, validation and test files:

```sh
scripts/build_dataset.py              # options: --dataset DIR, --out DIR, --seed N (default 42)
```

It writes `dataset/splits/train.txt`, `val.txt` and `test.txt` (80 / 10 / 10 %, same format as the
dataset) and prints the number of boards per label in each:

- **Duplicates are removed first**: two boards count as the same when the network sees the same input
  (see [Board encoding](#board-encoding)), so no board of the test set was seen during training.
  9,801 boards are removed this way.
- **The split is stratified**: each file keeps the proportion of every label.
- **It is reproducible**: the same seed always gives byte-identical files.

## Training the pre-trained networks

The networks at the root of the repository (`my_torch_network_basic.nn` and
`my_torch_network_full.nn`) are rebuilt from the dataset with one command:

```sh
scripts/train.sh
```

1. `make`, then `scripts/build_dataset.py --seed 42` rebuilds the splits.
2. `my_torch_generator config/best.conf 1` creates the untrained network. `config/best.conf` is the
   configuration chosen by the hyperparameter search (see `docs/benchmarks.md`).
3. `my_torch_analyzer --train` trains it on `train.txt`, holding back 10% of it to stop at the best epoch,
   and saves `my_torch_network_basic.nn`.
4. `my_torch_network_full.nn` is the same network with `"output": "full"`: it prints `Check White`
   instead of `Check`.

Every random step uses a seed (the split, the boards held back, the initial weights, the order of the
mini-batches and dropout), so running the script again gives byte-identical networks. `val.txt` and
`test.txt` are never used for training: they measure the networks.

On `test.txt` the networks reach **92.84% accuracy** (macro-F1 0.9223). To measure a network:

```sh
scripts/evaluate.py my_torch_network_full.nn          # accuracy, confusion matrices, precision/recall/F1
```

The detailed results are in [docs/benchmarks.md](docs/benchmarks.md#evaluation-on-the-test-set).

## NN module

The module only works with numbers: it knows nothing about chess.

```python
from src.nn.network import Network

net = Network.from_config("config/basic.conf")
history = net.train(x_train, y_train, x_val, y_val, class_weights=weights, verbose=True)
probabilities = net.predict(x_test)
```

| Method | Input | Output |
|---|---|---|
| `train(x, y, x_val=None, y_val=None, class_weights=None, verbose=False)` | `x`: `(n, input_size)`, `uint8` or float. `y`: class indices `(n,)` or one-hot `(n, classes)` | one dict per epoch: `epoch`, `loss`, `accuracy`, and `val_loss`, `val_accuracy` when validation data is given |
| `predict(x)` | `(n, input_size)` | `(n, classes)` probabilities |
| `evaluate(x, y)` | same as `train` | `(loss, accuracy)` |
| `save(path)` | file path | writes the network as a `.nn` file (see [Network](#network-nn)) |
| `Network.load(path)` | file path | the saved network, ready to predict or to continue training |

- Each epoch shuffles the training set, then updates the weights once per mini-batch of `batch_size`
  boards, with the optimizer of the configuration.
- `class_weights` gives one weight per class: the loss of every board is multiplied by the weight of its
  class, so rare classes count as much as common ones.
- `verbose=True` prints one line per epoch on the error output.
- When `early_stopping` is set and validation data is given, the network with the lowest validation loss
  is kept in memory. The training stops after `early_stopping` epochs without improvement, and the
  best network is put back at the end, even if the training was not stopped.
- `l2` adds `l2 * weight` to the gradient of every weight (not the biases), which keeps the weights
  small. It only changes the updates: the `loss` and `val_loss` that are printed don't include the
  penalty, so they can be compared.
- `dropout` turns off that part of the neurons of every hidden layer, chosen at random for each board at
  each training step, and scales up the others by `1 / (1 - dropout)`. The output layer is never dropped.
  `predict` and `evaluate` use every neuron, so the `loss` and `accuracy` of the training, which are
  measured with dropout, can be worse than the validation ones.
- Inputs are converted to floats one mini-batch at a time (and `predict` works by blocks of 4,096
  boards), so the whole `uint8` dataset never has to be converted at once.

## File formats

Both formats are JSON, so they are easy to read, to diff, and to load even without NumPy.

### Configuration (`.conf`)

Describes a network before training: its shape and its hyperparameters. Example `config/basic.conf`:

```json
{
    "input_size": 776,
    "layers": [
        {"size": 256, "activation": "relu"},
        {"size": 64, "activation": "relu"},
        {"size": 3, "activation": "softmax"}
    ],
    "classes": ["Nothing", "Check", "Checkmate"],
    "init": "he",
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "lr_decay": 0.0,
    "batch_size": 64,
    "epochs": 30,
    "validation_split": 0.1,
    "dropout": 0.0,
    "l2": 0.0001,
    "early_stopping": 5,
    "seed": 42
}
```

| Key | Values | Meaning |
|---|---|---|
| `input_size` | integer | size of the input vector: 776, see [Board encoding](#board-encoding) |
| `layers` | list | hidden and output layers, in order: `size` and `activation` |
| `activation` | `sigmoid`, `tanh`, `relu`, `softmax` | activation function of a layer |
| `classes` | list of strings | output labels, in the order of the output neurons |
| `output` | `basic`, `full` | what `--predict` prints: `Check`, or `Check White` with the color of the attacker (default `basic`) |
| `init` | `xavier`, `he`, `lecun` | weight initialization (default `xavier`); a layer can override it with its own `init` key |
| `loss` | `cross_entropy` (softmax output), `binary_cross_entropy` (sigmoid output), `mse` | cost function |
| `optimizer` | `sgd`, `momentum`, `adam` | weight update rule (default `sgd`) |
| `learning_rate` | float > 0 | step size (default `0.01`) |
| `momentum` | float | `momentum` optimizer only: part of the last step kept (default `0.9`) |
| `beta1`, `beta2` | floats | `adam` optimizer only: decay of the gradient mean and variance (default `0.9`, `0.999`) |
| `lr_decay` | float ≥ 0 | learning rate at epoch `e` = `learning_rate / (1 + lr_decay * e)` (default `0`) |
| `batch_size` | integer | samples per gradient step: 1 = stochastic, dataset size = full batch (default `32`) |
| `epochs` | integer | maximum number of passes on the training set (default `10`) |
| `validation_split` | float in [0, 1) | part of CHESSFILE that `--train` holds back for validation, 0 = none (default `0.1`) |
| `dropout` | float in [0, 1) | probability to drop a hidden neuron during training (default `0` = off) |
| `l2` | float ≥ 0 | L2 regularization strength, on the weights only (default `0` = off) |
| `early_stopping` | integer ≥ 0 | stop after this many epochs without validation improvement and keep the best network; needs validation data (default `0` = off) |
| `seed` | integer | random seed, for reproducible results |

### Board encoding

Each FEN becomes a vector of 776 values (0 or 1), built by `src/chess/encoding.py`:

| Indices | Content |
|---|---|
| 0–767 | 12 piece types × 64 squares, one-hot: `piece × 64 + square`, pieces in the order `PNBRQKpnbrqk`, squares from a8 to h1 |
| 768–775 | en passant file (a to h), all 0 when there is none |

The board is always seen **from the side to move**: when Black is to move, the board is flipped
vertically and the colors are swapped. "White" in the input always means the side to move, so the
network learns one pattern ("the side to move is in check") instead of one per color, and no
side-to-move input is needed. The color printed by the analyzer still comes from the FEN.

Castling rights are left out: castling is never legal while in check, so it can never change the
label. En passant is kept: capturing en passant can be the only way out of a check.

### Network (`.nn`)

A trained (or freshly generated) network: the configuration it was built from, plus its weights.

```json
{
    "format": "my_torch_network",
    "version": 1,
    "config": { "...": "the .conf content" },
    "classes": ["Nothing", "Check", "Checkmate"],
    "input_size": 776,
    "layers": [
        {
            "activation": "relu",
            "weights": [[0.012, -0.031, "..."], "..."],
            "biases": [0.0, "..."]
        }
    ]
}
```

- `weights` of a layer is a matrix of `input size × layer size`.
- `biases` has one value per neuron of the layer.
- Only what is needed to predict and to continue the training is saved. The optimizer state is not
  saved: a new `--train` starts the optimizer from zero.
- Numbers are written with full precision, so a loaded network gives exactly the same predictions as
  the one that was saved. A network built from `config/basic.conf` takes about 4.7 MB.
- `Network.load` rebuilds the network from `config`, then checks the file before using it: right
  `format` and `version`, `classes` and `input_size` equal to the config, one entry per layer with
  the same activation, weights and biases of the right shape and only finite numbers. Anything else
  raises a `ValueError` that names the file and the layer, so the analyzer can print it and exit
  with `84`.

The pre-trained networks used by the autograder are at the root of the repository and their name
starts with `my_torch_network`.

## Project layout

| Path | Content |
|---|---|
| `Makefile` | builds `my_torch_analyzer` and `my_torch_generator` |
| `src/analyzer.py` | the analyzer program |
| `src/generator.py` | the generator program |
| `src/nn/` | the NN module: layers, activations, loss, optimizers, save/load |
| `src/chess/` | FEN parser, board encoding, labels, dataset loading |
| `src/metrics.py` | confusion matrix, precision, recall, F1 |
| `config/` | network configurations (`.conf`) |
| `scripts/` | environment check, training, hyperparameter search, benchmarks |
| `tests/` | unit tests |
| `docs/` | benchmarks and graphs |
| `dataset/` | training data (not versioned, too heavy) |
