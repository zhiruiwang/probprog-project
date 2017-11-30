# Experimentation: Bayes by Backprop in Predictive Maintenance

This folder saves the experimentation code of using *Bayes by Backprop* technique to infer a noisy LSTM recurrent net for the predictive maintenance task. The algorithm is *not complete yet* as we didn't manage to finish debugging the import/export operations from the training to the testing behaviors.

Algorithm see notebook.

## Run

```
python3 bbb.py -download # download the dataset for the first time running
```

use `-[parameter name] [parameter values]` to send customized values into the model.

## Update State Notes

Nov 30, 2017

1. Our code inherits `Tensorflow`'s tutorial of RNN with the Penn Tree Bank dataset, as the `BayesByBackprop` class utilizes many functions within the original `PTBModel` class in [their code](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py).

2. Putting inference within backpropagation on the Penn Tree Bank dataset by user [@mirceamironenco](https://github.com/mirceamironenco/BayesianRecurrentNN/blob/master/bayesian_rnn.py) is an existing great implementation of the original paper. We differ from his implementation in terms of:

- Network flexibility: the `BayesianLSTM` class now allow recurrent net to have different layer sizes by using the `tf.nn.dynamic_rnn` class.
- Input pipeline easy application: generalize data preprocessing and batch-generating processes to cases not limited to the word embedding applications. Now `preprocessing` creates all sequences beforehand. Other datasets are easier to configure through those new functions.

## Issues

Nov 30, 2017

1. Error in the import/export operations: 

```
Error reported to Coordinator: <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>, The node 'Merge/MergeSummary' has inputs from different frames. The input 'Train/Model/rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/bayesian_lstm/bbb_lstm_1_b_rho_hist' is in frame 'Train/Model/rnn/while/Train/Model/rnn/while/'. The input 'Train/Learning_Rate' is in frame ''.
```

2. Docstring not complete

## References

1. [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
Meire Fortunato, Charles Blundell, Oriol Vinyals, 2017

2. [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra, 2017

3. [Tenforflow RNN tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)

4. [@mirceamironenco's implementation on the PTB dataset](https://github.com/mirceamironenco/BayesianRecurrentNN/blob/master/bayesian_rnn.py)
