# mxLSTM
LSTM models for regression analysis with mxNet.

# Features
Based on the excellent example [here](https://www.r-bloggers.com/recurrent-models-and-examples-with-mxnetr/).

Added following features:
- takes numeric inputs instead of text (for timeseries regression)
- zoneout as described in Krueger et al. 2017 "Zoneout: Regularizing RNNs by randomly preserving hidden activations".
- batch normalization as described in T. Cooljmans et al. ILRC 2017 "Recurrent batch normalization".
- choice of different activation functions.

