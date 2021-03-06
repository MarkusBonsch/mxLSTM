% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mxLSTM.R, R/mxLSTMsource.R
\docType{package}
\name{mxLSTM}
\alias{mxLSTM}
\alias{mxLSTM-package}
\alias{mxLSTM}
\title{mxLSTM: A library facilitating regression analysis with LSTMs}
\usage{
mxLSTM(x, y, num.epoch, test.x = NULL, test.y = NULL, num.hidden,
  optimizeFullSequence = FALSE, dropoutLstm = num.hidden * 0,
  zoneoutLstm = num.hidden * 0, batchNormLstm = FALSE, gammaInit = 1,
  batch.size = 128, activation = "relu", optimizer = "rmsprop",
  learning.rate = 0.002, initializer = mx.init.Xavier(), shuffle = TRUE,
  initialModel = NULL, ...)
}
\arguments{
\item{x}{array containing the features:
\itemize{
  \item{Dimension 1:} one entry for each feature
  \item{Dimension 2:} one entry for each element in the sequence
  \item{Dimension 3:} one entry for each training event
  }
  Use \code{\link{transformLSTMinput}} to transform data.frames into this structure.}

\item{y}{array containing the target labels:
\itemize{
  \item{Dimension 1:} one entry for each output variable
  \item{Dimension 2:} one entry for each element in the sequence
  \item{Dimension 3:} one entry for each training event
  }
  Use \code{\link{transformLSTMinput}} to transform data.frames into this structure.}

\item{num.epoch}{integer number of training epochs over full ldataset}

\item{test.x}{same as x, but for testing, not for training}

\item{test.y}{same as y but for testing, not for training}

\item{num.hidden}{integer vector of flexible length. For each entry, an LSTM layer with the corresponding number of
neurons is created.}

\item{optimizeFullSequence}{Boolean. If TRUE, each sequence element is in the output and adds to the loss.
If FALSE (default), only the last element of each sequence will be used to optimize the model and 
the outputs of the rest of the sequence are not available in the output.}

\item{dropoutLstm}{numeric vector of same length as num.hidden. Specifies the dropout probability for each LSTM layer.
Dropout is applied according to Cheng et al. "An exploration of dropout with LSTMs". 
Difference: we employ a constant dropout rate; we do per element dropout.}

\item{zoneoutLstm}{numeric vector of same length as num.hidden. Specifies the zoneout probability for each LSTM layer.
Zoneout is implemented according to 
Krueger et al. 2017 "Zoneout: Regularizing RNNs by randomly preserving hidden activations".}

\item{batchNormLstm}{logical. If TRUE, each LSTM layer is batch normalized according to the recommendations in
T. Cooljmans et al. ILRC 2017 "Recurrent batch normalization".}

\item{gammaInit}{numeric value. Will be used to initialize the gamma matrices of batchNormLayers. 
Cooljmans et al. recommend 0.1 (for use with tanh activation), mxnet default is 1.
My experience: 0.1 works very poorly with relu activation.}

\item{batch.size}{self explanatory}

\item{activation}{activation function for update layers in the LSTM cells. "relu" or "tanh"}

\item{optimizer}{character specifying the type of optimizer to use.}

\item{learning.rate}{learning rate for the optimizer. Can be a single number or a named vector for adaptive learning rate.
If it is a vector, the names have to specify the epoch at which this value becomes active. For example
\code{learning.rate = c(1=0.004, 30 = 0.002, 50 = 0.0005)} will train epochs 1 to 29 with \code{0.004}, epochs 30 to 49
with \code{0.002} and everything after 50 with \code{0.0005}}

\item{initializer}{random initializer for weights}

\item{shuffle}{Boolean. Should the training data be reordered randomly prior to training? 
(reorders full sequences, order within each sequence is unaffected.)}

\item{initialModel}{mxLSTM model. If provided, all weights are initialized based on the given model.}

\item{...}{Additional arguments to optimizer}
}
\value{
object of class mxLSTM: list: a symbol, arg.params, aux.params, a log, and the variable names
}
\description{
Provides functions for doing and evaluating regressions with LSTMs.

Builds an LSTM model
}
\details{
sequence length is inferred from input (dimension 2).
}
\examples{
\dontrun{
library(mxLSTM)
library(data.table)
## simple data: two numeric outputs as a function of two numeric inputs.
## including lag values
## with some noise.
nObs <- 20000
dat <- data.table(x = runif(n = nObs, min = 1000, max = 2000),
                  y = runif(n = nObs, min = -10, max = 10))
## create target
dat[, target := 0.5 * x + 0.7 * lag(y, 3) - 0.2 * lag(x, 5)]
dat[, target2 := 0.1 * x + 0.3 * lag(y, 1) - 0.4 * lag(x, 2)]
dat[, target := target + rnorm(nObs, 0, 10)]
dat[, target2 := target2 + rnorm(nObs, 0, 10)]

## convert to nxLSTM input
dat <- transformLSTMinput(dat = dat, targetColumn = c("target", "target2"), seq.length = 5)

## split into training and test set
trainIdx <- sample(dim(dat$x)[3], as.integer(dim(dat$x)[3]/2))
testIdx  <- seq_len(dim(dat$x)[3])[-trainIdx]

## train model
model <- mxLSTM(x = dat$x[,,trainIdx], 
                y = dat$y[,,trainIdx], 
                num.epoch = 50, 
                num.hidden = 64, 
                dropoutLstm = 0, 
                zoneoutLstm = 0.01, 
                batchNormLstm = TRUE, 
                batch.size = 128, 
                optimizer = "rmsprop",
                learning.rate =  c("1" = 0.005, "20" = 0.002, "40" = 0.0005))

## plot training history
plot_trainHistory(model)

## get some predictions (on test set)
predTest <- predictLSTMmodel(model = model, dat = dat$x[,,testIdx], fullSequence = FALSE)

## nice plot
plot_goodnessOfFit(predicted = predTest$y1, observed = dat$y[1,5, testIdx])
plot_goodnessOfFit(predicted = predTest$y2, observed = dat$y[2,5, testIdx])

## save the model
## saveLstmModel(model, "testModel")
}
}
\seealso{
\code{\link{fitLSTMmodel}}, \code{\link{predictLSTMmodel}}, \code{\link{getLSTMmodel}},
         \code{\link{plot_trainHistory}}
}
