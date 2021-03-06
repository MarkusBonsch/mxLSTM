% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitLSTMmodel.R
\name{fitLSTMmodel}
\alias{fitLSTMmodel}
\title{fitLSTMmodel}
\usage{
fitLSTMmodel(x, y, param, test.x = NULL, test.y = NULL,
  initialModel = NULL, seed = NULL, ...)
}
\arguments{
\item{x}{array containing the x variables for training. See \code{\link{mxLSTM}} for format.}

\item{y}{array containing the y variables for training. See \code{\link{mxLSTM}} for format.}

\item{param}{Parameters as given by caret's tuneGrid. See \code{\link{getLSTMmodel}} for details.}

\item{test.x}{same as x, but for testing after each epoch}

\item{test.y}{same as y, but for testin after each epoch}

\item{initialModel}{A pretrained mxLSTM model as returned by this function. Training will continue 
on initialModel, using its weights as a starting point.
Be careful, all data shape, hidden layers, and preProcessing must be identical to the 
first training of initialModel.}

\item{seed}{Random seed tp be set before model training. Defaults to NULL for no explicit seed setting}

\item{...}{further arguments for the call to \code{\link{mxLSTM}}}
}
\value{
list of class mxLSTM. 
       This is the result of \code{\link{mxLSTM}}
       Additionally, some other list items are added:
       \itemize{
       \item{xVariables:}{ Names of the xVariables}
       \item{yVariables:}{ Names of the yVariables}
       }
}
\description{
Fits an mxNet LSTM recurrent neural network. For use in getLSTMmodel
}
\seealso{
\code{\link{mxLSTM}}, \code{\link{predictLSTMmodel}}, \code{\link{getLSTMmodel}}
}
