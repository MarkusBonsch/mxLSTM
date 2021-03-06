% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/saveCaretLstmModel.R
\name{saveCaretLstmModel}
\alias{saveCaretLstmModel}
\title{saveCaretLstmModel}
\usage{
saveCaretLstmModel(model, outFolder, overwrite = FALSE, fullHistory = FALSE)
}
\arguments{
\item{model}{the model as returned by \code{\link[caret]{train}} when 
using \code{\link{getLSTMmodel}}}

\item{outFolder}{name of the folder to store results in.}

\item{overwrite}{If FALSE, function throws an error if outFolder already exists.}

\item{fullHistory}{Boolean. If TRUE, saves all the history, including logs of parameters.
If FALSE, only save bare last model (for storage efficiency)}
}
\value{
No return
}
\description{
saves an LSTM model that was trained by caret.
}
\seealso{
\code{\link{saveLstmModel}}, \code{\link{loadCaretLstmModel}}
}
