% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/getTrainHistory.character.R
\name{getTrainHistory.character}
\alias{getTrainHistory.character}
\title{getTrainHistory.character}
\usage{
\method{getTrainHistory}{character}(model)
}
\arguments{
\item{model}{the path to a model that has been saved for example with \code{\link{saveLstmModel}}}
}
\value{
data.frame with train and eval performance
}
\description{
returns the logged performance metrics of an stored model object.
}
