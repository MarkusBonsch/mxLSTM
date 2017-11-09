#' @title
#'  predictAll
#' @description
#' Function to predict all y variables from a neuralnet regression model.
#' @param object trained model as returned by \code{\link[caret]{train}}.
#' @param newdata data.frame with new x variables
#' @param ... additional arguments to the \code{predict} function of the model.
#' @return data.frame with the predictions
#' @seealso
#' \code{\link[caret]{predict.train}}
#' @export predictAll
predictAll <- function (object, newdata, ...) {
  predictions <- 
    object$modelInfo$predict(modelFit = object$finalModel,
                             newdata  = newdata,  
                             allY     = TRUE, 
                             ...)
  return(predictions)
}