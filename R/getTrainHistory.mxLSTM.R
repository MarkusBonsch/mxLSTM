#' @title getTrainHistory.mxLSTM
#' @description returns the logged performance metrics of an mxLSTM object.
#' @param model the model as returned by \code{\link{fitLSTMmodel}} or \code{\link{mxLSTM}}
#' @return data.frame with train and eval performance
#' @export getTrainHistory.mxLSTM
#' 
getTrainHistory.mxLSTM <- function(model){
  
  ## extract data
  dat <- data.frame(train = model$log$train)
  if(length(model$log$eval) > 0){
    dat$eval <- model$log$eval
  }
 
  return(dat)
}