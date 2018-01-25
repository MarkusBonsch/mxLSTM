#' @title
#'  fitLSTMmodel
#' @description
#' Fits an mxNet LSTM recurrent neural network. For use in getLSTMmodel
#' @param x array containing the x variables for training. See \code{\link{mxLSTM}} for format.
#' @param y array containing the y variables for training. See \code{\link{mxLSTM}} for format.
#' @param test.x same as x, but for testing after each epoch
#' @param test.y same as y, but for testin after each epoch
#' @param param Parameters as given by caret's tuneGrid. See \code{\link{getLSTMmodel}} for details.
#' @param initialModel A pretrained mxLSTM model as returned by this function. Training will continue 
#'                     on initialModel, using its weights as a starting point.
#'                     Be careful, all data shape, hidden layers, and preProcessing must be identical to the 
#'                     first training of initialModel. 
#' @param seed Random seed tp be set before model training. Defaults to NULL for no explicit seed setting
#' @param ... further arguments for the call to \code{\link{mxLSTM}}
#' @return list of class mxLSTM. 
#'        This is the result of \code{\link{mxLSTM}}
#'        Additionally, some other list items are added:
#'        \itemize{
#'        \item{xVariables:}{ Names of the xVariables}
#'        \item{yVariables:}{ Names of the yVariables}
#'        } 
#' @seealso \code{\link{mxLSTM}}, \code{\link{predictLSTMmodel}}, \code{\link{getLSTMmodel}}
#' @export fitLSTMmodel

fitLSTMmodel <- function(x, y, param, test.x = NULL, test.y = NULL, initialModel = NULL,
                         seed = NULL, ...){
    
  if(is.data.table(x)) x <- as.data.frame(x)
  if(is.data.table(y)) y <- as.data.frame(y)
  if(is.data.table(test.x)) test.x <- as.data.frame(test.x)
  if(is.data.table(test.y)) test.y <- as.data.frame(test.y)
  
    xVariables <- dimnames(x)[[1]]
    yVariables <- unique(dimnames(y)[[2]])
    
    ## convert tuning parameters for hidden neurons and dropout to inputs for mxLSTM
    num.hidden  <- c("1" = param$layer1, "2" = param$layer2, "3" = param$layer3)
    dropoutLstm <- c("1" = param$dropout1, "2" = param$dropout2, "3" = param$dropout3)
    ## remove layers with 0 hidden nodes
    dropoutLstm    <- dropoutLstm[num.hidden > 0]
    num.hidden <- num.hidden[num.hidden > 0]
    ## check that no layer has 0 nodes while the subsequent layer has nodes
    if(any(!names(num.hidden) == seq_along(num.hidden))) stop("Implausible hidden layer configuration: layer with 0 neurons precedes layer with >0 neurons.")
    
    if(!is.null(seed)){
      set.seed(seed)
      mx.set.seed(seed)
    }
    
    ## do the real regression
    out <- mxLSTM(x = x, 
                  y = y, 
                  num.epoch    = param$num.epoch, 
                  test.x       = test.x, 
                  test.y       = test.y, 
                  num.hidden   = num.hidden, 
                  dropoutLstm  = dropoutLstm, 
                  batch.size   = param$batch.size, 
                  activation   = param$activation, 
                  optimizer    = "rmsprop", 
                  initializer  = mx.init.Xavier(), 
                  shuffle      = param$shuffle,
                  initialModel = initialModel,
                  ## optimizer arguments
                  wd           = param$weight.decay,
                  gamma1       = param$learningrate.momentum,
                  gamma2       = param$momentum,
                  ...)

    out$xVariables <- xVariables
    out$yVariables <- yVariables
    
    return(out)
}