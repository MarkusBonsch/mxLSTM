#' @title mx.callback.log
#' @description custom logger function that is executed after each 
#'              epoch of mxnet training.
#' @param period interval of logger execution in epochs.
#' @param loggerEnv environment for storing the logger results
#' @return The logger will create a list named 'logger' in loggerEnv 
#'         with the following items:
#'         \itemize{
#'           \item{train} {numeric vector with training metric evaluations}
#'           \item{eval}  {numeric vector with test metric evaluations}
#'           \item{checkpoint} {List of model weights for restoring the 
#'                              state after the specified epoch. 
#'                              See ?getMxNetModel on how to restore checkpoint}
#'          }
#' @seealso \code{\link{plot_trainHistory}}
#' @export mx.callback.log

mx.callback.log <- function (period, loggerEnv) {
  
  function(iteration, nbatch, env, verbose = TRUE) {
    if (nbatch%%period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (!"logger" %in% ls(envir = loggerEnv)) {
        loggerEnv$logger <- list(train = numeric(0), 
                                 eval = numeric(0), checkpoint = list()) 
      }
      ## add train metric
      loggerEnv$logger$train <- c(loggerEnv$logger$train, result$value)
      if (!is.null(env$eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        ## add eval metric
        loggerEnv$logger$eval <- c(loggerEnv$logger$eval, result$value)
      }
      ## add arg.params and aux.params
      arg.arrays <- env$model$arg.arrays
      aux.arrays <- env$model$aux.arrays

      test <<- function()return(NULL)
      loggerEnv$logger$checkpoint[[length(loggerEnv$logger$checkpoint) + 1]] <- 
        list(arg.params = lapply(arg.arrays, as.array),
             aux.params = lapply(aux.arrays, as.array))
      
    }
    return(TRUE)
  }
}