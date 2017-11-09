#' @title saveLstmModel
#' @description saves an LSTM model that was trained using \code{\link{mxLSTM}}.
#' @param model the model as returned from \code{\link{mxLSTM}}
#' @param outFolder name of the folder to store results in.
#' @param overwrite If FALSE, function throws an error if outFolder already exists.
#' @param fullHistory Boolean. If TRUE, saves all the history, including logs of parameters.
#'                    If FALSE, only save bare last model (for storage efficiency)
#' @return No return
#' @seealso \code{\link{loadLstmModel}}
#' @importFrom mxnet mx.symbol.save mx.nd.save
#' @export saveLstmModel
#' 
saveLstmModel <- function(model, outFolder, overwrite = FALSE, fullHistory = TRUE){
  
  if(dir.exists(outFolder)){
    if(overwrite == FALSE) stop("output directory not empty.")
  } else {
    dir.create(outFolder)
  }
  baseFolder <- getwd()
  setwd(outFolder)
  on.exit(setwd(baseFolder))
  
  mx.symbol.save(model$symbol, "symbol")
  
  ## save parameters
  mx.nd.save(ndarray = model$arg.params, filename = "argParams")
  
  if(length(model$aux.params) > 0){
    mx.nd.save(ndarray = model$aux.params, filename = "auxParams")
  }
  
  ## save the rest of the model. If fullHistory == FALSE, set the checkpoints to NULL for storage efficiency
  if(!fullHistory){
    model$log$checkpoint <- NULL
  }
  saveRDS(model[setdiff(names(model), c("symbol", "arg.params", "aux.params"))], file = "rest.rds")

  ## create a file that contains the modelClass
  cat("mxLSTM", file = "modelType.txt")
  
  return(invisible(NULL))
}