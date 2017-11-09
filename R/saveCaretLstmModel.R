#' @title saveCaretLstmModel
#' @description saves an LSTM model that was trained by caret.
#' @param model the model as returned by \code{\link[caret]{train}} when 
#'              using \code{\link{getLSTMmodel}}
#' @param outFolder name of the folder to store results in.
#' @param overwrite If FALSE, function throws an error if outFolder already exists.
#' @param fullHistory Boolean. If TRUE, saves all the history, including logs of parameters.
#'                    If FALSE, only save bare last model (for storage efficiency)
#' @return No return
#' @seealso \code{\link{saveLstmModel}}, \code{\link{loadCaretLstmModel}}
#' @export saveCaretLstmModel
#' 
saveCaretLstmModel <- function(model, outFolder, overwrite = FALSE, fullHistory = FALSE){
  
  if(dir.exists(outFolder)){
    if(overwrite == FALSE) stop("output directory not empty.")
  } else {
    dir.create(outFolder)
  }
  baseFolder <- getwd()
  setwd(outFolder)
  on.exit(setwd(baseFolder))
  
  ## save the real model
  saveLstmModel(model     = model$finalModel, outFolder   = ".", 
                overwrite = TRUE,            fullHistory = fullHistory)
  
  ## save the caret stuff
  model$finalModel <- NULL
  saveRDS(model, file = "caret.rds")
  
  return(invisible(NULL))
}