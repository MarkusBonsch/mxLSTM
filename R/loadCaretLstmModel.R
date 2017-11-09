#' @title loadCaretLstmModel
#' @description loads an LSTM model that was trained by caret and saved using \code{\link{saveCaretLstmModel}}.
#' @param folder The folder where the model was stored.
#' @return caret training output with finalModel of class mxLSTM model
#' @seealso \code{\link{loadLstmModel}}, \code{\link{saveCaretLstmModel}}
#' @export loadCaretLstmModel
#' 
loadCaretLstmModel <- function(folder){
  
  if(!dir.exists(folder)) stop("input folder not found")
  baseFolder <- getwd()
  setwd(folder)
  on.exit(setwd(baseFolder))
  
  ## load the caret stuff
  model <- readRDS("caret.rds")
  
  ## load the real model
  model$finalModel <- loadLstmModel(folder = ".")
  
  return(model)
}