#' @title loadLstmModel
#' @description loads an LSTM mdoel that was saved using \code{\link{saveLstmModel}}.
#' @param folder The folder where the model was stored.
#' @return mxLSTM model
#' @seealso \code{\link{saveLstmModel}}
#' @importFrom mxnet mx.symbol.load mx.nd.load
#' @export loadLstmModel
#' 
loadLstmModel <- function(folder){
  
  if(!dir.exists(folder)) stop("input folder not found")
  baseFolder <- getwd()
  setwd(folder)
  on.exit(setwd(baseFolder))
  
  if(!file.exists("modelType.txt")) stop("Can't load, modelType.txt not found.")
  if(type <- readLines("modelType.txt", warn = FALSE) != "mxLSTM") stop("loadLstmModel cant load model of type ", type)

  ## load the base list with log etc
  model <- readRDS("rest.rds")
  
  ## load the symbol
  model$symbol <- mx.symbol.load("symbol")
  
  ## load the  parameters
  model$arg.params <- mx.nd.load("argParams")
  
  if(file.exists("auxParams")){
    model$aux.params <- mx.nd.load("auxParams")
  }

  return(structure(model, class = "mxLSTM"))
}