#' @title getTrainHistory.character
#' @description returns the logged performance metrics of an stored model object.
#' @param model the path to a model that has been saved for example with \code{\link{saveLstmModel}}
#' @return data.frame with train and eval performance
#' @export getTrainHistory.character
#' 
getTrainHistory.character <- function(model){
  
  dir <- model
  if(!dir.exists(dir)) stop("Input is not a valid directory")
  
  if(!file.exists(file.path(dir, "modelType.txt"))){
      stop("Input directory is not a valid model directory")
  }
  
  modelType <- readLines(file.path(dir, "modelType.txt"))
  
  if(modelType == "mxLSTM"){
    loader <- loadLstmModel
  } else {
    stop("modelType not supported")
  }
                         
  model <- loader(dir)

  return(getTrainHistory(model))
}