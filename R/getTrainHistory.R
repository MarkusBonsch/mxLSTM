#' @title getTrainHistory
#' @description gets the logged training performance of models
#' @param model the model
#' @return data.frame with two columns: train and test with 
#'         performances on training and eval sets
#' @export getTrainHistory
#' 
getTrainHistory <- function(model){
  
  UseMethod("getTrainHistory", object = model)

}