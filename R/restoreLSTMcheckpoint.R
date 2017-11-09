#' @title restoreLstmCheckpoint
#' @description  function to restore logged checkpoint from mxLSTM model
#' @param model the model as returned by \code{\link{fitLSTMmodel}} or \code{\link{mxLSTM}}. 
#'              Is returned as model$finalModel by \code{\link[caret]{train}}
#' @param checkpointNumber The number of the checkpoint to restore. 
#'                         Investigate the training history with 
#'                         \code{\link{plot_trainHistory}} to choose a checkpoint.
#' @return the updated model. Parameters are set to the checkpoint and 
#'         model$log$activeCheckpoint is updated.
#' @seealso \code{\link{plot_trainHistory}}
#' @export restoreLstmCheckpoint
restoreLstmCheckpoint <- function(model, checkpointNumber = NULL){

  if(is.null(checkpointNumber)){
    cat("Updating to last available checkpoint\n")
    checkpointNumber <- length(model$log$checkpoint)
  }
  if(checkpointNumber > length(model$log$checkpoint)){
    stop(sprintf("Checkpoint %s is not available. Maximum checkpoint number is %s", 
                 checkpointNumber, length(model$log$checkpoint)))
  }
  
  ## update the weights based on checkpoint.
  model$arg.params <- lapply(model$log$checkpoint[[checkpointNumber]][["arg.params"]], mx.nd.array)
  model$aux.params <- lapply(model$log$checkpoint[[checkpointNumber]][["aux.params"]], mx.nd.array)

  ## update the active checkpoint information
  model$log$activeCheckpoint <- checkpointNumber
  
  return(model)
}