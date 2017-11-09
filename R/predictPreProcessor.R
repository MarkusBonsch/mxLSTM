#' @title
#'  predictPreProcessor
#' @description
#' Perform stored preProcessing procedure on new data
#' @param preProcess Stored preProcessing procedure. Result of \code{\link{getPreProcessor}}.
#' @param dat data.frame containing the data to be preprocessed. Must have the correct column names.
#'                    
#' @return data.frame with preprocessed data
#' @export predictPreProcessor
#' @seealso \code{\link{getPreProcessor}}, \code{\link{invertPreProcessor}}
#' 

predictPreProcessor <- function(preProcess, dat){

  if(is.data.table(dat)) dat <- as.data.frame(dat)
  
  ## if dat contains spurious columns, extract the interesting ones.
  allVars <- unlist(preProcess$preProcessStep1$method[setdiff(names(preProcess$preProcessStep1$method), c("ignore"))])

  if(length(allVars) == 0){ ## no step 1 executed, check for step 2
    allVars <- unlist(preProcess$preProcessStep2$method[setdiff(names(preProcess$preProcessStep2$method), c("ignore"))])
  }

  if(length(allVars) == 0){ ## no step 2 executed, check for step 3
    allVars <- names(preProcess$preProcessMinMax$min)
  }

  if(length(allVars) == 0){ ## no preProcessing executed at all. Take input varnames
    allVars <- names(dat)
  }

  allVars <- unique(allVars)

  dat     <- dat[, allVars, drop = FALSE]

  ## first step: methods offered by caret
  if(!is.null(preProcess$preProcessStep1)){
    dat <- predict(preProcess$preProcessStep1, dat)
  }
  ## second step: scaleAgain and centerAgain
  if(!is.null(preProcess$preProcessStep2)){
    dat <- predict(preProcess$preProcessStep2, dat)
  }
  ## step three: minMax trafo
  if(!is.null(preProcess$preProcessMinMax)){
    variables <- names(dat)
    names(variables) <- names(dat)
    dat <- as.data.frame(lapply(X = variables, 
                                FUN = preProcess$preProcessMinMax$fun, 
                                dat = dat, 
                                min = preProcess$preProcessMinMax$min, 
                                max = preProcess$preProcessMinMax$max))
  }
  
  return(dat)
}