#' @title
#'  invertPreProcessor
#' @description
#' Do inverted transformation of stored preProcessing procedure on new data
#' @param preProcessor Stored preProcessing procedure. Result of \code{\link{getPreProcessor}}.
#' @param dat data.frame containing the data to be preprocessed. Must have the correct column names.
#'                    
#' @return data.frame with data transofrmed by inverse preProcessing procedure
#' @export invertPreProcessor
#' @seealso \code{\link{getPreProcessor}}, \code{\link{predictPreProcessor}}
#' 

invertPreProcessor <- function(preProcessor, dat){

  if(is.data.table(dat)) dat <- as.data.frame(dat)
  
  yVariables        <- names(dat)
  names(yVariables) <- names(dat)
  
  if(!is.null(preProcessor$preProcessMinMax)){
    ## revert min max trafo
    dat <- as.data.frame(lapply(X = yVariables,
                                FUN = function(x, dat, preProcessMinMax){
                                  result <- 
                                    dat[,x] * 
                                    (preProcessMinMax$max[x] - preProcessMinMax$min[x]) +
                                    preProcessMinMax$min[x]
                                  return(result)
                                },
                                dat = dat,
                                preProcessMinMax = preProcessor$preProcessMinMax))
  }
  
  if(!is.null(preProcessor$preProcessStep2)){
    ## revert scaleAgain and centerAgain
    
    methods <- setdiff(names(preProcessor$preProcessStep2$method), "ignore")
    
    if(!any(methods == "scale")){
      preProcessor$preProcessStep2$std  <- setNames(rep(1, length(yVariables)), names(yVariables))
    }
    
    if(!any(methods == "center")){
      preProcessor$preProcessStep2$mean <- setNames(rep(0, length(yVariables)), names(yVariables))
    }
    
    dat <- as.data.frame(lapply(X = yVariables,
                                FUN = function(x, dat, preProcessStep){
                                  result <- 
                                    dat[,x] * 
                                    preProcessStep$std[x] +
                                    preProcessStep$mean[x]
                                  return(result)
                                },
                                dat = dat,
                                preProcessStep = preProcessor$preProcessStep2))
  }
  
  if(!is.null(preProcessor$preProcessStep1)){
    ## revert pca, scale, center
    
    methods <- setdiff(names(preProcessor$preProcessStep1$method), "ignore")
    
    if(!all(methods %in% c("scale", "center", "pca"))) {
      stop(sprintf("Unknown preProcessing methods %s", 
                   paste0(setdiff(methods, c("scale", "center", "pca")), 
                          collapse=",")
                   )
           )
    }
    
    if(any(methods == "pca")){
      dat <- as.matrix(dat) %*% t(preProcessor$preProcessStep1$rotation)
      dat <- as.data.frame(dat)
      ## new yVariables!!!
      yVariables <- setNames(names(dat), names(dat))
    }
    
    if(!any(methods == "scale")){
      preProcessor$preProcessStep1$std <- setNames(rep(1, length(yVariables)), names(yVariables))
    }
    if(!any(methods == "center")){
      preProcessor$preProcessStep1$mean <- setNames(rep(0, length(yVariables)), names(yVariables))
    }
    
    dat <- as.data.frame(lapply(X = yVariables,
                                FUN = function(x, dat, preProcessStep){
                                  result <- 
                                    dat[,x] * 
                                    preProcessStep$std[x] +
                                    preProcessStep$mean[x]
                                  return(result)
                                },
                                dat = dat,
                                preProcessStep = preProcessor$preProcessStep1))
    }
  
  return(dat)
}