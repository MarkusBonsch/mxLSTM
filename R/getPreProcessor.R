#' @title
#'  getPreProcessor
#' @description
#' Preprocessing for regression in caret
#' @param dat data.frame containing the data to be preprocessed.
#' @param preProcess  Character vector with preprocessing steps for dat. 
#'                    Can be anything that caret allows 
#'                    (see \code{\link[caret]{preProcess}}) 
#'                    plus \code{c('scaleAgain', 'centerAgain', 'minMax')}.
#'                    \code{scaleAgain} and \code{centerAgain} are used to rescale 
#'                    data after a PCA or a similar transformation has 
#'                    destroyed the original scaling by caret.
#'                    minMax transforms the output into [0;1]. 
#'                    minMax is always executed as the last step.
#' @param preProcOptions list of options for the different preProcessing steps from caret. 
#'                    
#' @return List with the following items:
#'         \itemize{
#'           \item{preProcessStep1:}{ preProcessing result for dat, 
#'           first step (all native caret methods). See \code{\link[caret]{preProcess}}}
#'           \item{preProcessStep2:}{ preProcessing result for dat for \code{scaleAgain, centerAgain}}
#'                                Result of caret preProcess with scale and center
#'           \item{preProcessMinMax:}{ preProcessing result for minMax trafo. 
#'                                   Comprises min value, max value and trafo function.}
#'         }
#' 
#' @importFrom caret preProcess
#' @export getPreProcessor
#' @seealso \code{\link{predictPreProcessor}}, \code{\link{invertPreProcessor}}
#' 

getPreProcessor <- function(dat, preProcess,
                                   preProcOptions = list(thresh = 0.95, ICAcomp = 3, k = 5, freqCut = 95/5,
                                                         uniqueCut = 10, cutoff = 0.9)){

    if(is.data.table(dat)) dat <- as.data.frame(dat)
    
    ## define function for minMax Trafo for use in lapply
    minMaxTrafo <- function(x, dat, min, max){return((dat[,x] - min[x])/(max[x] - min[x]))}  

    ## first step: methods offered by caret
    preProcessStep1 <- setdiff(preProcess, c("scaleAgain", "centerAgain", "minMax"))
    if(length(preProcessStep1 > 0)){
      preProcessStep1 <- c(list(x = dat, method = preProcessStep1), preProcOptions)
      preProcessStep1 <- do.call(caret::preProcess, preProcessStep1)
      dat             <- predict(preProcessStep1, dat)
    } else {
        preProcessStep1 <- NULL
      }
      
    ## second step: scaleAgain, centerAgain
    preProcessStep2 <- preProcess[which(preProcess %in% c("scaleAgain", "centerAgain"))]
    if(length(preProcessStep2 > 0)){
      preProcessStep2 <- list(x = dat, method = sub("Again", "", preProcessStep2))
      preProcessStep2 <- do.call(caret::preProcess, preProcessStep2)
    } else {
      preProcessStep2 <- NULL
    }
    
    ## third step minMaxTrafo
    preProcessMinMax <- list()
    if("minMax" %in% preProcess){
      
      preProcessMinMax$min <- unlist(lapply(dat, min, na.rm = TRUE))
      preProcessMinMax$max <- unlist(lapply(dat, max, na.rm = TRUE))
      preProcessMinMax$fun <- minMaxTrafo
    } else {
      preProcessMinMax <- NULL
    }
    
      out <- list()
      out$preProcessStep1 <- preProcessStep1
      out$preProcessStep2 <- preProcessStep2
      out$preProcessMinMax <- preProcessMinMax
      return(out)
    }