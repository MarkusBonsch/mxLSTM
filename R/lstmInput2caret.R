#' @title lstmInput2caret
#' @description transforms a list of arrays as returned by \code{\link{transformLSTMinput}} 
#'              into the correct format for training an LSTM model with caret using \code{\link{getLSTMmodel}}
#' @param dat List of arrays as returned by \code{\link{transformLSTMinput}}
#' @return a data.frame as required by caret's \code{\link[caret]{train}} with \code{\link{getLSTMmodel}}.
#'         Each row is one event. The columns contain x and y variables for the full sequence with names:
#'         x1_seq1, x2_seq1, ..., xN_seq1, x1_seq2, ..., xN_seq2, x1_seqM, ..., xN_seqM, y1_seq1Y, ..., y1_seqMY.
#'         Additioanlly, the y-variable of the last sequence element is duplicated into the 'dummy' variable for the left
#'         hand side of the caret formula
#' @export         
lstmInput2caret <- function(dat){
  
  seq.length <- dim(dat$x)[[2]]
  
  extractor <- function(i, dat){
    
    return(as.data.table(t(c(as.numeric(dat$x[,,i]), as.numeric(dat$y[,i])))))
    
  }
  
  out <- 
    lapply(seq_len(dim(dat$x)[[3]]), extractor, dat = dat) %>% 
    rbindlist %>% 
    setnames(c(as.character(outer(dimnames(dat$x)[[1]], paste0("seq", 1: seq.length), paste, sep = "_")),
               as.character(outer(unique(dimnames(dat$y)[[2]]), paste0("seq", 1: seq.length, "Y"), paste, sep = "_")))
    )
  
  out[, dummy := get(paste0(unique(dimnames(dat$y)[[2]]), "_seq", seq.length, "Y"))]
  
  return(out)
  
}

