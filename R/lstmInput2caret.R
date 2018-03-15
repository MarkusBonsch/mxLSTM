#' @title lstmInput2caret
#' @description transforms a list of arrays as returned by \code{\link{transformLSTMinput}} 
#'              into the correct format for training an LSTM model with caret using \code{\link{getLSTMmodel}}
#' @param dat List of arrays as returned by \code{\link{transformLSTMinput}}
#' @return a data.frame as required by caret's \code{\link[caret]{train}} with \code{\link{getLSTMmodel}}.
#'         Each row is one event. The columns contain x and y variables for the full sequence with names:
#'         x1_seq1, x2_seq1, ..., xN_seq1, x1_seq2, ..., xN_seq2, x1_seqM, ..., xN_seqM, y1_seq1Y, y2_seq1Y ..., y1_seqMY, yK_seqMY.
#'         Additionally, y1_seqMY into the 'dummy' variable for the left
#'         hand side of the caret formula and serves as the primary target.
#' @export         
lstmInput2caret <- function(dat){
  
  seq.length <- dim(dat$x)[[2]]
  nEvents    <- dim(dat$x[[3]])
  
  out <- 
    ## transform x to data.table
    CJ(var  = dimnames(dat$x)[[1]],
       elem = paste0("seq", seq_len(seq.length)),
       seq  = seq_len(dim(dat$x)[3]), 
       sorted = FALSE) %>% 
    setkey(seq, elem) %>% 
    .[, val := as.numeric(dat$x)] %>% 
    .[, name := paste0(var, "_", elem)] %>% 
    ## add y
    rbind(
      CJ(var  = dimnames(dat$y)[[1]],
         elem = paste0("seq", seq_len(seq.length)),
         seq  = seq_len(dim(dat$y)[3]), 
         sorted = FALSE) %>% 
        setkey(seq, elem) %>% 
        .[, val := as.numeric(dat$y)] %>% 
        .[, name := paste0(var, "_", elem, "Y")])
  
  ## remember order of names for restoring later
  nameOrder <- unique(out$name)
  
  out <- 
    out %>% 
    ## cast to wide format
    .[, c("var", "elem") := NULL] %>% 
    dcast(seq ~ name, value.var = "val") %>% 
    .[, seq := NULL] %>% 
    ## restore correct order
    setcolorder(nameOrder)
    
  
  out[, dummy := get(paste0(unique(dimnames(dat$y)[[1]])[1], "_seq", seq.length, "Y"))]
  
  return(out)
  
}

