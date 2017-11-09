#' @title transformLSTMinput
#' @description transforms a data.frame input into the correct format for the LSTM model
#' @param dat data.frame with variables. Is assumed to be a time-series without gaps. 
#'            Each column corresponds to one variable
#' @param targetColumn Name of the column that represents the regression target. 
#'                     All other columns are considered as inputs
#' @param seq.length sequence length. 
#' @param seq.freq frequency of sequence starts. Defaults to sequence length.
#' @importFrom abind abind
#' @import data.table
#' @return a list of two: 'x' is an array with input variables as required for \code{\link{mxLSTM}}.
#'                        'y' is an array of target values as required for \code{\link{mxLSTM}}.
#' @details The last sequence is discarded if it is not complete. 
#'          Sequences containing missing values are discarded.
#' @export         
transformLSTMinput <- function(dat, targetColumn, seq.length, seq.freq = seq.length){
  
  if(length(targetColumn) != 1) stop("Only single column target data supported")
  dat <- data.table(dat)
  
  xVariables <- setdiff(names(dat), targetColumn)
  
  
  ## add sequence identifier column
  if(seq.freq == seq.length){
    
    ## very simple. Second sequence starts when first has ended and so on
    dat[, sequence := (.I - 1) %/% seq.length]
    
  } else if (seq.freq > seq.length){
    
    ## there are gaps between sequences. Add full sequence first, then remove spurious rows
    dat[, sequence := (.I - 1) %/% seq.freq]
    gapId <- outer((seq.length + 1) : seq.freq, 0:(nrow(dat) %/% seq.freq) * seq.freq, FUN = "+") %>% as.numeric()
    dat <- dat[setdiff(seq_len(.N), gapId), ]
    
  } else { ## seq.freq < seq.length
    
    ## sequences are overlapping. We add several iterations of non-overlapping sequences 
    ## to an output dataset
    datRaw <- copy(dat)
    datRaw[, order := .I] ## remember original order
    
    ## determine how many sequences overlap
    numOverlap <- ceiling(seq.length / seq.freq)
    ## new seq.freq for the individual series
    seq.freq2 <- numOverlap * seq.freq
    
    dat <- data.table()
    for(i in seq_len(numOverlap)){
      
      ## remove leading part that is not interesting for this series
      thisDat <- datRaw[(((i - 1) * (seq.freq)) + 1) : nrow(datRaw)]
      ## add the sequence based on new seq.freq
      thisDat[, sequence := (((.I - 1) %/% (seq.freq2))) * (numOverlap) + i]
      ## remove rows that are too much
      if(seq.freq2 > seq.length){
        gapId <- outer((seq.length + 1) : seq.freq2, 0:(nrow(thisDat) %/% seq.freq2) * seq.freq2, FUN = "+") %>% as.numeric()
        thisDat <- thisDat[setdiff(seq_len(.N), gapId), ]
      }
      
      dat <- rbind(dat, thisDat)
    }
    
    setkey(dat, sequence, order)
    dat[, order := NULL]
    
  }
  
  ## discard sequences that are not complete
  dat[, completeSequence := .N == seq.length, by = "sequence"]
  
  dat <- dat[completeSequence == TRUE]
  dat[, completeSequence := NULL]
  
  ## convert inputs to 3d array
  x <- split(dat[, -c("sequence", targetColumn), with = FALSE], dat$sequence) %>% 
    ## transpose to get features into 1st dimension
    lapply(t) %>% 
    ## bind everything together according to the correct dimension
    abind(along = 3) 
  
  ## convert target to 2d array
  ## convert inputs to 3d array
  y <- split(dat[, targetColumn, with = FALSE], dat$sequence) %>% 
    ## bind everything together according to the correct dimension
    abind(along = 2)
  
  ## discard sequences that contain NAs
  naData <- colSums(is.na(x), dims = 2) > 0 | colSums(is.na(y), dims = 1) > 0
  
  x <- x[,, !naData, drop = FALSE]
  y <- y[, !naData, drop = FALSE]
  
  return(list(x = x, y = y))
  
}
