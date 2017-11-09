#' @title caret2lstminput
#' @description transforms a data.frame as returned by \code{\link{lstmInput2caret}} into a list of arrays as required by \code{\link{mxLSTM}} 
#' @param dat data.frame as returned by \code{\link{lstmInput2caret}}
#' @return a list of arrays as required by \code{\link{fitLSTMmodel}} and \code{\link{mxLSTM}}
#'         Same structure as the return value of \code{\link{transformLSTMinput}}
#' @importFrom splitstackshape cSplit
#' @export         
caret2lstmInput <- function(dat){
  
  dat <- copy(data.table(dat))
  
  ## infer variable names and sequence length from columnnames
  yVariables <- 
    names(dat) %>% 
    grep(pattern = "_seq[0-9]+Y", value = TRUE) %>% 
    sub(pattern = "_seq[0-9]+Y", replacement = "") %>% 
    unique
  
  seq.length <- 
    setdiff(names(dat), "dummy") %>% 
    sub(pattern = "^.*seq([0-9]+)Y?$", replacement = "\\1") %>% 
    as.numeric() %>% 
    max
  
  
  out <- 
    dat %>% 
    caret2lstmInput1 %>% 
    ## discard ordering columns
    .[, c("eventId", "sequenceId") := NULL] %>% 
    ## use my transformation function to get lstm input
    transformLSTMinput(targetColumn = yVariables, seq.length = seq.length)
  
  return(out)
  
}
