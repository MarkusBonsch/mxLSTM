#' @title caret2lstminput1
#' @description transforms a data.frame as returned by \code{\link{lstmInput2caret}} 
#'              into a data.frame with all measurements of the same variable in one column.
#'              For preProcessing.
#' @param dat data.frame as returned by \code{\link{lstmInput2caret}}
#' @return data.table with one column per variable, one column "eventId", and one column "sequenceId"
#' @importFrom splitstackshape cSplit
caret2lstmInput1 <- function(dat){
  
  dat <- data.table(dat)
  
  out <- 
    dat %>% 
    ## add eventID for remembering event order
    .[, eventId := .I] %>%
    ## remove the dummy variable if it exists
    .[, dummy := NA] %>% 
    .[, dummy := NULL] %>% 
    ## get into long format
    melt(id.vars = "eventId", variable.factor = FALSE) %>% 
    ## split variable column into 2 to get sequence and variable separated
    cSplit(splitCols = "variable", sep = "_", drop = TRUE, type.convert = FALSE) %>% 
    setnames(old = "variable_2", new = "sequenceId") %>% 
    ## get sequence number from split column
    .[, sequenceId := as.integer(sub("^seq([0-9]+)Y?$", "\\1", sequenceId))]
  
  ## remember variable order for restoring after dcast
  nameOrder <- unique(out$variable_1)
  
  out <-
    out %>% 
    ## cast variables to wide again
    dcast(eventId + sequenceId ~ ..., value.var = "value", drop = FALSE, fill = NA) %>% 
    ## get correct torder
    setkey(eventId, sequenceId) %>% 
    ##get correct column order
    setcolorder(c("eventId", "sequenceId", nameOrder))
    
  
  
  return(out)
  
}
