#' @title predictLSTMmodel
#' @description get predictions for an LSTM recurrent network
#' @param model LSTM model as returned by \code{\link{mxLSTM}} or \code{\link{fitLSTMmodel}}
#' @param dat input data as provided by \code{\link{transformLSTMinput}} in the 'x' element of the list.
#' @param fullSequence Boolean. If FALSE, only the last predicted element of a sequence is returned.
#'                              If TRUE, a prediction for each step in the sequence is returned.            
#' @return data.frame with predictions.
#' @details the sequence length is inferred from the \code{model} argument.
#' @seealso \code{\link{mxLSTM}}, \code{\link{fitLSTMmodel}}, \code{\link{getLSTMmodel}}
#' @export 
predictLSTMmodel <- function(model, dat, fullSequence = TRUE){
  
  if(!"mxLSTM" %in% class(model)) stop("'model' must be an mxLSTM object")
  
  if(!all(model$varNames$x %in% dimnames(dat)[[1]])) {
    stop("Wrong variables in input data for prediction")
  }
  
  seq.len    <- dim(model$arg.params$data)[2]
  
  if(dim(dat)[2] != seq.len) stop("Prediction data has a wrong sequence length")
  
  ## create executor from input symbol and parameters
  exec <- mxnet:::mx.symbol.bind(symbol     = model$symbol,     ctx        = mx.cpu(), 
                                 arg.arrays = model$arg.params, aux.arrays = model$aux.params, 
                                 grad.reqs  = rep("null", length(model$arg.params)) ## no gradients needed in testing
  )
  
  ## create init state arrays with all 0s for clearing after each batch
  init.states.name <- grep(".*\\.[ch]$", names(model$arg.params), value = TRUE)
  
  init.states.cleared <- 
    lapply(model$arg.params[init.states.name], function(x) return(x * 0))
  
  ## get correct order of variables
  dat <- dat[model$varNames$x,,]
  
  batch.size <- dim(model$arg.params$data)[3]  
  
  ## create dummy y variables as placeholder
  y <- array(0, dim = dim(dat))

  ## create the iterator over batches
  input <- mx.io.arrayiter(data    = dat, 
                           label   = y, 
                           batch.size = batch.size, 
                           shuffle = FALSE)
  
  input$reset()
  if (!input$iter.next()) stop("Cannot predict on empty iterator")
  
  input$reset()
  
  ## result container
  packer <- mxnet:::mx.nd.arraypacker()
  
  while (input$iter.next()) {
    
    ## clear initial states
    mx.exec.update.arg.arrays(exec, init.states.cleared, match.name = TRUE)
    
    ## set inputs
    mx.exec.update.arg.arrays(exec, list(data = input$value()$data), 
                              match.name = TRUE)
    
    ## calculate outputs
    mx.exec.forward(exec, is.train = FALSE)
    
    out.pred <- exec$ref.outputs[[1]]
    
    
    ## reorder
    
    padded <- input$num.pad()
    
    if(fullSequence){
      ## reorder the output so that it is elem1[seq1], elem1[seq2], ..., elem1[seqN], elem2[seq1],,,
      ## that makes it time-ordered, as the original label should be
      ## if the last batch is not fully filled, outputs from previous batch are repeated..
      ## num.pad indicates, how many elements in the batch are padded from the back.
      ## remove those from the output. Be careful: the output is ordered as follows: 
      ## elem1seq1, elem2seq1, ..., elemNseq1, ..., elemNseq2, ..., elemNseqN
      timeOrderIndices <- integer(0)
      for(s in seq_len(batch.size - padded)){
        timeOrderIndices <-
          c(timeOrderIndices,
            seq(s - 1,
                seq.len * batch.size- 1,
                by = batch.size
                )
          )
      }
  
      timeOrderIndices <- mx.nd.array(timeOrderIndices)

      out.pred <- mx.nd.take(a = out.pred, indices = timeOrderIndices, mode = "clip") # mode = "raise" would be preferred but does not work anymore.
      
    } else { # fullSequence == FALSE
      
      ## if fullSequence is FALSE, select the last element of each sequence
      lastElementIndices <- 
        (seq.len - 1) * batch.size + (0 : (batch.size - padded - 1)) %>% 
        mx.nd.array()
      
      out.pred <- mx.nd.take(a = out.pred, indices = lastElementIndices, mode = "clip") # mode = "raise" would be preferred but does not work anymore.
    
    }
    
    packer$push(out.pred)
    
  }
  
  input$reset()
  
  out <- 
    packer$get() %>% 
    t %>% 
    data.table %>% 
    setnames(if(length(.) > 1) paste0("y", seq_along(.)) else "y")
  
  
  ## get an index of row numbers
  if(fullSequence){
    index <- seq_len(nrow(out))
  } else {
    index <- seq(from = seq.len, by = seq.len, length.out = nrow(out))
  }
  
  out[, rowIndex := index]
  
  return(as.data.frame(out))
}
