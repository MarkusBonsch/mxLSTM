#' @title mxLSTM
#' @description Builds an LSTM model
#' @param x array containing the features:
#'          \itemize{
#'            \item{Dimension 1:} one entry for each feature
#'            \item{Dimension 2:} one entry for each element in the sequence
#'            \item{Dimension 3:} one entry for each training event
#'            }
#'            Use \code{\link{transformLSTMinput}} to transform data.frames into this structure.
#' @param y array containing the target labels:
#'          \itemize{
#'            \item{Dimension 1:} one entry for each output variable
#'            \item{Dimension 2:} one entry for each element in the sequence
#'            \item{Dimension 3:} one entry for each training event
#'            }
#'            Use \code{\link{transformLSTMinput}} to transform data.frames into this structure.
#' @param test.x same as x, but for testing, not for training
#' @param test.y same as y but for testing, not for training
#' @param num.epoch integer number of training epochs over full ldataset
#' @param num.hidden integer vector of flexible length. For each entry, an LSTM layer is with the corresponding number of
#'                   neurons is created.
#' @param dropoutLstm    numeric vector of same length as num.hidden. Specifies the dropout probability for each LSTM layer.
#'                       Dropout is applied according to Cheng et al. "An exploration of dropout with LSTMs". 
#'                       Difference: we employ a constant dropout rate; we do per element dropout.
#' @param zoneoutLstm    numeric vector of same length as num.hidden. Specifies the zoneout probability for each LSTM layer.
#'                       Zoneout is implemented according to 
#'                       Krueger et al. 2017 "Zoneout: Regularizing RNNs by randomly preserving hidden activations". 
#'                       
#' @param batchNormLstm  logical. If TRUE, each LSTM layer is batch normalized according to the recommendations in
#'                      T. Cooljmans et al. ILRC 2017 "Recurrent batch normalization".
#' @param gammaInit numeric value. Will be used to initialize the gamma matrices of batchNormLayers. 
#'                  Cooljmans et al. recommend 0.1 (for use with tanh activation), mxnet default is 1.
#'                  My experience: 0.1 works very badly with relu activation.
#' @param batch.size self explanatory
#' @param activation activation function for update layers in the LSTM cells. "relu" or "tanh"
#' @param optimizer character specifying the type of optimizer to use.
#' @param initializer random initializer for weights
#' @param shuffle    Boolean. Should the training data be reordered randomly prior to training? 
#'                   (reorders full sequences, order within each sequence is unaffected.)
#' @param initialModel mxLSTM model. If provided, all weights are initialized based on the given model.
#' @param ... Additional arguments to optimizer
#' @return object of class mxLSTM: list: a symbol, arg.params, aux.params, a log, and the variable names
#' @details sequence length is inferred from input (dimension 2).
#' @seealso \code{\link{fitLSTMmodel}}, \code{\link{predictLSTMmodel}}, \code{\link{getLSTMmodel}},
#'          \code{\link{plot_trainHistory}}
#' @import mxnet
#' @export mxLSTM
#' @examples 
#'\dontrun{
#' library(mxLSTM)
#' library(data.table)
#' 
#' ## simple data: one numeric output as a function of two numeric inputs.
#' ## including lag values
#' ## with some noise.
#' dat <- data.table(x = runif(n = 8000, min = 1000, max = 2000),
#'                   y = runif(n = 8000, min = -10, max = 10))
#' ## create target
#' dat[, target := 0.5 * x + 0.7 * lag(y, 3) - 0.2 * lag(x, 5)]
#' dat[, target := target + rnorm(8000, 0, 10)]
#' ## convert to nxLSTM input
#' dat <- transformLSTMinput(dat = dat, targetColumn = "target", seq.length = 5)
#' 
#' ## train model
#' model <- mxLSTM(x = dat$x, y = dat$y, num.epoch = 10, num.hidden = 64, 
#'                 dropoutLstm = 0, zoneoutLstm = 0, batchNormLstm = FALSE, batch.size = 128)
#' 
#' ## plot training history
#' plot_trainHistory(model)
#' 
#' ## get some predictions (on training set)
#' predTrain <- predictLSTMmodel(model = model, dat = dat$x, fullSequence = FALSE)
#' 
#' ## nice plot
#' plot_goodnessOfFit(predicted = predTrain$y, observed = dat$y[5,])
#' }

mxLSTM <- function(x, y, num.epoch, test.x = NULL, test.y = NULL, num.hidden, dropoutLstm = num.hidden * 0,
                   zoneoutLstm = num.hidden * 0, batchNormLstm = FALSE, gammaInit = 0.1, batch.size = 128, activation = "relu", optimizer = "rmsprop", 
                   initializer = mx.init.Xavier(), shuffle = TRUE, initialModel = NULL, ...){
  
  
  if(!all(dim(x)[2:3] == dim(y))) stop("x and y don't fit together.")
  if(any(dropoutLstm * zoneoutLstm != 0)) stop("dropout and zoneout are mutually exclusive. Please adapt arguments 'dropoutLSTM' or 'zoneoutLSTM")
  
  seq.length <- dim(x)[2]
  
  datTrain <- list(data  = x,
                   label = y)
  
  if(!is.null(test.x)){
    
    datEval <- list(data  = test.x,
                    label = test.y)
    
  } else {
    
    datEval <- NULL
    
  }
  
  ###############################################################
  ## construct model
  
  ## raw symbol
  model <- mxLSTMcreate(seq.length = seq.length, 
                        num.hidden = num.hidden, 
                        dropoutLstm   = dropoutLstm, 
                        zoneoutLstm   = zoneoutLstm,
                        batchNormLstm = batchNormLstm,
                        batch.size = batch.size, 
                        activation = activation)
  
  ################################################################
  ## train model
  thisLoggerEnv <- new.env()

  ## remember the log of inital model if existent
  if(!is.null(initialModel)){
    thisLoggerEnv$logger <- initialModel$log
  }

  model <- mxLSTMtrain(datTrain   = datTrain,
                       datEval    = datEval, 
                       symbol     = model, 
                       batchSize  = batch.size,
                       num.hidden = num.hidden,
                       num.rounds = num.epoch, 
                       optimizer  = optimizer, 
                       initializer =initializer,
                       initialModel=initialModel,
                       shuffle    = shuffle, 
                       gammaInit  = gammaInit,
                       epoch.end.callback = mx.callback.log(period = 1, loggerEnv = thisLoggerEnv),
                       ...)
  
  ## store the active checkpoint in the logger. By default, it is the last iteration.
  thisLoggerEnv$logger$activeCheckpoint <- length(thisLoggerEnv$logger$train)
  
  return(structure(list(symbol = model$symbol,
                        arg.params = model$arg.params,
                        aux.params = model$aux.params,
                        log   = thisLoggerEnv$logger,
                        varNames = list(x = dimnames(x)[[1]],          ## remember variable names 
                                        y = unique(dimnames(y)[[2]]))) ## to order input at prediction
                   , class = "mxLSTM"))
  
}



#' @title mxLSTMcreate
#' @description Creates the basic network.
#'  consists only of symbols, no binding to values yet.
#' @param seq.length see \code{\link{mxLSTM}}
#' @param num.hidden see \code{\link{mxLSTM}}
#' @param dropoutLstm see \code{\link{mxLSTM}}
#' @param zoneoutLstm see \code{\link{mxLSTM}}
#' @param batchNormLstm see \code{\link{mxLSTM}}
#' @param batch.size see \code{\link{mxLSTM}}
#' @param activation see \code{\link{mxLSTM}}
#' @return MXSymbol 

mxLSTMcreate <- function(seq.length, num.hidden, dropoutLstm = 0, zoneoutLstm = 0, batchNormLstm = FALSE, 
                         batch.size = 128, activation = "relu"){
  
  if(any(num.hidden <=0)) stop("num.hidden must consist of positive numbers")
  
  if(any(dropoutLstm < 0 | dropoutLstm >=1)) stop("dropout must be in [0;1)")
  
  if(any(zoneoutLstm < 0 | zoneoutLstm >=1)) stop("zoneout must be in [0;1)")
  
  if(length(num.hidden) != length(dropoutLstm) |
     length(num.hidden) != length(zoneoutLstm)){
    stop("num.hidden, zoneout, and dropout must be vectors of identical length")
  }

  if(any(dropoutLstm * zoneoutLstm != 0)) stop("dropout and zoneout are mutually exclusive. Please adapt arguments 'dropoutLSTM' or 'zoneoutLSTM")
  
  num.lstm.layer <- length(num.hidden)
  
  ## create input data and target (label)
  
  ## input dimensions: 1 = features, 2 = sequence, 3 = batch-size
  ## input is sliced so that always one element from a sequence is accessible
  data <- mx.symbol.Variable('data')
  
  data <- mx.symbol.SliceChannel(data, 
                                 num_outputs = seq.length, 
                                 axis = 1, ## 1 = slice on rows. Rows correspond to sequence
                                 squeeze_axis = TRUE, ## drop the squuezed axis
                                 name = "data")
  
  ## target dimensions: 1 = sequence, 2 = batch-size
  ## target is reshpaed so that targets of all 
  ## sequences are concatenated to result in a one dimensional matrix with length seq.length * batch.size
  ## The order of element is as follows: batch1seq1, batch2seq1, ..., batch[batch.size]seq1, batch1seq2, ...
  label <- mx.symbol.Variable("label")
  label <- mx.symbol.transpose(data = label)
  label <- mx.symbol.Reshape(data = label, shape = batch.size * seq.length)
  
  ## create symbol variables for the memory
  
  ## init.c and init.h for each lstm layer, 
  ## wrapped up in list param.cells
  ## (will contain state (c) and output(h) values from last sequence element)
  
  last.states <- list()
  for(l in seq_len(num.lstm.layer)){
    last.states[[l]] <- list()
    last.states[[l]]$h <- mx.symbol.Variable(name = (paste0("l", l , ".h")))
    last.states[[l]]$c <- mx.symbol.Variable(name = (paste0("l", l , ".c")))
  }
  
  ## memory for weights and biases. 
  ## wrapped up in list param.cells
  param.cells <- list()
  for(l in seq_len(num.lstm.layer)){
    param.cells[[l]] <- list()
    param.cells[[l]]$i2h.weight <- mx.symbol.Variable(name = (paste0("l", l , ".i2h.weight")))
    param.cells[[l]]$h2h.weight <- mx.symbol.Variable(name = (paste0("l", l , ".h2h.weight")))
    param.cells[[l]]$i2h.bias <- mx.symbol.Variable(name = (paste0("l", l , ".i2h.bias")))
    param.cells[[l]]$h2h.bias <- mx.symbol.Variable(name = (paste0("l", l , ".h2h.bias")))
  }
  
  ## memory for the models of each sequence element
  sequenceModels <- list()
  
  ## create the LSTM layer(s).
  ## in each layer, there is one cell for each sequence element. 
  ## But weights are shared across all of them and a cell 
  ## will always receive the state and output of the previous one.
  
  for(elem in seq_len(seq.length)){
    ## each sequence element gets its own model. 
    ## These individaul models share weights and are concatenated in the end.
    ## each model starts with the input data:
    elemModel <- data[[elem]]
    
    for(layer in seq_len(num.lstm.layer)){
      ## create the lstm cell. The return return value is a list with the state (c) and output(h) as symbols
      this.cell <- lstmCell(num.hidden = num.hidden[layer], ## user argument
                            indata = elemModel, ## whatever has been modelled before (the computation graph)
                            prev.state = last.states[[layer]], ## state from previous sequence element
                            param = param.cells[[layer]], ## weights (shared across sequences)
                            seqidx = elem, ## only for bookkeeping
                            layeridx = layer, ## only for bookkeeping
                            dropout = dropoutLstm[layer],
                            zoneout = zoneoutLstm[layer],
                            batchNorm = batchNormLstm,
                            activation = activation)
      
      ## the new model is the output of the lstm cell
      elemModel <- this.cell$h
      ## remember the state and output for passing to next sequence element
      last.states[[layer]] <- this.cell
    }
    
    ## after all lstm cells are created for our sequence model,
    ## add the model to the list of models for all sequence elements
    sequenceModels <- c(sequenceModels, elemModel)
  }
  
  ## the dimension of each sequence element model is:
  ## 1 = num.hidden, 2 = batch size.
  ## There are seq.length elements in the list.
  ## Now they are concatenated so that the output is two dimensional:
  ## 1 = num.hidden, 2 = sequence element (seq.length * batch.size elements).
  ## In dimension 2, the order of element is as follows: batch1seq1, batch2seq1, ..., batch[batch.size]seq1, batch1seq2, ...
  model <- mx.symbol.concat(data = sequenceModels, 
                            num.args = seq.length,
                            dim  = 0, ## mxnet counts dimensions from the back. 0 means: increase the number of columns
                            name = "model")
  
  ## create fully connected output layer 
  # weights first
  outWeights <- mx.symbol.Variable("out.weight")
  outBias    <- mx.symbol.Variable("out.bias")
  model <- mx.symbol.FullyConnected(data   = model, 
                                    weight = outWeights, 
                                    bias   = outBias, 
                                    num.hidden = 1 ## only one output per element for the final prediction
  )
  
  ## regression output function
  model <- mx.symbol.LinearRegressionOutput(data=model, label=label, name='lstm')
  
  return(model)
  
}


#' @title mxLSTMsetup
#' @description initialize the weights and input matrices with random number to create an executable model
#' @param model mxSymbol as returned by mxLSTMcreate
#' @param num.features number of input features.
#' @param num.hidden see \code{\link{mxLSTM}}
#' @param seq.length see \code{\link{mxLSTM}}
#' @param batch.size see \code{\link{mxLSTM}}
#' @param initializer see \code{\link{mxLSTM}}
#' @param initialModel see \code{\link{mxLSTM}}
#' @param gammaInit    see \code{\link{mxLSTM}}
#' @return MXExecutor
mxLSTMsetup <- function(model, num.features, num.hidden, seq.length, batch.size, initializer, initialModel = NULL, gammaInit){
  
  ## provide a list with known shapes of input arrays.
  ## This will help to estimate the weight dimensions.
  initShapes <- list()
  initShapes$data <- c(num.features, seq.length, batch.size)
  initShapes$label <- c(seq.length, batch.size) ## will fail for num.outputs > 1
  
  for(layer in seq_along(num.hidden)){
    
    ## for each lstm layer, there is a .h and .c input that 
    ## reflects the memory of the previous calculations
    initShapes[[paste0("l", layer, ".c")]] <- c(num.hidden[layer], batch.size)
    initShapes[[paste0("l", layer, ".h")]] <- c(num.hidden[layer], batch.size)
  }

  # infer the shape of the internal arrays (weigths and biases)
  allShapes <- model$infer.shape(initShapes)
  
  ## initialize the executor: bind the model to the input shapes.
  args <- initShapes
  args$symbol <- model
  args$ctx <- mx.cpu()
  ## the following parameters are not updated during backprop
  args$fixed.param <- c(names(initShapes), ## no update on input data, including memory
                        grep("^.*\\.[ih]2h\\.beta", names(allShapes$arg.shapes), value = TRUE) ## no update on offset for bathc norm on inputs. These are fixed to 0
                        )
  args$grad.req <- "write" ## gradient update mode for non-input and non-fixed parameters
  executor <- do.call(mx.simple.bind, args)
  
  ## initialize all weight and bias arrays with random numbers or based on initialModel
  
  if(is.null(initialModel)){
    
    initValues <- list(arg.params = mx.init.create(initializer  = initializer,
                                                   shape.array  = allShapes$arg.shapes, 
                                                   ctx          = mx.cpu(), 
                                                   skip.unknown = TRUE),
                       aux.params = mx.init.create(initializer  = initializer,
                                                   shape.array  = allShapes$aux.shapes, 
                                                   ctx          = mx.cpu(), 
                                                   skip.unknown = TRUE)
    )  
    
    ## by default, gamma of batchNorm layers are initialized to 1. However, Cheng et al. strongly recommend
    ## initializing to 0.1 Do that here.
    whichGamma <- grep("gamma", names(initValues$arg.params))
    initValues$arg.params[whichGamma] <- lapply(initValues$arg.params[whichGamma], function(x) return(x * gammaInit))
    
  } else {
    
    if(!"mxLSTM" %in% class(initialModel)) stop("initialModel must be an mxLSTM object")
    
    initialDims <- list(arg.params = lapply(initialModel$arg.params, dim),
                        aux.params = lapply(initialModel$aux.params, dim))
    thisDims    <- list(arg.params = lapply(executor$arg.arrays, dim),
                        aux.params = lapply(executor$aux.arrays, dim))
    
    ## check if initial model has the correct structure for this model
    equalVectors <- function(x, one, two) all(one[x] == two[x])

    test.arg.params <- 
      lapply(names(initialDims$arg.parms),
             equalVectors,
             one = initialDims$arg.params,
             two = thisDims$arg.params) %>% 
      unlist %>% 
      all
    
    test.aux.params <- 
      lapply(names(initialDims$aux.parms),
             equalVectors,
             one = initialDims$aux.params,
             two = thisDims$aux.params) %>% 
      unlist %>% 
      all

    if(!(test.arg.params & test.aux.params)) {
      stop("Initial model does not fit to current settings")
    }
    
    f <<- function()return(NULL)
    
    initValues <- list(arg.params = initialModel$arg.params,
                       aux.params = initialModel$aux.params)
  }
  
  mx.exec.update.arg.arrays(exec = executor, 
                            arg.arrays = initValues$arg.params, 
                            match.name = TRUE, 
                            skip.null = FALSE)
  
  mx.exec.update.aux.arrays(exec = executor, 
                            arg.arrays = initValues$aux.params, 
                            match.name = TRUE, 
                            skip.null = FALSE)
  
  return(executor)
}


#' @title mxLSTMtrain
#' #' @param model mxSymbol as returned by mxLSTMcreate
#' @description train the LSTM
#' @param datTrain list with entries 'data' and 'label'. 
#'                 'data' is a 3D array of dimension num.features:seq.length:number of observations
#'                 'label is a 2D array of dimension seq.length:numer of observations
#' @param datEval similar to datTrain, but for evaluation instead of training.
#' @param symbol mxSymbol as returned by \code{\link{mxLSTMcreate}}
#' @param batchSize see \code{\link{mxLSTM}}
#' @param num.hidden see \code{\link{mxLSTM}}
#' @param num.rounds see num.epoch argument in \code{\link{mxLSTM}}
#' @param optimizer see \code{\link{mxLSTM}}
#' @param initializer see \code{\link{mxLSTM}}
#' @param initialModel see \code{\link{mxLSTM}}
#' @param shuffle see \code{\link{mxLSTM}}
#' @param gammaInit see \code{\link{mxLSTM}}
#' @param epoch.end.callback function to be called at the end of each epoch.
#' @param ... further arguments passed to optimizer
#' @return  object of class 'mxLSTM' 
#'        list: 'model' is the actual symbol, 'arg.params' and 'aux.params' are the parameters,
#'         'log' is the training log, 'optimizerEnv' is an optional environment with optimizer parameters.
mxLSTMtrain <- function(datTrain, datEval, symbol, batchSize, num.hidden, num.rounds, optimizer = "rmsprop", initializer = mx.init.Xavier(),
                        initialModel = NULL, shuffle = TRUE, gammaInit, epoch.end.callback = NULL, ...){
  
  seq.length   <- dim(datTrain$data)[2]
  batch.size   <- batchSize
  num.features <- dim(datTrain$data)[1]
  
  ## executor
  model  <- mxLSTMsetup(model        = symbol, 
                        num.features = num.features, 
                        num.hidden   = num.hidden, 
                        seq.length   = seq.length, 
                        batch.size   = batch.size, 
                        initializer  = initializer, 
                        initialModel = initialModel,
                        gammaInit    = gammaInit)

  init.states.name <- grep(".*\\.[ch]$", symbol$arguments, value = TRUE)
  
  ## prepare the input data iterators
  trainIterator <- mx.io.arrayiter(datTrain$data, datTrain$label, batch.size, shuffle = shuffle)
  
  if(!is.null(datEval)){
    
    evalIterator  <- mx.io.arrayiter(datEval$data, datEval$label, batch.size, shuffle = FALSE)
    
  } else {
    
    evalIterator <- NULL
  }
  
  ## prepare optimizer
  opt <- mx.opt.create(optimizer, rescale.grad = (1/batch.size), ...)
  updater <- mx.opt.get.updater(opt, model$ref.arg.arrays)
  
  ## set evaluation metric to RMSE
  metric <- mx.metric.rmse
  
  for (epoch in 1:num.rounds) {
    
    ## beginning of an epoch:
    ## Clear input state arrays
    init.states.cleared <- 
      lapply(model$arg.arrays[init.states.name], function(x) return(x * 0))
    
    mx.exec.update.arg.arrays(model, init.states.cleared, match.name = TRUE)
    
    ## reset train iterator to first batch
    trainIterator$reset()
    
    ## initialize train metric for this epoch
    train.metric <- metric$init()  
    
    while (trainIterator$iter.next()) {
      
      ## beginning of training batch
      
      ## set input data
      data <- trainIterator$value()
      mx.exec.update.arg.arrays(model, data, 
                                match.name = TRUE)
      
      ## forward pass
      mx.exec.forward(model, is.train = TRUE)
      
      ## backward pass
      mx.exec.backward(model)
      
      ## get the updated weights and biases from the optimizer
      arg.blocks <- updater(model$ref.arg.arrays, 
                            model$ref.grad.arrays)
      
      names(arg.blocks) <- names(model$arg.arrays)
      
      ## insert the updated weights and biases
      mx.exec.update.arg.arrays(model, arg.blocks, match.name = TRUE, skip.null = TRUE)
      
      ## clear input state arrays after update
      mx.exec.update.arg.arrays(model, init.states.cleared, match.name = TRUE)
      
      
      ## update the train metric.
      ## Only use the last value of each sequence
      train.metric <- 
        metric$update(label = data$label %>% 
                        mx.nd.choose.element.0index(rhs = rep(seq.length - 1, 
                                                              batch.size) %>% mx.nd.array()),
                      pred  = model$ref.outputs$lstm_output %>% ## will fail with num.output > 1 
                        mx.nd.take(indices = (seq.length - 1) * batch.size + (0 : (batch.size - 1)) %>% 
                                     mx.nd.array()),
                      state = train.metric)
      
    }
    
    ## start evaluation
    if(!is.null(evalIterator)){
      
      ## reset eval iterator to first batch
      evalIterator$reset()
      
      ## initialize evaluation metric for this epoch
      eval.metric <- metric$init()
      
      while (evalIterator$iter.next()) {
        
        ## beginning of evaluation batch
        ## set input data
        data <- evalIterator$value()
        mx.exec.update.arg.arrays(model, data, 
                                  match.name = TRUE)
        
        ## forward pass
        mx.exec.forward(model, is.train = FALSE)
        
        ## update the evaluation metric (only use the last prediction of each sequence)
        eval.metric <- 
          metric$update(label = data$label %>% 
                          mx.nd.choose.element.0index(rhs = rep(seq.length - 1, 
                                                                batch.size) %>% mx.nd.array()),
                        pred  = model$ref.outputs$lstm_output %>% ## will fail with num.output > 1 
                          mx.nd.take(indices = (seq.length - 1) * batch.size + (0 : (batch.size - 1)) %>% 
                                       mx.nd.array()),
                        state = eval.metric)
        
        ## clear input state arrays after update
        mx.exec.update.arg.arrays(model, init.states.cleared, match.name = TRUE)
        
      }
    }
    
    
    ## epoch end callback here
    continue <- TRUE
    if(!is.null(epoch.end.callback)){
      
      continue <- epoch.end.callback(epoch, 0, environment())
    }
    
    if(!continue) break
    
  }
  
  return(list(symbol = symbol, arg.params = model$arg.arrays, aux.params = model$aux.arrays))
  
}

#' @title lstmCell
#' @description constructs an lstm cell. Identical to mxnet:::lstm, except for relu activation instead of tanh
#' @param num.hidden number of hidden neurons in the cell's state
#' @param indata the input to the cell
#' @param prev.state the memorized state from the previous cell.
#' @param param list of variables with weights and biases. must contain elements i2h.weight, i2h.bias, h2h.weight, h2h.bias
#' @param seqidx sequence index. Purely bookkeeping
#' @param layeridx index for the layer that the cell belongs to
#' @param dropout see \code{\link{mxLSTM}}
#' @param zoneout see \code{\link{mxLSTM}}
#' @param batchNorm see \code{\link{mxLSTM}}
#' @param activation activation function for update layers. "relu" or "tanh"
#' @return mxSymbol
lstmCell <- function (num.hidden, indata, prev.state, param, seqidx, layeridx, 
                      dropout = 0, zoneout = 0, batchNorm = FALSE, activation = "relu") 
{

  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, 
                                  bias = param$i2h.bias, num.hidden = num.hidden * 4, 
                                  name = paste0("t", seqidx, ".l", layeridx, ".i2h"))
  
  h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                  no.bias = TRUE, ## bias of i2h layer is suficient. Avoid redundancy
                                  num.hidden = num.hidden * 4, name = paste0("t", seqidx, ".l", layeridx, ".h2h"))
  
  if(batchNorm) {
    ## batch normalize i2h and h2h separately. Be careful in setup: offset parameters (beta) should be 0 and not trained
    ## because ofset is sufficiently addressed by the bias of i2h.
    ## Also be careful to initialize the scale (gamma) to 0.1 (strong recommendation of Cheng et al.)
    i2h <- mx.symbol.BatchNorm(data        = i2h, 
                               gamma       = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".i2h.gamma")), 
                               beta        = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".i2h.beta")),
                               moving_mean = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".i2h.mean")),
                               moving_var  = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".i2h.var")),
                               name = paste0("t", seqidx, ".l", layeridx, ".i2h.batchNorm"))
    
    h2h <- mx.symbol.BatchNorm(data        = h2h, 
                               gamma       = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".h2h.gamma")), 
                               beta        = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".h2h.beta")),
                               moving_mean = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".h2h.mean")),
                               moving_var  = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".h2h.var")),
                               name = paste0("t", seqidx, ".l", layeridx, ".h2h.batchNorm"))
  }
  
  gates <- i2h + h2h
  slice.gates <- mx.symbol.SliceChannel(gates, num.outputs = 4, 
                                        name = paste0("t", seqidx, ".l", layeridx, ".slice"))
  in.gate <- mx.symbol.Activation(slice.gates[[1]], act.type = "sigmoid")
  if(dropout > 0) in.gate <- mx.symbol.Dropout(data = in.gate, p = dropout)
  in.transform <- mx.symbol.Activation(slice.gates[[2]], act.type = activation)
  forget.gate <- mx.symbol.Activation(slice.gates[[3]], act.type = "sigmoid")
  if(dropout > 0) forget.gate <- mx.symbol.Dropout(data = forget.gate, p = dropout)
  out.gate <- mx.symbol.Activation(slice.gates[[4]], act.type = "sigmoid")
  if(dropout > 0) out.gate <- mx.symbol.Dropout(data = out.gate, p = dropout)
  
  next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  if(!batchNorm){
    next.h <- out.gate * mx.symbol.Activation(next.c, act.type = activation)
  } else {
    next.cNorm <- mx.symbol.BatchNorm(data        = next.c, 
                                       gamma       = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".c.gamma")), 
                                       beta        = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".c.beta")),
                                       moving_mean = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".c.mean")),
                                       moving_var  = mx.symbol.Variable(paste0("t", seqidx, ".l", layeridx, ".c.var")),
                                       name = paste0("t", seqidx, ".l", layeridx, ".c.batchNorm"))
    next.h <- out.gate * mx.symbol.Activation(next.cNorm, act.type = activation)
  }
  
  if(zoneout > 0){
    next.c <- zoneout(thisState = next.c, prevState = prev.state$c, p = zoneout)
    next.h <- zoneout(thisState = next.h, prevState = prev.state$h, p = zoneout)
  }
  
  return(list(c = next.c, h = next.h))
}

#' @title zoneout
#' @description Applies zoneout as described in Krueger et al. 2017
#' @param thisState the current state (mx.symbol)
#' @param prevState the state from the previous sequence (mx.symbol). 
#' @param p zoneout probability in (0;1]. Rounded to 6 digits
#' @return During training: out(t) = x(t) * thisState(t) + (1 - x(t)) * prevState(t). 
#'         x(t) is a 0/1 mask for each element t with the probability p of being 0.
#'         During application: out(t) = (1 - p) * thisState(t) + p * prevState(t)

zoneout <- function(thisState, prevState, p){
  
  if(!"Rcpp_MXSymbol" %in% class(thisState)) stop("thisState must be an Rcpp_MXSymbol")
  if(!"Rcpp_MXSymbol" %in% class(prevState)) stop("prevState must be an Rcpp_MXSymbol")
  if(length(p) != 1 | !is.numeric(p) | p < 0 |  p >= 1) stop("p must be a length 1 numeric vector in the interval (0;1]")

  ## create a vector that is one after the 1/(1-p) transformation of mx.symbol.Dropout during training
  ## during inference, it will simply contain values (1-p)
  zoneoutVec  <- mx.symbol.ones_like(data = prevState, name = "zoneoutVec") * (1-p)
  zoneoutMask <- mx.symbol.Dropout(data = zoneoutVec, p = p, mode = "training", name = "zoneoutMask")
  
  out <- thisState * zoneoutMask + prevState * (1-zoneoutMask)
  
  return(out)
}