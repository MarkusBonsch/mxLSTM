#' @title
#' getLSTMmodel
#' @description
#' Constructs a custom \code{\link{mxLSTM}} model for use in the caret 
#' \code{\link[caret]{train}} logic. It behaves slightly different than the usual 
#' caret models as retrieved by \code{\link[caret]{getModelInfo}}. See details.
#' @details
#' \strong{model setup\cr}{The model is an LSTM recurrent neural network model with rmsprop optimizer.\cr }
#' 
#' \strong{Purpose}\cr 
#' The purpose of the custom model is the following:
#' \describe{
#'   \item{Allow multiple y}{Allow a regression within caret that predicts multiple y in one model.}
#'   \item{Scaling of y}{Allow for scaling of y. Possible options are \code{c('scale', 'center', 'minMax')}}
#'   \item{scale x variables again}{If e.g. a PCA is conducted in the preprocessing, the resulting inputs can be scaled again
#'                                  by preProcessing options \code{c('scaleAgain', 'centerAgain')}}
#' }
#' 
#' \strong{Usage}\cr 
#' The model differs from 'usual' caret models in its usage. 
#' Differences when using it in \code{\link[caret]{train}}:
#' \describe{
#' \item{Different formula for model specification}{Usually, the formula would be for example \code{y1+y2+y3 ~ x1+x2+x3}.
#'                                                   Caret does not allow this specification, therefore a hack is used:
#'                                                   \itemize{
#'                                                     \item {construct a column \code{dummy = y1}}
#'                                                     \item {Specify the formula as \code{dummy~x1+x2+x3+y1+y2+y3}.}
#'                                                     \item {Determine x and y variables with the arguments \code{xVariables = c('x1', 'x2', 'x3')}
#'                                                     and \code{yVariables = c('y1','y2','y3')}}}}
#' \item{Different pre-processing arguments}{\itemize{
#'                                            \item{Don't us the caret \code{preProcess} argument. Use \code{preProcessX} and 
#'                                                  \code{preProcessY} instead}
#'                                            \item{Don't specify \code{preProcOptions} in the \code{\link[caret]{trainControl}} call.
#'                                                 Specify them in the call to \code{\link[caret]{train}}. 
#'                                                 They will be valid for preProcessX only since y pre-processing does not require further arguments.
#'                                                 preProcessX can be anything that caret allows plus \code{c('scaleAgain', 'centerAgain')} 
#'                                                 for scaling as a last preProcessing step. \code{preProcessY} can include
#'                                                 \code{c('scale', 'center', 'minMax')}.}}}
#'                                                 
#' \item{Additional mandatory arguments to fit function} {For transforming the input to the LSTM, the following additional arguments 
#'                                       must be specified to the train function:
#'                                       \itemize{
#'                                            \item{\code{seqLength}: The sequence length (number of rows in input)}
#'                                            \item{\code{seqFreq}: frequency of sequence starts (in rows). 
#'                                                                  If smaller than seqLength, sequences are overlapping}
#'                                       }}
#'   
#' \strong{Additional argumets to fit function:}
#'       \itemize{
#'          \item{testData:}{ dataset for evaluating performance after each epoch}
#'          \item{initialModel:} { can be specified if the aim is to continue 
#'                               training on an existing model. \cr
#'                               Has to be the output of a call to \code{\link{fitLSTMmodel}}. \cr
#'                               ATTENTION! Be sure, to specify the same xVariables, yVariables, hidden layers, and
#'                               preProcessing steps as in the original training.\cr
#'                            }
#'          \item{seed:} {Optional random seed that is set before model training for reproducibility.}
#'       }
#'       
#' \strong{Additional argumets to predict function:}
#'       \itemize{
#'          \item{fullSequence:}{ Boolean. If FALSE, only the last output of a sequence is returned.
#'                                         If TRUE, the output for the whole sequence is returned.}
#'       }
#'                                                 
#'                                               
#' \item{Different prediction function}{For predicting from the model as returned by caret's \code{\link[caret]{train}},
#'                                      you have to use the \code{\link{predictAll}}} function. This will call the internal
#'                                      predict function of \code{getLSTMmodel} returning predictions for all y-variables.
#'                                                 
#' \strong{tuning parameters\cr}
#'                        \itemize{
#'                          \item{num.epoch:}{ number of training epochs}
#'                          \item{batch.size:}{ batch size}
#'                          \item{layer1}{ number of hidden units in LSTM layer 1.}
#'                          \item{layer2}{ number of hidden units in LSTM layer 2.}
#'                          \item{layer3}{ number of hidden units in LSTM layer 3.}
#'                          \item{dropout1}{ dropout probability for LSTM layer 1}
#'                          \item{dropout2}{ dropout probability for LSTM layer 2}
#'                          \item{dropout3}{ dropout probability for LSTM layer 3}
#'                          \item{activation}{ Activation function for the update layers in the LSTM cells. "relu" or "tanh"}
#'                          \item{shuffle}{ Boolean. Should the training batches be randomly reordered? 
#'                                Each sequence of course stays in its native order}
#'                          \item{learning_rate:} { defaults to 0.002}
#'                          \item{weight_decay:} { defaults to 0}
#'                          \item{dropout:}{ [0;1] fraction of neurons that are randomly discarded in each hidden layer during training. Default: 0}
#'                          \item{learningrate_momentum:} { gamma1. See API description of mx.opt.rmsprop}
#'                          \item{momentum:} { gamma2. See API description of mx.opt.rmsprop.}
#'                          \item{clip_gradients:} { See API description of mx.opt.rmsprop}
#'                                             
#'                        }
#' 
#' \strong{Other specific features}
#'   \itemize{
#'     \item{\strong{plot training history}} {It is possible to plot the training history
#'       of an mxLSTM model with \code{\link{plot_trainHistory}}}
#'     \item{\strong{restore checkpoint from specified epoch}}{ It is possible to restore
#'      the model weights after a given epoch with the function \code{\link{restoreLSTMcheckpoint}}.}
#'       
#'   }
#' @return A list of functions similar to the output of caret's \code{\link[caret]{getModelInfo}}:\cr
#' @seealso \code{\link{saveCaretLstmModel}}, \code{\link{loadCaretLstmModel}},
#' \code{\link{plot_trainHistory}}, \code{\link{fitLSTMmodel}}, 
#' \code{\link{predictLSTMmodel}}, \code{\link{getPreProcessor}}, 
#' \code{\link{predictPreProcessor}}, \code{\link{invertPreProcessor}}
#' 
#' @examples 
#' \dontrun{
#' }
#' @export getLSTMmodel

getLSTMmodel <- function(){

  lstmModel <- list()
  
  lstmModel$label <- "MXnet Long-short-term memory recurrent neural network with rmsProp optimizer"
  
  lstmModel$library <- "mxnet"
  
  lstmModel$type <- "Regression"
  
  lstmModel$tags <- "LSTM recurrent neural network"
  
  ## prob is not required for regression
  lstmModel <- c(lstmModel, prob = NA)
  
  lstmModel$parameters <- 
    data.frame(parameter = "layer1", class = "numeric", label = "Hidden nodes in LSTM layer 1")
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "layer2", class = "numeric", label = "Hidden nodes in LSTM layer 2"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "layer3", class = "numeric", label = "Hidden nodes in LSTM layer 3"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "num.epoch", class = "numeric", label = "number of passes over the full training set"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "activation", class = "character", label = "activation function for update layers"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "batch.size", class = "numeric", label = "batch size for training"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "dropout1", class = "numeric", label = "Dropout probability for inputs to LSTM layer 1"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "dropout2", class = "numeric", label = "Dropout probability for inputs to LSTM layer 2"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "dropout3", class = "numeric", label = "Dropout probability for inputs to LSTM layer 3"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "shuffle", class = "logical", label = "switch for activating reshuffling of training events"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "learning.rate", class = "numeric", label = "Initial learning rate"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "weight.decay", class = "numeric", label = "Weight decay"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "learningrate.momentum", class = "numeric", label = "Factor for adjusting moving average for gradient^2 (gamma1)"))
  lstmModel$parameters <- 
    rbind(lstmModel$parameters, 
          data.frame(parameter = "momentum", class = "numeric", label = "Momentum (gamma2)"))

  
  lstmModel$grid <- 
    function(x, y, len = NULL, search = "grid"){
      if(search != "grid") stop("Only grid search implemented so far for mxLSTM")
      out <- expand.grid(seq.length = 12, layer1 = ((1:len) * 2) - 1, layer2 = 0, layer3 = 0, 
                         num.epoch = 10, activation = "relu", batch.size = 128, 
                         dropout1 = 0, dropout2 = 0, dropout3 = 0, shuffle = TRUE,
                         learning.rate = 0.002, weight.decay = 0, learningrate.momentum = 0.9, 
                         momentum = 0.9)
    }
  

  ## add scaling of predictors and target to the fit function
  lstmModel$fit <- function(x, y, wts, param, lev, last, classProbs, xVariables, yVariables,
                             preProcessX = NULL,
                             preProcessY = NULL,
                             preProcOptions = list(thresh = 0.95, ICAcomp = 3, k = 5, freqCut = 95/5,
                                                   uniqueCut = 10, cutoff = 0.9),
                             debugModel = FALSE,
                             testData = NULL, ## data frame with the same format as x for validation
                             initialModel = NULL, 
                             seed = NULL,
                             ...){
  
    
    ## set global option for debug so that also the predict function will be in debug mode.
    if(debugModel) options(mxLSTM.debug = TRUE)
    
    if(getOption("mxLSTM.debug")){
      cat("###############################################################\n")
      cat("#################Preprocessing x and y        #################\n")
      cat("###############################################################\n")
      cat("###########input data##############\n")
      print(summary(x))
      cat("###########input xVariables##############\n")
      print(xVariables)
      cat("###########input yVariables##############\n")
      print(yVariables)
      cat("###########preProcessing options##############\n")
      print(preProcOptions)
    }
    
    if(length(yVariables) > 1) stop("mxLSTM currently only available for single y column")
    
    ## do the preProcessing

    ## transform the data to pre-processable data.frame
    dat <- caret2lstmInput1(x)
    
    if(getOption("mxLSTM.debug")){
      cat("#################input after caret2lstmInput1 #################\n")
      print(summary(dat))
    }
    
    if(any(!xVariables %in% names(dat))) stop("xVariables contains variables that are not in x or on
                                              the right hand side of the formula")
    
    if(any(!yVariables %in% names(dat))) stop("yVariables contains columns that are not in x or that also appear on the right hand side of the formula. 
                                               Make sure that the left hand side of the formula
                                               contains a copy of y that is called 'dummy'. 
                                               The right hand side of the formula should contain the real y variables")
    
    

    
    seq.length <- max(dat$sequenceId)
                      
    ## split data into id,  x and y
    datX <- data.frame(dat[, xVariables, with = FALSE])
    datY <- data.frame(dat[, yVariables, with = FALSE])
    
    ## preProcess x and y
    preProcessX <- getPreProcessor(dat = datX,
                                     preProcess = preProcessX,
                                     preProcOptions = preProcOptions)
    datX <- predictPreProcessor(preProcess = preProcessX, dat = datX)
    
    preProcessY <- getPreProcessor(dat = datY,
                                     preProcess = preProcessY,
                                     preProcOptions = preProcOptions)
    datY <- predictPreProcessor(preProcess = preProcessY, dat = datY)

    dat <- transformLSTMinput(cbind(datX, datY), targetColumn = yVariables, seq.length = seq.length)

    if(getOption("mxLSTM.debug")){
      cat("#################x and y after preProcessing #################\n")
      print(summary(dat$x))
      print(summary(dat$y))
      print("xVariables:")
      print(dimnames(dat$x)[[1]])
      print("yVariables:")
      print(unique(dimnames(dat$y)[[2]]))
    }
        
    ## same for test data if applicable
    if(!is.null(testData)){
      
      testData <- caret2lstmInput1(testData)
      
      testX    <- data.frame(testData[, xVariables, with = FALSE])
      testY    <- data.frame(testData[, yVariables, with = FALSE])
      
      testX    <- predictPreProcessor(preProcessX, testX)
      testY    <- predictPreProcessor(preProcessY, testY)
      
      testData <- transformLSTMinput(cbind(testX, testY), targetColumn = yVariables, seq.length = seq.length)
      
    } else {
      
      testData <- list(x = NULL, 
                       y = NULL)

    }
    
    if(getOption("mxLSTM.debug")){
      cat("#################input data right before model fit #################\n")
      cat("dat$x:\n")
      print(dim(dat$x))
      print(summary(dat$x))
      cat("dat$y:\n")
      print(dim(dat$y))
      print(summary(dat$y))
    }
    
    ## fit the model
    model <- fitLSTMmodel(x = dat$x, y = dat$y, test.x = testData$x, test.y = testData$y, 
                           initialModel = initialModel, seed = seed, param = param, ...)
                  
    ## add the preProcessing information, so that it can be used in predict.
    model$preProcessX <- preProcessX
    model$preProcessY <- preProcessY
    model$xVariables  <- xVariables
    model$yVariables  <- yVariables

    ## if it's the last training, reset debug option
    if(last == TRUE) options(mxLSTM.debug = FALSE)
    
    return(model)
    }
  
  ## add scaling to the predict function
  lstmModel$predict <- function(modelFit, newdata, submodels = NULL, allY = FALSE, debugModel = FALSE, fullSequence = FALSE){
    
    ## set debugging option.
    if(debugModel) options(mxLSTM.debug = TRUE)

    ## get x and y variable names
    xVariables <- modelFit$xVariables
    names(xVariables) <- xVariables
    yVariables <- modelFit$yVariables
    names(yVariables) <- yVariables
    
    if(getOption("mxLSTM.debug")){
      cat("###############################################################\n")
      cat("#################Starting prediction          #################\n")
      cat("###############################################################\n")
      cat("#################newdata before transformation#################\n")
      print(summary(newdata))
      cat("#################xVariables#################\n")
      print(xVariables)
      cat("#################yVariables#################\n")
      print(yVariables)
      cat("#################Preprocessing#################\n")
      cat("x: ")
      print(modelFit$preProcessX)
      cat("y: ")
      print(modelFit$preProcessY)
    }
    
    newdata <- as.data.frame(newdata)

    ## transform newdata
    newdata <- as.data.frame(caret2lstmInput1(newdata))
    
    if(getOption("mxLSTM.debug")){
      cat("###########newdata after caret2lstmInput1##############\n")
      print(summary(newdata))
    }
    
    seq.length <- max(newdata$sequenceId)
    
    newdataX <- newdata[, xVariables, drop = FALSE]
    newdataY <- newdata[, yVariables, drop = FALSE]
    
    ## apply the scaling to xVariables
    newdataX <- predictPreProcessor(modelFit$preProcessX, newdataX)
    
    if(getOption("mxLSTM.debug")){
      cat("###########newdata after predictPreprocessor##############\n")
      print(summary(cbind(newdataX, newdataY)))
      print(yVariables)
      print(seq.length)
    }
    
    newdata <- transformLSTMinput(dat          = cbind(newdataX, newdataY), 
                                  targetColumn = yVariables, 
                                  seq.length   = seq.length)
    
    if(getOption("mxLSTM.debug")){
      cat("###########newdata right before prediction##############\n")
      cat("newdata$x:\n")
      print(dim(newdata$x))
      print(summary(newdata$x))
    }
    
    ## now do the real prediction. Since we have one row per event in the input, 
    ## we take the alst sequence element for each event as prediction
    out <- predictLSTMmodel(model = modelFit, dat = newdata$x, fullSequence = fullSequence)
    
    if(getOption("mxLSTM.debug")){
      cat("###########predicted y before retransformation##############\n")
      print(summary(out))
      print(modelFit$yVariables)
    }
    
    out$rowIndex <- NULL
    names(out) <- yVariables

    ## invert transformation on y.
    out <- invertPreProcessor(preProcessor = modelFit$preProcessY, dat = out)
    
    
    if(!allY) {
      
      ## select first yVariable if required  
      out <- as.numeric(out[,1])
    
    }
    
    if(getOption("mxLSTM.debug")){
      cat("###########predicted y after retransformation##############\n")
      print(summary(out))
    }
    
    ## reset global debugging option
    options(mxLSTM.debug = FALSE)
    
    return(out)
  }

  return(lstmModel)
}