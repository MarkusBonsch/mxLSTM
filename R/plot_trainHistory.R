#' @title plot_trainHistory
#' @description Plots the performance after each training epoch for a model 
#' as returned by \code{\link{fitLSTMmodel}} 
#' @param models a named vector with models.Can either be a model object or a path to a saved model
#' @return plot as returned by highcharter
#' @importFrom highcharter highchart hc_xAxis hc_yAxis hc_title hc_add_series hc_chart
#' @importFrom magrittr %>%
#' @export plot_trainHistory
#' @seealso \code{\link{getTrainHistory}}
#' 
plot_trainHistory <- function(models){
  
  ## create plot
  p <- highchart() %>% 
    hc_xAxis(title = list(text = "epoch"), type = "linear") %>% 
    hc_yAxis(title = list(text = "Value of error function"), 
             labels=list(format = "{value:.2f}")) %>% 
    hc_title(text =  paste0("Model performance over training epoch"), style = 
               list(fontWeight = "bold"))
  
  ## take care if only a single model is specified
  if(class(models) != "list") models <- list(models)
  
  colors <- rep(c("#ff0000", "#00ff00", "#0000ff", "000000"), length.out = length(models))
  
  for(i in seq_along(models)){
    
    name <- names(models[i])
    if(is.null(name)) name <- i
    
    ## extract data
    plotData <- getTrainHistory(models[[i]])
    p <- 
      p %>% 
      hc_add_series(name = paste0(name, "_train"), type = "line", data = plotData$train,
                    color = colors[i], marker = list(symbol = "square"), dashStyle = "LongDash") %>% 
      hc_add_series(name = paste0(name, "_test"), type = "line", data = plotData$eval,
                    color = colors[i], marker = list(symbol = "square"), dashStyle = "Solid") %>%   
      hc_chart(zoomType = "xy")
  }
  
  return(p)
}