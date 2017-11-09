#' @title
#'  plot_GoodnessOfFit
#' @description
#' Function to plot some godness of fit indicators for regression results
#' @param predicted vector of predicted values
#' @param observed vector of observed values
#' @param plotIt If TRUE, plots are printed, if FALSE, they are only returned.
#' @param combine If TRUE, all plots are combined before, if FALSE, a list of individual plots is returned.
#' @return a plot (combine = TRUE) or a list of plots (combine = FALSE)
#' @details For details on the goodness of fit measures, see \code{\link{calc_goodnessOfFit}}
#' The plots contain the following information:
#' \describe{
#' \item{\code{Predicted vs observed values}}{scatterplot of predicted over observed values. \cr 
#'                                     Blue line indicates perfect fit, red line indicates best 
#'                                     fit to data. For the meaning of performance measures, 
#'                                     see \code{\link{calc_goodnessOfFit}}.}
#' \item{\code{Distribution of residuals}}{density plot of the distribution of residuals (red area) 
#'                                  with best fit gaussian (black dashed line). 
#'                                  Blue line at x=0, red line for mean of residuals. 
#'                                  For the meaning of performance measures, 
#'                                  see \code{\link{calc_goodnessOfFit}}.}
#' \item{\code{qqplot}}{standard qqplot. See \code{\link[stats]{qqnorm}}.}
#' \item{\code{Residuals over observed values}}{Scatterplot of residuals over observed values. 
#'                                       Blue line at 0. Red line for best fit to data.}
#' \item{\code{Residuals over order}}{Scatterplot of residuals over order (in case of timeseries: time). 
#'                             Red area indicates the 3 sigma interval around the mean of the 
#'                             residuals.}
#' }
#' @seealso
#' \code{\link{calc_goodnessOfFit}}
#' @importFrom cowplot plot_grid
#'
#' @export plot_goodnessOfFit
#' @examples test <- plot_goodnessOfFit(predicted = 1:10 + rnorm(10,0,0.3), observed = 1:10)

plot_goodnessOfFit <- function(predicted, observed, plotIt = TRUE, combine = TRUE) {
  
  ###########################################
  ## get goodness of fit measures
  ###########################################
  measures <- calc_goodnessOfFit(predicted, observed)
  
  delta <- predicted - observed
  plots <- list()
  
  ###############################################
  ## scatterplot predicted vs observed with y=x line and best fit line
  ## RMSE, and rSquared as text
  ###############################################
  plots$scatterObservedPredicted <-
    ggplot(data = data.frame(x = observed, y = predicted),
           aes(x = x, y=y)) +
    geom_point(alpha = 0.3) +
    ## ideal result line
    geom_abline(slope = 1, intercept = 0, color = "blue", size = 1.5) +
    ## best fit line
    geom_smooth(method = "lm", se = FALSE, color = "red", size = 1.5) + 
    ## add text with goodness of fit measures
    annotate(geom = "text", 
             y = (1 - 0.1 * sign(max(observed, na.rm = TRUE))) * max(observed, na.rm = TRUE),
             x = min(observed, na.rm = TRUE) + 0.2 * (max(observed, na.rm = TRUE) - min(observed, na.rm = TRUE)),
             label = paste0("RMSE: ", round(measures$rmse, 2), 
                            "\nR^2: ", round(measures$rSquared, 2))) +
    ## set scaling so that the ideal line is 45 degree
    coord_cartesian(xlim = range(observed, na.rm = TRUE), ylim = range(observed, na.rm = TRUE)) + 
    ## nice styling
    theme_bw() + 
    labs(title = "Predicted vs observed values",
         x    = "Observed",
         y    = "Predicted")
  
  ################################################
  ## density plot of residuals with fitted gauss +
  ## test for normality test and best gauss fit
  ################################################
  meanResiduals <- mean(delta, na.rm = TRUE)
  sdResiduals   <- sd(delta,na.rm = TRUE)
  meanErrorResiduals <- sdResiduals/(sqrt(length(residuals)) * sdResiduals)
  
  plots$residualsGauss <-
    ## density plot of data
    ggplot(data = data.frame(x = delta), aes(x = delta)) + 
    geom_density(color = "red", fill = "red", alpha = 0.1) +
    ## add fitted gauss
    stat_function(fun = dnorm, args = list(mean = meanResiduals, sd = sdResiduals),
                  color = "black", size = 1.5, linetype = 5) +
    ## add vertical line at 0
    geom_vline(xintercept = 0, color = "blue", size = 1.5) +
    ## add vertical line at meanResiduals
    geom_vline(xintercept = meanResiduals, color = "red", size = 1.5) +
    ## add text with goodness of fit measures
    annotate(geom = "text", 
             y = max(density(delta, na.rm = TRUE)$y) * 0.9,
             x = min(delta, na.rm = TRUE) + 0.15 * (max(delta, na.rm = TRUE) - min(delta, na.rm = TRUE)),
             label = paste0("Mean: ", round(meanResiduals, 2), "+- ", round(meanErrorResiduals, 2), 
                            "\nsigma: ", round(sdResiduals, 2),
                            "\np-value :", round(measures$normalityTestResiduals,2),
                            "\n(H0: mu=0; sigma=", round(sdResiduals, 2),")"))+
    ## nice styling
    theme_bw() + 
    labs(title = "Distribution of residuals",
         x    = "Residuals",
         y    = "Probability density")
  
  
  
  ################################################
  ## qqplot: quantiles of residuals vs 
  ## theoretical quantiles
  ################################################
  
  dat <- as.data.frame(qqnorm(delta, plot.it = FALSE))
  ## get slope and intercept of perfect fit (see qqline code)
  y <- quantile(delta, c(0.25, 0.75), names = FALSE, type = 7, na.rm = TRUE)
  x <- qnorm(c(0.25, 0.75))
  qqslope <- diff(y) / diff(x)
  qqintercept <- y[1] - qqslope * x[1]
  rm(x,y)
  
  plots$qqplot <-
    ggplot(data = dat, aes(x = x, y = y)) +
    geom_point() + geom_abline(intercept = qqintercept, slope = qqslope, size = 1.5, color = "blue") +
    ## nice styling
    theme_bw() + 
    labs(title = "qqplot",
         x    = "Theoretical quantiles",
         y    = "Sample quantiles")
  
  
  
  ################################################
  ## Residuals over observed value with trend
  ## (test for heteroscedasticity)
  ################################################
  
  plots$residualsObserved <- 
    ggplot(data = data.frame(x = observed, y = delta), aes(x = x, y = y)) +
    geom_point() + geom_smooth(method = "lm", size = 1.5, color = "red") +
    geom_hline(yintercept = 0, size = 1.5, color = "blue") +
    ## nice styling
    theme_bw() + 
    labs(title = "Residuals over observed",
         x    = "Observed",
         y    = "Residuals")
  
  ################################################
  ## Residuals over time with 
  ## three sigma band
  ################################################
  
  plots$residualsTime <- 
    ggplot(data = data.frame(x = seq_along(predicted), y = delta), aes(x = x, y = y)) +
    geom_point() +
    ## add best fit line
    geom_hline(yintercept = 0, size = 1.5, color = "blue") +
    ## add three sigma region
    geom_ribbon(ymax = mean(delta, na.rm = TRUE) + 3* sd(delta, na.rm = TRUE),
                ymin = mean(delta, na.rm = TRUE) - 3* sd(delta, na.rm = TRUE),
                fill = "red", alpha = 0.15) +
    ## nice styling
    theme_bw() + 
    labs(title = "Residuals over order",
         x    = "Order of observations",
         y    = "Residuals")
  
  
  if(combine) {
    ## combine all plots into one
    plots <- do.call(plot_grid, args = plots)
  }
  
  if(plotIt){
    do.call(print, args = list(plots))
  }
  
  return(plots)
  
}