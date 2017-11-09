#' @title
#'  calc_GoodnessOfFit
#' @description
#' Function to calculate goodness of fit measures for regressions.
#' @param predicted vector of predicted values
#' @param observed vector of observed values
#' @return A list with the following items:\cr
#'        \describe{
#'           \item{\code{rmse}}{Root mean square error 
#'                       = \code{sqrt(mean((predicted - observed)^2), na.rm = TRUE))}}
#'           \item{\code{rSquared}}{Coefficient of determination =  
#'                            \code{1 - sum((predicted - observed)^2) / sum((observed - mean(observed, na.rm = TRUE))^2)} 
#'                            Useful indicator, although for non-linear models it is not strictly appropriate since the 
#'                            'null hypothesis', a horizontal line, is not necessarily a subset of the model space. 
#'                            Based on the traditional definition, see \code{?caret::R2}.}
#'           \item{\code{normalityTestResiduals}}{P-value of a kolmogorov-smirnoff test 
#'                                          (\code{\link[stats]{ks.test}}). Null-hypothesis: Normal distribution 
#'                                           with mean 0 and standard deviation = sd(predicted - observed))}
#'         }
#' @seealso
#' \code{\link{plot_goodnessOfFit}}
#'
#' @export calc_goodnessOfFit
#' @examples test <- calc_goodnessOfFit(predicted = 1:10 + rnorm(10,0,0.3), observed = 1:10)

calc_goodnessOfFit <- function(predicted, observed) {

  if(length(predicted) != length(observed)) stop("Predicted and observed need to have same length")  
  ## residuals
  delta <- predicted - observed
  
  if(length(na.omit(delta)) < 2) stop("Too few non-missing residuals. At least two are needed.")
  
  ## calculate important quantities
  measures <- list()
  
  ##############################################
  ## root mean square error
  ##############################################
  measures$rmse <- sqrt(mean(delta^2, na.rm = TRUE))
  
  ##############################################
  ## R^2 
  ##############################################
  ## useful, although for non-linear models it is not 
  ## strictly appropriate since the 'null hypothesis', a horizontal line, 
  ## is not necessarily a subset of the model space. Although, mostly it is, I guess.)
  ## based on the traditional definition, see ?caret::R2
  measures$rSquared <- 1 - sum((predicted - observed)^2, na.rm = TRUE) / sum((observed - mean(observed, na.rm = TRUE))^2, na.rm = TRUE)
  
  ##############################################
  ## normality test for residuals
  ##############################################
  ## kolmogorov smirnoff test to see, whether
  ## residuals are normally distributed around 0 with sigma = sd(residuals)
  ## > 0.05 says yes
  measures$normalityTestResiduals <- ks.test(x = na.omit(delta), y = pnorm, mean = 0, sd = sd(delta, na.rm = TRUE))$p.value
  
  return(measures)
}