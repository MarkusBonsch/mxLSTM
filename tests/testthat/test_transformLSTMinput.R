context("Check transformLSTMinput function")

test_that("Result correctness", {
  
  #######################################################
  # test1 isLabel == FALSE, single column data
  dat <- data.frame(x = 1:10,
                    y  = 21:30)
  
  targetX <- array(dat$x, dim = c(1,4,2))
  targetY <- array(dat$y, dim = c(1,4,2))

  expect_lt(max(abs(transformLSTMinput(dat, targetColumn = "y", seq.length = 4)$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, targetColumn = "y", seq.length = 4)$y - targetY)), 1e-7)

  # test1
  ########################################################

  #######################################################
  # test2 three column input data
  dat <- data.frame(x1 = 1:10,
                    x2 = 11:20,
                    x3 = 21:30,
                    y  = 31:40)
  
  targetX <- array(dat$x1, dim = c(3,4,2))
  targetX[,1,1] <- c(1,11,21)
  targetX[,2,1] <- c(2,12,22)
  targetX[,3,1] <- c(3,13,23)
  targetX[,4,1] <- c(4,14,24)
  targetX[,1,2] <- c(5,15,25)
  targetX[,2,2] <- c(6,16,26)
  targetX[,3,2] <- c(7,17,27)
  targetX[,4,2] <- c(8,18,28)  
  
  targetY <- array(dat$y, dim = c(1,4,2))
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test2
  ########################################################
  
  #######################################################
  # test3 seq.freq > seq.length; incomplete last sequence
  dat <- data.frame(x1 = 1:10,
                    x2 = 11:20,
                    x3 = 21:30,
                    y  = 31:40)
  
  targetX <- array(dat$x1, dim = c(3,3,2))
  targetX[,1,1] <- c(1,11,21)
  targetX[,2,1] <- c(2,12,22)
  targetX[,3,1] <- c(3,13,23)
  targetX[,1,2] <- c(6,16,26)
  targetX[,2,2] <- c(7,17,27)
  targetX[,3,2] <- c(8,18,28)

  targetY <- array(dat$y, dim = c(1,3,2))
  targetY[1,1,1] <- 31
  targetY[1,2,1] <- 32
  targetY[1,3,1] <- 33
  targetY[1,1,2] <- 36
  targetY[1,2,2] <- 37
  targetY[1,3,2] <- 38
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 3, seq.freq = 5, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 3, seq.freq = 5, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test3
  ########################################################
  
  #######################################################
  # test4 seq.freq > seq.length; overcomplete last sequence
  dat <- data.frame(x1 = 1:14,
                    x2 = 11:24,
                    x3 = 21:34,
                    y  = 31:44)
  
  targetX <- array(dat$x1, dim = c(3,3,3))
  targetX[,1,1] <- c(1,11,21)
  targetX[,2,1] <- c(2,12,22)
  targetX[,3,1] <- c(3,13,23)
  targetX[,1,2] <- c(6,16,26)
  targetX[,2,2] <- c(7,17,27)
  targetX[,3,2] <- c(8,18,28)
  targetX[,1,3] <- c(11,21,31)
  targetX[,2,3] <- c(12,22,32)
  targetX[,3,3] <- c(13,23,33)
  
    
  targetY <- array(dat$y, dim = c(1,3,3))
  targetY[1,1,1] <- 31
  targetY[1,2,1] <- 32
  targetY[1,3,1] <- 33
  targetY[1,1,2] <- 36
  targetY[1,2,2] <- 37
  targetY[1,3,2] <- 38
  targetY[1,1,3] <- 41
  targetY[1,2,3] <- 42
  targetY[1,3,3] <- 43
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 3, seq.freq = 5, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 3, seq.freq = 5, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test4
  ########################################################
  

  #######################################################
  # test5 seq.freq < seq.length; uneven seq.freq and seq.length
  dat <- data.frame(x1 = 1:20,
                    x2 = 21:40,
                    x3 = 41:60,
                    y  = 61:80)
  
  targetX <- array(dat$x1, dim = c(3,7,5))
  targetX[,1,1] <- c(1,21,41)
  targetX[,2,]  <- targetX[,1,] + 1
  targetX[,3,]  <- targetX[,1,] + 2
  targetX[,4,]  <- targetX[,1,] + 3
  targetX[,5,]  <- targetX[,1,] + 4
  targetX[,6,]  <- targetX[,1,] + 5
  targetX[,7,]  <- targetX[,1,] + 6
  
  targetX[,,2]   <- targetX[,,1] + 3
  targetX[,,3]   <- targetX[,,1] + 6
  targetX[,,4]   <- targetX[,,1] + 9
  targetX[,,5]   <- targetX[,,1] + 12

  targetY <- array(dat$y, dim = c(1,7,5))
  targetY[1,1,1] <- 61
  targetY[1,2,1] <- 62
  targetY[1,3,1] <- 63
  targetY[1,4,1] <- 64
  targetY[1,5,1] <- 65
  targetY[1,6,1] <- 66
  targetY[1,7,1] <- 67
  
  targetY[1,,2]  <- targetY[1,,1] + 3
  targetY[1,,3]  <- targetY[1,,1] + 6
  targetY[1,,4]  <- targetY[1,,1] + 9
  targetY[1,,5]  <- targetY[1,,1] + 12

  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 7, seq.freq = 3, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 7, seq.freq = 3, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test5
  ########################################################
  
  #######################################################
  # test6 seq.freq < seq.length; even seq.freq and seq.length
  dat <- data.frame(x1 = 1:20,
                    x2 = 21:40,
                    x3 = 41:60,
                    y  = 61:80)
  
  targetX <- array(dat$x1, dim = c(3,8,4))
  targetX[,1,1] <- c(1,21,41)
  targetX[,2,]  <- targetX[,1,] + 1
  targetX[,3,]  <- targetX[,1,] + 2
  targetX[,4,]  <- targetX[,1,] + 3
  targetX[,5,]  <- targetX[,1,] + 4
  targetX[,6,]  <- targetX[,1,] + 5
  targetX[,7,]  <- targetX[,1,] + 6
  targetX[,8,]  <- targetX[,1,] + 7
  
  targetX[,,2]   <- targetX[,,1] + 4
  targetX[,,3]   <- targetX[,,1] + 8
  targetX[,,4]   <- targetX[,,1] + 12
  
  
  targetY <- array(dat$y, dim = c(1,8,4))
  targetY[1,1,1] <- 61
  targetY[1,2,1] <- 62
  targetY[1,3,1] <- 63
  targetY[1,4,1] <- 64
  targetY[1,5,1] <- 65
  targetY[1,6,1] <- 66
  targetY[1,7,1] <- 67
  targetY[1,8,1] <- 68
  
  targetY[1,,2]  <- targetY[1,,1] + 4
  targetY[1,,3]  <- targetY[1,,1] + 8
  targetY[1,,4]  <- targetY[1,,1] + 12

  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 8, seq.freq = 4, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 8, seq.freq = 4, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test6
  ########################################################
  
  #######################################################
  # test7 seq.freq < seq.length; even seq.length and uneven seq.freq
  dat <- data.frame(x1 = 1:20,
                    x2 = 21:40,
                    x3 = 41:60,
                    y  = 61:80)
  
  targetX <- array(dat$x1, dim = c(3,8,5))
  targetX[,1,1] <- c(1,21,41)
  targetX[,2,]  <- targetX[,1,] + 1
  targetX[,3,]  <- targetX[,1,] + 2
  targetX[,4,]  <- targetX[,1,] + 3
  targetX[,5,]  <- targetX[,1,] + 4
  targetX[,6,]  <- targetX[,1,] + 5
  targetX[,7,]  <- targetX[,1,] + 6
  targetX[,8,]  <- targetX[,1,] + 7
  
  targetX[,,2]   <- targetX[,,1] + 3
  targetX[,,3]   <- targetX[,,1] + 6
  targetX[,,4]   <- targetX[,,1] + 9
  targetX[,,5]   <- targetX[,,1] + 12
  
  
  targetY <- array(dat$y, dim = c(1,8,5))
  targetY[1,1,1] <- 61
  targetY[1,2,1] <- 62
  targetY[1,3,1] <- 63
  targetY[1,4,1] <- 64
  targetY[1,5,1] <- 65
  targetY[1,6,1] <- 66
  targetY[1,7,1] <- 67
  targetY[1,8,1] <- 68
  
  targetY[1,,2]  <- targetY[1,,1] + 3
  targetY[1,,3]  <- targetY[1,,1] + 6
  targetY[1,,4]  <- targetY[1,,1] + 9
  targetY[1,,5]  <- targetY[1,,1] + 12
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 8, seq.freq = 3, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 8, seq.freq = 3, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test7
  ########################################################
  
  #######################################################
  # test8 multiple y
  dat <- data.frame(x1 = 1:10,
                    x2 = 11:20,
                    y1 = 21:30,
                    y2 = 31:40)
  
  targetX <- array(dat$x1, dim = c(2,4,2))
  targetX[,1,1] <- c(1,11)
  targetX[,2,1] <- c(2,12)
  targetX[,3,1] <- c(3,13)
  targetX[,4,1] <- c(4,14)
  targetX[,1,2] <- c(5,15)
  targetX[,2,2] <- c(6,16)
  targetX[,3,2] <- c(7,17)
  targetX[,4,2] <- c(8,18)  
  
  targetY <- array(dat$y1, dim = c(2,4,2))
  targetY[,1,1] <- c(21,31)
  targetY[,2,1] <- c(22,32)
  targetY[,3,1] <- c(23,33)
  targetY[,4,1] <- c(24,34)
  targetY[,1,2] <- c(25,35)
  targetY[,2,2] <- c(26,36)
  targetY[,3,2] <- c(27,37)
  targetY[,4,2] <- c(28,38)  
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = c("y1", "y2"))$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = c("y1", "y2"))$y - targetY)), 1e-7)
  
  # test8
  ########################################################
  
  #######################################################
  # test9 variables not in alphabetical order
  dat <- data.frame(x3 = 1:10,
                    x2 = 11:20,
                    x1 = 21:30,
                    y  = 31:40)
  
  targetX <- array(dat$x1, dim = c(3,4,2))
  targetX[,1,1] <- c(1,11,21)
  targetX[,2,1] <- c(2,12,22)
  targetX[,3,1] <- c(3,13,23)
  targetX[,4,1] <- c(4,14,24)
  targetX[,1,2] <- c(5,15,25)
  targetX[,2,2] <- c(6,16,26)
  targetX[,3,2] <- c(7,17,27)
  targetX[,4,2] <- c(8,18,28)  
  
  targetY <- array(dat$y, dim = c(1,4,2))
  
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = "y")$x - targetX)), 1e-7)
  expect_lt(max(abs(transformLSTMinput(dat, seq.length = 4, targetColumn = "y")$y - targetY)), 1e-7)
  
  # test9
  ########################################################
  
  
})

test_that("Error handling", {
  
})


context("Check caret2lstmInput and lstmInput2caret functions")

test_that("Result correctness", {
  
  #######################################################
  # test1 Check that transformation and inverse transformation are identity
  
  dat <- 
    data.table(x1 = 1:10,
               x2 = 11:20,
               x3 = 21:30,
               y  = 31:40) %>% 
    transformLSTMinput(targetColumn = "y", seq.length = 4)
  
  expect_equal(caret2lstmInput(lstmInput2caret(dat)), dat)
    
  
  # test1
  ########################################################
  
})
