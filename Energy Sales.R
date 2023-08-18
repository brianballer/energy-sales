## ----setup, include=FALSE-----------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

library(corrplot)
library(glmnet)
library(leaps)
library(gam)
library(tidyverse)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(ggplot2)
library(rattle)
library(knitr)


## -----------------------------------------------------------------------------------------
## import data
df <- read.delim("data/Midterm_FittingData.csv", quote=NULL, header=TRUE, stringsAsFactors = FALSE, na.strings=c("N/A","NaN", "na", "NA"))

## remove first row which contains the units of measurement
df <- df[-1,]

## structure of df shows that all columns which had units imported as characters
str(df)

## rename columns 1 and 35 (these columns contained weird quotes and had funny column names)
names(df)[1] <- "year"
names(df)[ncol(df)] <- "GSP"

## remove quote symbol from column entries
df$year <- as.integer(gsub('"', "", df$year))
df$GSP <- as.numeric(gsub('"', "", df$GSP))

## check for complete cases
df[which(complete.cases(df) == F),]
## Four rows contain one NA each.  Columns affected are MXSD, TSNW, DT00.  


## -----------------------------------------------------------------------------------------
## change all columns with class = chr to numeric
for (i in 1:ncol(df)) {
  if (class(df[,i]) == "character") {
    df[,i] <- as.numeric(df[,i])
  }
}



## -----------------------------------------------------------------------------------------
## check sum of df columns against sum of colummns in excel
 excel.check <- c(600600,1950,2816.03,2539591.51,2369.28,1922750.65,503.24,0.008947745,1409.552,0.0082021,26791.33,15427.27,24516.94,18973.3,21744.94,2104.48,0.489460784,0.308823529,167.41,1946.54,878.9,419.93,18422.11,2634.83,1972.01,3794.24,5945.34,30923.23,50215.12,2434817343,2284163532,150653811,1840.7,10626805.68,189633444)
r.check <- apply(df, 2, sum)
round(excel.check-r.check, 2)
## checks good with 3 NA columns (due to NAs in each column)



## -----------------------------------------------------------------------------------------
## How to handle the NAs?
par(mfrow=c(1,3))
hist(df$MXSD)
hist(df$TSNW)
hist(df$DT00)

## replace all NAs with median value of column
df <- apply(df, 2, function(x){replace(x, is.na(x), median(x, na.rm=T))})
df <- apply(df, 2, as.vector)
df <- as.data.frame(df)

## check NAs removed
which(complete.cases(df) == F)

## checksum
r.check <- apply(df, 2, sum)
round(excel.check-r.check, 2)


## -----------------------------------------------------------------------------------------
## explore data
par(mfrow=c(1,5))
for (i in 1:35) {
  boxplot(df[,i], main=colnames(df)[i])
}

## correlation between variables
M <- cor(df)
par(mfrow=c(1,1))
corrplot(M)
#write.csv(round(as.data.frame(M), 2), file = "Corr.csv")

plot(com.price ~ LABOR, df)

df <- subset(df, select = -c(MXSD,EMP,UNEMPRATE))


## -----------------------------------------------------------------------------------------
## setting up training and test sets
set.seed(234)
trainrows <- sample(nrow(df), 0.8*nrow(df))
tr.df <- df[trainrows,]
te.df <- df[-trainrows,]


## -----------------------------------------------------------------------------------------
rmse <- function(mod, newdata, response) {
  rmse <- sqrt(mean((predict(mod, newdata) - response)^2))
  return(rmse)
}


## -----------------------------------------------------------------------------------------
lm.mod <- lm(res.sales.adj ~., data=tr.df)
summary(lm.mod)
#Anova(lm.mod)


## -----------------------------------------------------------------------------------------
lm.mod <- lm(res.sales.adj ~.-com.sales.adj, data=tr.df)
summary(lm.mod)
#Anova(lm.mod)

err.lm <- c(rmse(lm.mod, tr.df, tr.df$res.sales.adj), rmse(lm.mod, te.df, te.df$res.sales.adj))
err.lm


## -----------------------------------------------------------------------------------------
train_x <- as.matrix(tr.df[,-c(4,6)])
train_y <- as.matrix(tr.df[,4])

test_x <- as.matrix(te.df[,-c(4,6)])
test_y <- as.matrix(te.df[,4])

## Pick the best LASSO regression model using built-in K-fold CV
set.seed(1)
cv_lasso <- cv.glmnet(train_x, train_y, alpha=1)

## Plot of MSE vs. lambda
plot(cv_lasso)

## Lambda with minimum MSE
cv_lasso$lambda.min

coef(cv_lasso, s = "lambda.min")

lasso.mod <- glmnet(train_x, train_y, alpha=1, lambda=45)

err.lasso <- c(rmse(lasso.mod, train_x, train_y), rmse(lasso.mod, test_x, test_y))
err.lasso


## -----------------------------------------------------------------------------------------
regfit.full=regsubsets(res.sales.adj ~ ., data=tr.df[,-6], nvmax = 30)
reg.summary <- summary(regfit.full)


## -----------------------------------------------------------------------------------------
plot(reg.summary$bic)
plot(reg.summary$adjr2)
plot(reg.summary$cp)


# plot(regfit.full,scale="r2")
# plot(regfit.full,scale="adjr2") 
# plot(regfit.full,scale="Cp")
# plot(regfit.full,scale="bic")

which.min(reg.summary$bic)
which.min(reg.summary$cp)
which.max(reg.summary$adjr2)

coef(regfit.full, 10)
coef(regfit.full, 15)

bestlm10.mod <- lm(res.sales.adj ~ month+MMNT+MNTM+DT90+DP05+WDSP+GUST+HTDD+CLDD+UNEMP, tr.df)
bestlm15.mod <- lm(res.sales.adj ~ year+month+res.price+EMNT+MMNT+MNTM+DT90+DX32+DT32+DP05+WDSP+GUST+HTDD+CLDD+LABOR, tr.df)

rmse(bestlm10.mod, tr.df, tr.df$res.sales.adj)
rmse(bestlm10.mod, te.df, te.df$res.sales.adj)

rmse(bestlm15.mod, tr.df, tr.df$res.sales.adj)
rmse(bestlm15.mod, te.df, te.df$res.sales.adj)


## -----------------------------------------------------------------------------------------
#step(lm(res.sales.adj ~ ., data=tr.df[,-6]), method="both", trace=TRUE)
#step(lm(res.sales.adj ~., tr.df[,-6]), scope=list(lower=~CLDD, upper=~.), method="forward", trace=TRUE)


## -----------------------------------------------------------------------------------------
## polynomial fit for the 10 predictor model
poly.mod <- lm(res.sales.adj  ~ poly(month, 4) + poly(MMNT, 4) +poly(MNTM, 4) +poly(DT90, 4) +poly(DP05, 4) +poly(WDSP, 4) +poly(GUST, 4) +poly(HTDD, 4) +poly(CLDD, 4) +poly(UNEMP, 4), tr.df)
summary(poly.mod)

err.poly <- c(rmse(poly.mod, tr.df, tr.df$res.sales.adj), rmse(poly.mod, te.df, te.df$res.sales.adj))
err.poly

## add-in loop for best i


## -----------------------------------------------------------------------------------------
#### just polynomial on the MMNT and MNTM term
polybest.mod <- lm(res.sales.adj ~ month+poly(MMNT,4)+poly(MNTM,4)+DT90+DP05+WDSP+GUST+HTDD+CLDD+UNEMP, tr.df)

rmse(polybest.mod, tr.df, tr.df$res.sales.adj)
rmse(polybest.mod, te.df, te.df$res.sales.adj)


## -----------------------------------------------------------------------------------------
gam.mod <- gam(res.sales.adj ~ s(month, df=5) + s(MMNT, df=5) +s(MNTM, df=5) +s(DT90, df=5) +s(DP05, df=5) +s(WDSP, df=5) +s(GUST, df=5) +s(HTDD, df=5) +s(CLDD, df=5) +s(UNEMP, df=5), data=tr.df)

err.gam <- c(rmse(gam.mod, tr.df, tr.df$res.sales.adj), rmse(gam.mod, te.df, te.df$res.sales.adj))
err.gam

## do step.Gam



## -----------------------------------------------------------------------------------------
start.mod <- gam(res.sales.adj ~.-com.sales.adj, data = tr.df)

list.scope <- list(
  "Year" = ~1 + year + s(year, df=2) + s(year, df=3) + s(year, df =4) + s(year, df=5) +lo(year),
  "month" = ~1 + month + s(month, df=2) + s(month, df=3) + s(month, df=4) + s(month, df=5) + lo(month),
  "res.price" = ~1 + res.price + s(res.price, df=2) + s(res.price, df=3) + s(res.price, df=4) + s(res.price, df=5) + lo(res.price),
  "com.price" = ~1 + com.price + s(com.price, df=2) + s(com.price, df=3) + s(com.price, df=4) + s(com.price, df=5) + lo(com.price),
  "EMXP" = ~1 + EMXP + s(EMXP, df=2) + s(EMXP, df=3) + s(EMXP, df=4) + s(EMXP, df=5) + lo(EMXP),
  "TPCP" = ~1 + TPCP + s(TPCP, df=2) + s(TPCP, df=3) + s(TPCP, df=4) + s(TPCP, df=5) + lo(TPCP),
  #"TSNW" = ~1 + TSNW + s(TSNW, df=2) + s(TSNW, df=3) + s(TSNW, df=4) + s(TSNW, df=5),
  "EMXT" = ~1 + EMXT + s(EMXT, df=2) + s(EMXT, df=3) + s(EMXT, df=4) + s(EMXT, df=5) + lo(EMXT),
  "EMNT" = ~1 + EMNT + s(EMNT, df=2) + s(EMNT, df=3) + s(EMNT, df=4) + s(EMNT, df=5) + lo(EMNT),
  "MMXT" = ~1 + MMXT + s(MMXT, df=2) + s(MMXT, df=3) + s(MMXT, df=4) + s(MMXT, df=5) + lo(MMXT),
  "MNTM" = ~1 + MNTM + s(MNTM, df=2) + s(MNTM, df=3) + s(MNTM, df=4) + s(MNTM, df=5) + lo(MNTM),
  "DT90" = ~1 + DT90 + s(DT90, df=2) + s(DT90, df=3) + s(DT90, df=4) + s(DT90, df=5) + lo(DT90),
  #"DX32" = ~1 + DX32 + s(DX32, df=2) + s(DX32, df=3) + s(DX32, df=4) + s(DX32, df=5),
  "DT32" = ~1 + DT32 + s(DT32, df=2) + s(DT32, df=3) + s(DT32, df=4) + s(DT32, df=5) + lo(DT32),
  "DP01" = ~1 + DP01 + s(DP01, df=2) + s(DP01, df=3) + s(DP01, df=4) + s(DP01, df=5) + lo(DP01),
  "DP05" = ~1 + DP05 + s(DP05, df=2) + s(DP05, df=3) + s(DP05, df=4) + s(DP05, df=5) + lo(DP05),
  "DP10" = ~1 + DP10 + s(DP10, df=2) + s(DP10, df=3) + s(DP10, df=4) + s(DP10, df=5) + lo(DP10),
  "MDPT" = ~1 + MDPT + s(MDPT, df=2) + s(MDPT, df=3) + s(MDPT, df=4) + s(MDPT, df=5) + lo(MDPT),
  "VISIB" = ~1 + VISIB + s(VISIB, df=2) + s(VISIB, df=3) + s(VISIB, df=4) + s(VISIB, df=5) +lo(VISIB),
  "WDSP" = ~1 + WDSP + s(WDSP, df=2) + s(WDSP, df=3) + s(WDSP, df=4) + s(WDSP, df=5) + lo(WDSP),
  "MWSPD" = ~1 + MWSPD + s(MWSPD, df=2) + s(MWSPD, df=3) + s(MWSPD, df=4) + s(MWSPD, df=5) +lo(MWSPD),
  "GUST" = ~1 + GUST + s(GUST, df=2) + s(GUST, df=3) + s(GUST, df=4) + s(GUST, df=5) + lo(GUST),
  "HTDD" = ~1 + HTDD + s(HTDD, df=2) + s(HTDD, df=3) + s(HTDD, df=4) + s(HTDD, df=5) + lo(HTDD),
  "CLDD" = ~1 + CLDD + s(CLDD, df=2) + s(CLDD, df=3) + s(CLDD, df=4) + s(CLDD, df=5) + lo(CLDD),
  "LABOR" = ~1 + LABOR + s(LABOR, df=2) + s(LABOR, df=3) + s(LABOR, df=4) + s(LABOR, df=5) + lo(LABOR),
  "UNEMP" = ~1 + UNEMP + s(UNEMP, df=2) + s(UNEMP, df=3) + s(UNEMP, df=4) + s(UNEMP, df=5) + lo(UNEMP),
  "PCINCOME" = ~1 + PCINCOME + s(PCINCOME, df=2) + s(PCINCOME, df=3) + s(PCINCOME, df=4) + s(PCINCOME, df=5) + lo(PCINCOME),
  "GSP" = ~1 + GSP + s(GSP, df=2) + s(GSP, df=3) + s(GSP, df=4) + s(GSP, df=5) + lo(GSP)
)

step.Gam(start.mod, list.scope, direction="forward", trace=1)


## -----------------------------------------------------------------------------------------

best.gam <- gam(formula = res.sales.adj ~ DT00 + year + s(month, df = 5) + 
    res.price + com.price + EMXP + TPCP + s(EMXT, df = 5) + EMNT + 
    MMXT + MNTM + DT90 + s(DT32, df = 5) + DP01 + DP05 + s(DP10, 
    df = 5) + MDPT + VISIB + WDSP + MWSPD + s(GUST, df = 2) + 
    s(HTDD, df = 3) + CLDD + LABOR + s(UNEMP, df = 3) + PCINCOME + 
    GSP, data = tr.df, trace = FALSE)

best.gam2 <- gam(formula = res.sales.adj ~ DT00 + s(month, df = 5) + res.price + 
    com.price + DT90 + s(DP10, df = 5) + WDSP + MWSPD + s(HTDD, 
    df = 5) + CLDD + LABOR + UNEMP + GSP, data = tr.df, trace = FALSE)

best.gam3 <- gam(formula = res.sales.adj ~ DT00 + year + s(month, df = 5) + 
    res.price + com.price + EMXP + TPCP + s(EMXT, df = 5) + EMNT + 
    MMXT + MNTM + DT90 + s(DT32, df = 5) + DP01 + DP05 + lo(DP10) + 
    MDPT + VISIB + WDSP + MWSPD + s(GUST, df = 2) + s(HTDD, df = 3) + 
    CLDD + LABOR + s(UNEMP, df = 3) + PCINCOME + GSP, data = tr.df, 
    trace = FALSE)

c(rmse(best.gam, tr.df, tr.df$res.sales.adj), rmse(best.gam, te.df, te.df$res.sales.adj))
c(rmse(best.gam2, tr.df, tr.df$res.sales.adj), rmse(best.gam2, te.df, te.df$res.sales.adj))
c(rmse(best.gam3, tr.df, tr.df$res.sales.adj), rmse(best.gam3, te.df, te.df$res.sales.adj))



## -----------------------------------------------------------------------------------------
set.seed(1)
rpart.mod <- rpart(res.sales.adj ~.-com.sales.adj, data=tr.df)
printcp(rpart.mod)
#plotcp(rpart.mod)
minCP <- rpart.mod$cptable[which.min(rpart.mod$cptable[,"xerror"]),"CP"]

rmse(rpart.mod, tr.df, tr.df$res.sales.adj)
rmse(rpart.mod, te.df, te.df$res.sales.adj)

## Prune tree to cp with minimum error
par(mfrow=c(1,2))
plotcp(rpart.mod)
rpart.mod <- prune(rpart.mod, cp=minCP) 

## Plot tree diagram
rpart.plot(rpart.mod, main="Rpart Tree")

err.rpart <- c(rmse(rpart.mod, tr.df, tr.df$res.sales.adj), rmse(rpart.mod, te.df, te.df$res.sales.adj))
err.rpart


## -----------------------------------------------------------------------------------------

## varied both mtry and ntree -- ntree seemed to be around 10-30; mtry ~ 5-10

# for(i in 1:30) {
#   rf.mod <- randomForest(res.sales.adj ~.-com.sales.adj, data=tr.df, mtry=6, ntree=2*i)
# 
#   #print(rmse(rf.mod, tr.df, tr.df$res.sales.adj))
#   print(rmse(rf.mod, te.df, te.df$res.sales.adj))
# }

rf.mod <- randomForest(res.sales.adj ~.-com.sales.adj, data=tr.df)
varImpPlot(rf.mod)

err.rf <- c(rmse(rf.mod, tr.df, tr.df$res.sales.adj), rmse(rf.mod, te.df, te.df$res.sales.adj))
err.rf



## -----------------------------------------------------------------------------------------
boost.mod <- gbm(res.sales.adj ~.-com.sales.adj, data=tr.df, distribution="gaussian",n.trees=5000, interaction.depth=2, shrinkage=.05)

rmse.boost <- function(mod, newdata, response) {
  rmse <- sqrt(mean((predict(mod, newdata, n.trees=500) - response)^2))
  return(rmse)
}

err.boost <- c(rmse.boost(boost.mod, tr.df, tr.df$res.sales.adj), rmse.boost(boost.mod, te.df, te.df$res.sales.adj))
err.boost



## -----------------------------------------------------------------------------------------
yhat.boost=predict(boost.mod,newdata=tr.df, n.trees=500)
sqrt(mean((yhat.boost - tr.df$res.sales.adj)^2))


## -----------------------------------------------------------------------------------------
err.df <- as.data.frame(rbind(err.lm, err.lasso, err.poly, err.gam, err.rpart, err.rf, err.boost))
err.df[order(err.df$V2),]


## -----------------------------------------------------------------------------------------
## rmse.mt(df, # folds, column number of y values, "rpart" or "rf" or "lm")

rmse.mt <- function(mod, df) {           
  
  nfolds <- 10
  
  set.seed(1)
  df$fold <- sample(1:nfolds, nrow(df), replace = TRUE)        ## adds a column that assigns each row to a fold
  tr.rmse <- vector() 
  te.rmse <- vector()              ## initializes vector for results
  
  for (i in 1:nfolds) {
    tr_df <- df[df$fold != i, -ncol(df)]
    te_df <- df[df$fold == i, -ncol(df)]
    # colnames(tr_df)[coly] <- "try"
    # colnames(te_df)[coly] <- "tey"
    
    if (mod == "rf") {
      model <- randomForest(try ~ SST + ENSO + NAO, data = tr_df)
      #print(model)
    }else if (mod == "rpart") {
      model <- rpart(try ~ SST + ENSO + NAO, data = tr_df)
      #plotcp(model)
    }else if (mod == "lm") {
      model <- lm(res.sales.adj ~.-com.sales.adj, data = tr_df)
    }else {
      stop('cv.q3(df, # folds, column number of y values, "rpart" or "rf" or "lm")')
    }
    
    yhat <- predict(model, newdata = te_df)
    
    te.rmse[i] <- sqrt(mean((yhat - te_df$res.sales.adj)^2)     )                 ## vector to store test MSEs
  }

  return(te.rmse)
}


## lm model MSE
rmse.lm <- rmse.mt("lm", df)
rmse.lm
mean(rmse.lm)
sd(rmse.lm)

