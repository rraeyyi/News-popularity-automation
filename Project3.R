#Test sentence

# Load packages
library(readr)
library(dplyr)
library(caret)

# Data
## Read in and subset data
OnlineNewsPopularity <- read_csv("OnlineNewsPopularity.csv") 
OnlineNewsPopularity$url <- NULL
channel <- function(x){
  base <- "data_channel_is_"
  data <- paste0(base,x) %>% noquote()
  news <- OnlineNewsPopularity %>% filter(get(data) == 1)
  return(news)
}

lifestyle <- channel("lifestyle")
entertainment <- channel("entertainment")
bus <- channel("bus")
socmed <- channel("socmed")
tech <- channel("tech")
world <- channel("world")

## Split data into train and test
set.seed(216)
intrain <- createDataPartition(lifestyle$shares, p= 0.7, list = FALSE)
lifestyletrain <- lifestyle[intrain,]
lifestyletest <- lifestyle[-intrain,]

# Summarization

# Modeling
## Boosted tree model
tunegrid <- expand.grid(interaction.depth = 1:4,
                        n.trees = c(25, 50, 100, 150, 200),
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

gbmTune <- train(shares ~.,
                 data = lifestyletrain,
                 method = "gbm",
                 trControl = trainControl(method = "cv",
                                          number = 5),
                 tuneGrid = tunegrid)

# Comparison
## Apply model for prediction
pred_gbm <- predict(gbmTune,newdata = lifestyletest)

## Model performance
confusionMatrix(data = lifestyletest$shares,reference = pred_gbm)

# Automation
