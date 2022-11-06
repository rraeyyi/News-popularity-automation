ST 558 Project 3
================
Rachel Hencher and Yi Ren
2022-11-02

# Load packages

``` r
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
```

# Data

## Read in and subset data

``` r
OnlineNewsPopularity <- read_csv("OnlineNewsPopularity.csv") 
```

    ## Rows: 39644 Columns: 61
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (1): url
    ## dbl (60): timedelta, n_tokens_title, n_tokens_content, n_unique_tokens, n_no...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
OnlineNewsPopularity$url <- NULL
channel <- function(x){
  base <- "data_channel_is_"
  data <- paste0(base,x) %>%  
          noquote()
  news <- OnlineNewsPopularity %>% 
          filter(get(data) == 1) %>% 
          select("Number_Title_Words" = "n_tokens_title",
                 "Number_Content_Words" = "n_tokens_content",
                 "Number_Images" = "num_imgs",
                 "Number_Videos" = "num_videos",
                 "Published_Mon" = ends_with("monday"),
                 "Published_Tues" = ends_with("tuesday"),
                 "Published_Weds" = ends_with("wednesday"),
                 "Published_Thurs" = ends_with("thursday"),
                 "Published_Fri" = ends_with("friday"),
                 "Published_Sat" = ends_with("saturday"),
                 "Published_Sun" = ends_with("sunday"),
                 "Positive_Word_Rate" = "global_rate_positive_words",
                 "Negative_Word_Rate" = "global_rate_negative_words",
                 "Title_Polarity" = "title_sentiment_polarity",
                 "Shares" = "shares")

return(news)
}
```

## Choose an option for the `channel` function argument:

- *lifestyle*: Is the desired data channel lifestyle?  
- *entertainment*: Is the desired data channel entertainment?  
- *bus*: Is the desired data channel business?  
- *socmed*: Is the desired data channel social media?
- *tech*: Is the desired data channel technology?  
- *world*: Is the desired data channel world?

``` r
news_data <- channel("lifestyle")
news_data
```

    ## # A tibble: 2,099 × 15
    ##    Number_Title_Words Number_Content_… Number_Images Number_Videos Published_Mon
    ##                 <dbl>            <dbl>         <dbl>         <dbl>         <dbl>
    ##  1                  8              960            20             0             1
    ##  2                 10              187             1             0             1
    ##  3                 11              103             1             0             1
    ##  4                 10              243             0             0             1
    ##  5                  8              204             1             0             1
    ##  6                 11              315             1             0             1
    ##  7                 10             1190            20             0             1
    ##  8                  6              374             1             0             1
    ##  9                 12              499             1             0             0
    ## 10                 11              223             0             0             0
    ## # … with 2,089 more rows, and 10 more variables: Published_Tues <dbl>,
    ## #   Published_Weds <dbl>, Published_Thurs <dbl>, Published_Fri <dbl>,
    ## #   Published_Sat <dbl>, Published_Sun <dbl>, Positive_Word_Rate <dbl>,
    ## #   Negative_Word_Rate <dbl>, Title_Polarity <dbl>, Shares <dbl>

## Split data into train and test

``` r
set.seed(216)
intrain <- createDataPartition(news_data$Shares, p = 0.7, list = FALSE)
training <- news_data[intrain,]
testing <- news_data[-intrain,]
```

# Summarization

``` r
# Title length & polarity vs shares
ggplot(training, aes(x = Number_Title_Words, y=Shares)) + 
  geom_point(aes(color = Title_Polarity))
```

![](Project3_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

# Modeling

## Random forest model

``` r
rf_model <- train(Shares ~ ., 
                  data = training, 
                  method = "rf", 
                  preProcess = c("center", "scale"), 
                  trControl = trainControl(method = "cv", number = 5), 
                  tuneGrid = expand.grid(mtry = c(1:14)))
```

## Boosted tree model

``` r
tunegrid <- expand.grid(interaction.depth = 1:4,
                        n.trees = c(25, 50, 100, 150, 200),
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

gbm_model <- train(Shares ~.,
                 data = training,
                 method = "gbm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 5),
                 tuneGrid = tunegrid)
```

# Comparison

## Apply model for prediction

``` r
rf_predict <- predict(rf_model, newdata = testing)
```

``` r
gbm_predict <- predict(gbm_model, newdata = testing)
```

## Model performance

``` r
confusionMatrix(data = testing$Shares, reference = rf_predict)
```

``` r
confusionMatrix(data = testing$Shares, reference = gbm_predict)
```

# Automation
