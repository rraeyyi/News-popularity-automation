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
                 starts_with("weekday_is"),
                 "Positive_Word_Rate" = "global_rate_positive_words",
                 "Negative_Word_Rate" = "global_rate_negative_words",
                 "Title_Polarity" = "title_sentiment_polarity",
                 "Shares" = "shares")
  
news$Weekday <- NA
news$Weekday [news$weekday_is_monday==1] <- "Mon"
news$Weekday [news$weekday_is_tuesday==1] <- "Tues"
news$Weekday [news$weekday_is_wednesday==1] <- "Wed"
news$Weekday [news$weekday_is_thursday==1] <- "Thurs"
news$Weekday [news$weekday_is_friday==1] <- "Fri"
news$Weekday [news$weekday_is_saturday==1] <- "Sat"
news$Weekday [news$weekday_is_sunday==1] <- "Sun"
news$Weekday <- as.factor(news$Weekday)

news_final <- news %>%
  select(-starts_with("weekday_is"))

return(news_final)
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

    ## # A tibble: 2,099 × 9
    ##    Number_Title_Words Number_Content_Words Number_Images Number_Videos
    ##                 <dbl>                <dbl>         <dbl>         <dbl>
    ##  1                  8                  960            20             0
    ##  2                 10                  187             1             0
    ##  3                 11                  103             1             0
    ##  4                 10                  243             0             0
    ##  5                  8                  204             1             0
    ##  6                 11                  315             1             0
    ##  7                 10                 1190            20             0
    ##  8                  6                  374             1             0
    ##  9                 12                  499             1             0
    ## 10                 11                  223             0             0
    ## # … with 2,089 more rows, and 5 more variables: Positive_Word_Rate <dbl>,
    ## #   Negative_Word_Rate <dbl>, Title_Polarity <dbl>, Shares <dbl>, Weekday <fct>

## Split data into train and test

``` r
set.seed(216)
intrain <- createDataPartition(news_data$Shares, p = 0.7, list = FALSE)
training <- news_data[intrain,]
testing <- news_data[-intrain,]
```

# Summarization

``` r
# Summary stats for all variables
summary(training)
```

    ##  Number_Title_Words Number_Content_Words Number_Images     Number_Videos   
    ##  Min.   : 3.000     Min.   :   0.0       Min.   :  0.000   Min.   : 0.000  
    ##  1st Qu.: 8.000     1st Qu.: 298.8       1st Qu.:  1.000   1st Qu.: 0.000  
    ##  Median :10.000     Median : 498.0       Median :  1.000   Median : 0.000  
    ##  Mean   : 9.765     Mean   : 622.4       Mean   :  4.669   Mean   : 0.483  
    ##  3rd Qu.:11.000     3rd Qu.: 793.0       3rd Qu.:  8.000   3rd Qu.: 0.000  
    ##  Max.   :18.000     Max.   :8474.0       Max.   :111.000   Max.   :50.000  
    ##                                                                            
    ##  Positive_Word_Rate Negative_Word_Rate Title_Polarity        Shares      
    ##  Min.   :0.00000    Min.   :0.00000    Min.   :-1.0000   Min.   :    28  
    ##  1st Qu.:0.03475    1st Qu.:0.01040    1st Qu.: 0.0000   1st Qu.:  1100  
    ##  Median :0.04385    Median :0.01575    Median : 0.0000   Median :  1700  
    ##  Mean   :0.04446    Mean   :0.01657    Mean   : 0.1080   Mean   :  3687  
    ##  3rd Qu.:0.05333    3rd Qu.:0.02136    3rd Qu.: 0.2143   3rd Qu.:  3225  
    ##  Max.   :0.12139    Max.   :0.06180    Max.   : 1.0000   Max.   :196700  
    ##                                                                          
    ##   Weekday   
    ##  Fri  :214  
    ##  Mon  :222  
    ##  Sat  :122  
    ##  Sun  :155  
    ##  Thurs:247  
    ##  Tues :223  
    ##  Wed  :289

``` r
# Boxplot of weekday vs shares
ggplot(training, aes(x = Weekday, y = Shares)) +
  geom_boxplot(color = "royal blue") +
  scale_y_continuous(trans="log10")
```

![](Project3_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# Scatterplot of title length & polarity vs shares
ggplot(training, aes(x = Number_Title_Words, y=Shares)) + 
  geom_point(aes(color = Title_Polarity))
```

![](Project3_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

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

    ## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid mtry:
    ## reset to within valid range

    ## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid mtry:
    ## reset to within valid range

    ## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid mtry:
    ## reset to within valid range

    ## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid mtry:
    ## reset to within valid range

    ## Warning in randomForest.default(x, y, mtry = param$mtry, ...): invalid mtry:
    ## reset to within valid range

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
