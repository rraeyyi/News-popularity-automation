ST 558 Project 3
================
Rachel Hencher and Yi Ren
2022-11-04

# Load packages

``` r
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(ggcorrplot)
```

# Introduction

# Data

## Read in data

``` r
news <- read_csv("./OnlineNewsPopularity.csv") 
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
news$url <- NULL

news$channel <- as.factor(ifelse(news$data_channel_is_lifestyle == 1, "lifestyle",
                 ifelse(news$data_channel_is_entertainment == 1, "entertainment", 
                        ifelse(news$data_channel_is_bus == 1, "bus", 
                               ifelse(news$data_channel_is_socmed , "socmed",
                                      ifelse(news$data_channel_is_tech == 1, "tech", "world"))))))

news$weekday <- as.factor(ifelse(news$weekday_is_monday == 1, "monday",
                                 ifelse(news$weekday_is_tuesday == 1, "tuesday", 
                                        ifelse(news$weekday_is_wednesday == 1, "wednesday", 
                                               ifelse(news$weekday_is_thursday , "thursday",
                                                      ifelse(news$weekday_is_friday == 1, "friday",
                                                             ifelse(news$weekday_is_saturday == 1, "saturday", "sunday")))))))
```

## Split data into train and test

``` r
set.seed(216)
intrain <- createDataPartition(news$shares, p= 0.7, list = FALSE)
newstrain <- news[intrain,]
newstest <- news[-intrain,]
```

# Summarization

## Bar plot for weekday and channel

``` r
ggplot(newstrain, aes(x = channel, fill = weekday)) +
  geom_bar(position="dodge")
```

![](Project3---Rachel---Yi_files/figure-gfm/barplot-1.png)<!-- -->

From the bar plot, we can see lifestyle and socmed are less popular than
other channels. Unexpectedly, Wednesday, Tuesday and Sunday seems have
the highest page view in a week.

## Correlations between variables

``` r
newscor <- newstrain %>% select(-weekday, -channel)
cor <- cor(newstrain[,c(1:60)])
p.mat <- cor_pmat(newscor)
ggcorrplot(cor, p.mat = p.mat, hc.order = TRUE,
           type = "lower", insig = "blank")
```

![](Project3---Rachel---Yi_files/figure-gfm/cor-1.png)<!-- -->

In order to obtain a better model, we want to eliminated the effects
come from internal correlations. For example, n_unique_tokens and
n_non_stop_unique_tokens have the significantly high correlations
(0.94), which shouldn’t exist in the model simultaneously.

## Box plot for negative and positive polarity

``` r
par(mfrow=c(1,3))
boxplot(newstrain$max_negative_polarity)
boxplot(newstrain$min_negative_polarity)
boxplot(newstrain$avg_negative_polarity)
```

![](Project3---Rachel---Yi_files/figure-gfm/boxplot-1.png)<!-- -->

``` r
boxplot(newstrain$max_positive_polarity)
boxplot(newstrain$min_positive_polarity)
boxplot(newstrain$avg_positive_polarity)
```

![](Project3---Rachel---Yi_files/figure-gfm/boxplot-2.png)<!-- -->

Box plot is one of the most intuitive way to check outliers. Using
negative and positive polarity as an example, we can see the
min_negative_polarity and max_positive_polarity should under
consideration, with no outliers in the box plots. Similarly with
kw_max_avg and kw_avg_min.

# Modeling

## Set up cross validation

``` r
control<-trainControl(method="repeatedcv",number=3,repeats=5)
```

## Linear regression model

``` r
lasso <- train(shares ~ timedelta + n_tokens_title + kw_max_avg + avg_negative_polarity + weekday + channel,
                   data = newstrain,
                   method='lasso',
                   preProcess = c("center", "scale"),
                   trControl = control)
predict(lasso$finalModel, type = "coef")
```

    ## $s
    ##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    ## 
    ## $fraction
    ##  [1] 0.00000000 0.05263158 0.10526316 0.15789474 0.21052632 0.26315789
    ##  [7] 0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737
    ## [13] 0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684
    ## [19] 0.94736842 1.00000000
    ## 
    ## $mode
    ## [1] "step"
    ## 
    ## $coefficients
    ##    timedelta n_tokens_title kw_max_avg avg_negative_polarity weekdaymonday
    ## 0    0.00000       0.000000     0.0000               0.00000      0.000000
    ## 1    0.00000       0.000000   356.2945               0.00000      0.000000
    ## 2    0.00000       0.000000   432.4607               0.00000      0.000000
    ## 3    0.00000       0.000000   474.7436             -44.23427      0.000000
    ## 4    0.00000       0.000000   564.1926            -138.14461      0.000000
    ## 5    0.00000       2.050421   566.0277            -140.05505      0.000000
    ## 6   31.00573      29.445871   585.2552            -158.54740      0.000000
    ## 7   47.26178      45.767002   595.8945            -170.21590      0.000000
    ## 8   52.05961      50.843489   599.0392            -173.59196      0.000000
    ## 9  104.30132     106.345619   633.2466            -210.17169      0.000000
    ## 10 106.42146     108.577946   634.6255            -211.65342      2.055923
    ## 11 116.23884     118.876816   641.0188            -218.50126     10.795466
    ## 12 137.84778     142.522503   654.0817            -232.97543     30.276698
    ## 13 148.39308     154.441265   661.0944            -240.68663     40.167231
    ## 14 150.18781     156.476983   662.2900            -241.99690     41.550167
    ## 15 152.36951     158.940378   663.7379            -243.62187     45.129095
    ## 16 185.41957     196.357127   685.7058            -268.14692     92.143246
    ## 17 188.81634     200.124839   687.9369            -270.73905    102.618641
    ## 18 189.73966     201.142416   688.5482            -271.42825    104.509762
    ## 19 190.40899     201.891411   688.9830            -271.95443    107.530254
    ##    weekdaysaturday weekdaysunday weekdaythursday weekdaytuesday
    ## 0          0.00000       0.00000        0.000000      0.0000000
    ## 1          0.00000       0.00000        0.000000      0.0000000
    ## 2          0.00000       0.00000        0.000000      0.0000000
    ## 3          0.00000       0.00000        0.000000      0.0000000
    ## 4         97.99703       0.00000        0.000000      0.0000000
    ## 5        100.05682       0.00000        0.000000      0.0000000
    ## 6        120.47058       0.00000        0.000000      0.0000000
    ## 7        131.45726       0.00000        0.000000      0.0000000
    ## 8        134.67433       0.00000        0.000000      0.0000000
    ## 9        172.70363      37.83761        0.000000      0.0000000
    ## 10       174.49357      39.63699        0.000000      0.0000000
    ## 11       182.27917      47.43800       -3.105825      0.0000000
    ## 12       198.96469      64.19726       -9.989916      0.0000000
    ## 13       206.66063      72.51288      -13.059114      0.0000000
    ## 14       207.77840      73.72516      -13.899444     -0.8857072
    ## 15       210.35584      76.48951      -12.979586      0.0000000
    ## 16       244.78498     113.48134       -6.523739      0.0000000
    ## 17       251.94047     121.11070        0.000000     10.5134688
    ## 18       253.27186     122.53478        0.000000     12.3779421
    ## 19       255.29548     124.68789        3.058795     15.4429345
    ##    weekdaywednesday channelentertainment channellifestyle channelsocmed
    ## 0          0.000000              0.00000          0.00000      0.000000
    ## 1          0.000000              0.00000          0.00000      0.000000
    ## 2          0.000000              0.00000          0.00000      0.000000
    ## 3          0.000000              0.00000          0.00000      0.000000
    ## 4          0.000000              0.00000          0.00000      0.000000
    ## 5          0.000000              0.00000          0.00000      0.000000
    ## 6          0.000000              0.00000          0.00000      0.000000
    ## 7          0.000000            -11.11916          0.00000      0.000000
    ## 8          0.000000            -13.69495          0.00000      3.760387
    ## 9          0.000000            -42.67559          0.00000     45.032042
    ## 10         0.000000            -43.89598          0.00000     46.734528
    ## 11         0.000000            -49.54441          0.00000     54.647520
    ## 12         0.000000            -57.85137         18.49003     74.870627
    ## 13         0.000000            -49.44090         35.11223     92.581887
    ## 14         0.000000            -47.99933         37.94762     95.615178
    ## 15         3.481683            -46.20092         41.42854     99.335669
    ## 16        48.731779            -18.95074         94.15333    155.822053
    ## 17        59.253812            -16.15517         99.57765    161.529159
    ## 18        61.119523            -15.41356        101.03916    163.083479
    ## 19        64.187293            -14.84441        102.12096    164.205058
    ##    channeltech channelworld
    ## 0      0.00000      0.00000
    ## 1      0.00000      0.00000
    ## 2      0.00000     76.16615
    ## 3      0.00000    119.24370
    ## 4      0.00000    211.32184
    ## 5      0.00000    213.17480
    ## 6      0.00000    235.23828
    ## 7      0.00000    243.21139
    ## 8      0.00000    246.56244
    ## 9      0.00000    282.02288
    ## 10     0.00000    283.49214
    ## 11     0.00000    290.30055
    ## 12     0.00000    310.83869
    ## 13    24.11576    336.91196
    ## 14    28.24440    341.36725
    ## 15    33.29231    346.88267
    ## 16   109.95186    430.40529
    ## 17   117.68175    439.01228
    ## 18   119.76502    441.31969
    ## 19   121.30607    443.04785

``` r
lasso$bestTune
```

    ##   fraction
    ## 3      0.9

## Boosted tree model

``` r
boostedtree <- train(shares ~ timedelta + n_tokens_title + kw_max_avg + avg_negative_polarity + weekday + channel,
                     data = newstrain,
                     method = "gbm",
                     trControl = control,
                     preProcess = c("center", "scale"),
                     verbose = FALSE)
boostedtree
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 27752 samples
    ##     6 predictor
    ## 
    ## Pre-processing: centered (15), scaled (15) 
    ## Resampling: Cross-Validated (3 fold, repeated 5 times) 
    ## Summary of sample sizes: 18501, 18502, 18501, 18502, 18501, 18501, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   50      11152.56  0.013942085  3124.338
    ##   1                  100      11151.81  0.014102367  3123.233
    ##   1                  150      11153.41  0.013870312  3126.139
    ##   2                   50      11178.55  0.010538503  3136.125
    ##   2                  100      11186.84  0.010461844  3136.705
    ##   2                  150      11205.23  0.009419685  3151.882
    ##   3                   50      11215.46  0.008390016  3144.200
    ##   3                  100      11248.20  0.007459782  3160.782
    ##   3                  150      11296.65  0.006221657  3182.518
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

# Comparison

## Apply model for prediction

``` r
pred_lasso <- predict(lasso, newdata = newstest)
pred_boostedtree <- predict(boostedtree, newdata = newstest)
```

## Model performance

\#`{r perform} #postResample(pred_lasso, obs = newstest$shares) #confusionMatrix(data = newstest$shares, reference = pred_boostedtree) #`

# Automation
