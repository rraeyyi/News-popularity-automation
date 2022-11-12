Lifestyle Analysis
================
Rachel Hencher and Yi Ren
2022-11-02

- <a href="#introduction" id="toc-introduction">Introduction</a>
- <a href="#load-packages" id="toc-load-packages">Load packages</a>
- <a href="#data" id="toc-data">Data</a>
  - <a href="#read-in-and-subset-data" id="toc-read-in-and-subset-data">Read
    in and subset data</a>
  - <a href="#automation" id="toc-automation">Automation</a>
  - <a href="#split-data-into-train-and-test"
    id="toc-split-data-into-train-and-test">Split data into train and
    test</a>
- <a href="#summarization" id="toc-summarization">Summarization</a>
  - <a href="#numeric-summaries" id="toc-numeric-summaries">Numeric
    summaries</a>
  - <a href="#pairs-plot" id="toc-pairs-plot">Pairs plot</a>
  - <a href="#barplot-for-weekday" id="toc-barplot-for-weekday">Barplot for
    weekday</a>
  - <a href="#boxplot-of-weekday-vs-shares"
    id="toc-boxplot-of-weekday-vs-shares">Boxplot of weekday vs shares</a>
  - <a href="#scatterplot-of-title-length--polarity-vs-shares"
    id="toc-scatterplot-of-title-length--polarity-vs-shares">Scatterplot of
    title length &amp; polarity vs shares</a>
  - <a href="#scatterplots-of-negative--positive-word-rate-vs-shares"
    id="toc-scatterplots-of-negative--positive-word-rate-vs-shares">Scatterplots
    of negative &amp; positive word rate vs shares</a>
- <a href="#modeling" id="toc-modeling">Modeling</a>
  - <a href="#set-up-cross-validation" id="toc-set-up-cross-validation">Set
    up cross validation</a>
  - <a href="#lasso-model" id="toc-lasso-model">LASSO model</a>
  - <a href="#forward-stepwise-model"
    id="toc-forward-stepwise-model">Forward stepwise model</a>
  - <a href="#random-forest-model" id="toc-random-forest-model">Random
    forest model</a>
  - <a href="#boosted-tree-model" id="toc-boosted-tree-model">Boosted tree
    model</a>
- <a href="#comparison" id="toc-comparison">Comparison</a>
  - <a href="#apply-model-for-prediction"
    id="toc-apply-model-for-prediction">Apply model for prediction</a>
  - <a href="#model-performance" id="toc-model-performance">Model
    performance</a>
    - <a href="#best-model-by-rmse-criteria"
      id="toc-best-model-by-rmse-criteria">Best model by RMSE criteria</a>
    - <a href="#best-model-by-rsquared-criteria"
      id="toc-best-model-by-rsquared-criteria">Best model by Rsquared
      criteria</a>

# Introduction

This report analyzes data on almost 40,000 articles published by
Mashable throughout the years 2013 and 2014. Although the original data
set includes information on 61 different features about the articles,
this report excludes some of those and condenses others so that we may
focus on the following 10 variables:

| Name                 | Definition                                                                       |
|:---------------------|:---------------------------------------------------------------------------------|
| Channel              | Data channel is Lifestyle, Entertainment, Business, Social Media, Tech, or World |
| Number_Title_Words   | Number of words in the title                                                     |
| Number_Content_Words | Number of words in the content                                                   |
| Number_Images        | Number of images                                                                 |
| Number_Videos        | Number of videos                                                                 |
| Positive_Word_Rate   | Rate of positive words in the content                                            |
| Negative_Word_Rate   | Rate of negative words in the content                                            |
| Title_Polarity       | Title polarity                                                                   |
| Weekday              | Weekday published                                                                |
| Shares               | Number of shares                                                                 |

The purpose of this report is to look for patterns and to make
predictions regarding the number of shares for articles in one of six
different channels. Following some exploratory data analysis, four
different models are used to model the response: a LASSO regression
model, a forward stepwise regression model, a random forest model, and a
boosted tree model.

# Load packages

``` r
library(readr)
library(dplyr)
library(knitr)
library(caret)
library(ggplot2)
library(GGally)
library(ggpubr)
```

# Data

## Read in and subset data

``` r
OnlineNewsPopularity <- read_csv("OnlineNewsPopularity.csv") 
OnlineNewsPopularity$url <- NULL

news <- OnlineNewsPopularity %>% 
  select("Number_Title_Words" = "n_tokens_title",
         "Number_Content_Words" = "n_tokens_content",
         "Number_Images" = "num_imgs",
         "Number_Videos" = "num_videos",
         starts_with("weekday_is"),
         starts_with("data_channel_is"),
         "Positive_Word_Rate" = "global_rate_positive_words",
         "Negative_Word_Rate" = "global_rate_negative_words",
         "Title_Polarity" = "title_sentiment_polarity",
         "Shares" = "shares")
  
news$Weekday <- as.factor(ifelse(news$weekday_is_monday == 1, "Monday",
                                 ifelse(news$weekday_is_tuesday == 1, "Tuesday", 
                                        ifelse(news$weekday_is_wednesday == 1, "Wednesday", 
                                               ifelse(news$weekday_is_thursday , "Thursday",
                                                      ifelse(news$weekday_is_friday == 1, "Friday",
                                                             ifelse(news$weekday_is_saturday == 1, "Saturday", "Sunday")))))))

news$Channel <- as.factor(ifelse(news$data_channel_is_lifestyle == 1, "Lifestyle",
                                 ifelse(news$data_channel_is_entertainment == 1, "Entertainment", 
                                        ifelse(news$data_channel_is_bus == 1, "Bus", 
                                               ifelse(news$data_channel_is_socmed , "Socmed",
                                                      ifelse(news$data_channel_is_tech == 1, "Tech", "World"))))))
news_final <- news %>%
  select(-c(starts_with("weekday_is"), starts_with("data_channel_is")))
```

## Automation

``` r
news_data <- news_final %>% 
  filter(news_final$Channel == params$channel) %>% 
  select(-Channel)
```

## Split data into train and test

``` r
set.seed(216)
intrain <- createDataPartition(news_data$Shares, p = 0.7, list = FALSE)

training <- news_data[intrain,]
testing <- news_data[-intrain,]
```

# Summarization

## Numeric summaries

The following table displays five-number summaries for each of the
numeric variables explored.

``` r
stat <- training %>% 
  select(Number_Title_Words,
         Number_Content_Words,
         Number_Images,
         Number_Videos,
         Positive_Word_Rate,
         Negative_Word_Rate,
         Title_Polarity,
         Shares) %>% 
  apply(2, function(x){summary(x[!is.na(x)])}) 

kable(stat, caption = "Summary Stats for Numeric Variables", digits = 2)
```

|         | Number_Title_Words | Number_Content_Words | Number_Images | Number_Videos | Positive_Word_Rate | Negative_Word_Rate | Title_Polarity |    Shares |
|:--------|-------------------:|---------------------:|--------------:|--------------:|-------------------:|-------------------:|---------------:|----------:|
| Min.    |               3.00 |                 0.00 |          0.00 |          0.00 |               0.00 |               0.00 |          -1.00 |     28.00 |
| 1st Qu. |               8.00 |               298.75 |          1.00 |          0.00 |               0.03 |               0.01 |           0.00 |   1100.00 |
| Median  |              10.00 |               498.00 |          1.00 |          0.00 |               0.04 |               0.02 |           0.00 |   1700.00 |
| Mean    |               9.76 |               622.35 |          4.67 |          0.48 |               0.04 |               0.02 |           0.11 |   3687.17 |
| 3rd Qu. |              11.00 |               793.00 |          8.00 |          0.00 |               0.05 |               0.02 |           0.21 |   3225.00 |
| Max.    |              18.00 |              8474.00 |        111.00 |         50.00 |               0.12 |               0.06 |           1.00 | 196700.00 |

Summary Stats for Numeric Variables

## Pairs plot

The following graphic displays the correlation between each of the
variables explored. There are several things to look out for… The
correlation between `Shares`, our response, and each of the other
variables, our predictors. A value close to -1 or 1 indicates the two
variables are highly correlated. A value close to 0 indicates little to
no correlation. Additionally, we should consider correlation between two
predictor variables as well. A high correlation between two predictor
variables is an indication of collinearity, which should be taken into
account when creating models later.

``` r
training_sub <- training %>% 
  select(-Weekday)

ggpairs(training_sub)
```

![](Project3_files/figure-gfm/ggpairs-1.png)<!-- -->

## Barplot for weekday

The following barplot displays counts for how many articles in a
particular channel were published each day of the week over the time
frame covered by the data set.

``` r
ggplot(training, aes(x = Weekday)) +
  geom_bar(fill = "medium blue", position = "dodge") +
  labs(y = "Count")
```

![](Project3_files/figure-gfm/barplot-1.png)<!-- -->

## Boxplot of weekday vs shares

The following boxplots display a five-number summary of shares for each
day of the week. The axes are flipped, so if we wish to draw conclusions
regarding which day may be best to maximize shares, we would look for a
boxplot with a median furthest to the right.

``` r
ggplot(training, aes(x = Weekday, y = Shares)) +
  geom_boxplot(color = "royal blue") +
  coord_flip() +
  scale_y_continuous(trans = "log10")
```

![](Project3_files/figure-gfm/boxplot-1.png)<!-- -->

## Scatterplot of title length & polarity vs shares

The following scatterplot displays the number of shares for a given
title length. The peak of the data, excluding outliers, indicates the
title length that maximizes the number of shares. Additionally, the key
displays the color coding for polarity of the title so that we can look
for patterns to see whether the polarity of the title also has an effect
on the number of shares.

``` r
ggplot(training, aes(x = Number_Title_Words, y = Shares)) + 
  geom_point(aes(color = Title_Polarity))
```

![](Project3_files/figure-gfm/scatterplot-1.png)<!-- -->

## Scatterplots of negative & positive word rate vs shares

The following two scatterplots compare the number of shares for a given
positive word rate and negative word rate. The two graphs have been
scaled the same so that they can be compared. If the data appears to
peak further to the right on the positive rate graph and further to the
left on the negative rate graph, we might conclude that having a higher
percentage of positive words will yield more shares. If the data appears
to peak further to the right on the negative rate graph and further to
the left on the positive rate graph, we might conclude that having a
higher percentage of negative words will yield more shares.
Additionally, each graph displays the correlation between shares and
positve or negative word rate. Again, a value of R closer to -1 or 1
would indicate the two variables are highly correlated and a value
closer to 0 would indicate little to no correlation.

``` r
ggplot(training, aes(x = Positive_Word_Rate, y = Shares)) + 
  geom_point(size = 0.7, color = "royal blue") + 
  stat_cor(method = "pearson", label.x = 0, label.y = 100000, color = "royal blue") +
  xlim(0, 0.125) + ylim(0, 250000)
```

![](Project3_files/figure-gfm/scatterplot2-1.png)<!-- -->

``` r
ggplot(training, aes(x = Negative_Word_Rate, y = Shares)) + 
  geom_point(size = 0.7, color = "dark blue") + 
  stat_cor(method = "pearson", label.x = 0, label.y = 100000, color = "dark blue") +
  xlim(0, 0.125) + ylim(0, 250000)
```

![](Project3_files/figure-gfm/scatterplot2-2.png)<!-- -->

# Modeling

## Set up cross validation

``` r
control <- trainControl(method = "cv", number = 5)
```

## LASSO model

``` r
lasso_model <- train(Shares ~ .,
                     data = training,
                     method ='lasso',
                     preProcess = c("center", "scale"),
                     trControl = control)
predict(lasso_model$finalModel, type = "coef")
```

    ## $s
    ##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14
    ## 
    ## $fraction
    ##  [1] 0.00000000 0.07692308 0.15384615 0.23076923 0.30769231 0.38461538
    ##  [7] 0.46153846 0.53846154 0.61538462 0.69230769 0.76923077 0.84615385
    ## [13] 0.92307692 1.00000000
    ## 
    ## $mode
    ## [1] "step"
    ## 
    ## $coefficients
    ##    Number_Title_Words Number_Content_Words Number_Images Number_Videos
    ## 0             0.00000               0.0000      0.000000        0.0000
    ## 1             0.00000               0.0000      0.000000      138.5328
    ## 2             0.00000             308.2257      0.000000      446.7585
    ## 3             0.00000             490.5038      0.000000      618.3755
    ## 4             0.00000             547.7259      0.000000      674.4402
    ## 5             0.00000             560.1326      0.000000      687.1379
    ## 6             0.00000             564.4714      0.000000      691.4326
    ## 7             0.00000             584.4576      0.000000      711.4725
    ## 8           -63.54108             642.9014      0.000000      769.9609
    ## 9           -76.29415             653.8666      0.000000      782.2253
    ## 10          -76.97130             654.4843      0.000000      782.8756
    ## 11          -92.05088             668.5817      0.000000      796.8285
    ## 12          -95.23637             674.4743     -6.645061      798.9564
    ## 13         -120.22250             715.3273    -58.164322      812.5976
    ##    Positive_Word_Rate Negative_Word_Rate Title_Polarity WeekdayMonday
    ## 0             0.00000            0.00000       0.000000        0.0000
    ## 1             0.00000            0.00000       0.000000        0.0000
    ## 2             0.00000            0.00000       0.000000        0.0000
    ## 3             0.00000            0.00000       0.000000      180.5782
    ## 4             0.00000           59.67134       0.000000      239.6589
    ## 5             0.00000           73.20568       0.000000      255.1374
    ## 6             0.00000           77.11953      -4.488534      260.2012
    ## 7             0.00000           96.07728     -28.125811      288.7892
    ## 8             0.00000          152.44431     -96.006716      372.6313
    ## 9            11.05990          163.42588    -111.665319      389.8758
    ## 10           11.64972          164.00271    -112.471441      391.1050
    ## 11           24.59157          174.97297    -130.067141      429.8247
    ## 12           27.10524          176.73272    -132.882278      437.1123
    ## 13           46.23105          190.11502    -151.339412      556.8847
    ##    WeekdaySaturday WeekdaySunday WeekdayThursday WeekdayTuesday
    ## 0          0.00000       0.00000        0.000000        0.00000
    ## 1          0.00000       0.00000        0.000000        0.00000
    ## 2          0.00000       0.00000        0.000000        0.00000
    ## 3          0.00000       0.00000        0.000000        0.00000
    ## 4          0.00000       0.00000        0.000000        0.00000
    ## 5          0.00000      15.76728        0.000000        0.00000
    ## 6          0.00000      21.25706        0.000000        0.00000
    ## 7         28.99886      51.44346        0.000000        0.00000
    ## 8        112.01359     137.34380        0.000000        0.00000
    ## 9        129.09151     154.62991        0.000000        0.00000
    ## 10       130.23795     155.81450        1.258177        0.00000
    ## 11       164.35894     191.80560       41.174412       39.32642
    ## 12       171.66419     199.54596       48.542938       46.63928
    ## 13       276.41809     313.27092      171.675889      166.54379
    ##    WeekdayWednesday
    ## 0            0.0000
    ## 1            0.0000
    ## 2            0.0000
    ## 3            0.0000
    ## 4            0.0000
    ## 5            0.0000
    ## 6            0.0000
    ## 7            0.0000
    ## 8            0.0000
    ## 9            0.0000
    ## 10           0.0000
    ## 11           0.0000
    ## 12           0.0000
    ## 13         131.9782

``` r
lasso_model$bestTune
```

    ##   fraction
    ## 1      0.1

## Forward stepwise model

``` r
fwdstep_model <- train(Shares ~ .,
                       data = training,
                       method ='glmStepAIC',
                       preProcess = c("center", "scale"),
                       trControl = control,
                       direction = "forward",
                       trace = FALSE)
fwdstep_model
```

    ## Generalized Linear Model with Stepwise Feature Selection 
    ## 
    ## 1472 samples
    ##    8 predictor
    ## 
    ## Pre-processing: centered (13), scaled (13) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1177, 1177, 1178, 1178, 1178 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared     MAE     
    ##   8071.276  0.004286913  3351.775

## Random forest model

``` r
rf_model <- train(Shares ~ ., 
                  data = training, 
                  method = "rf", 
                  preProcess = c("center", "scale"), 
                  trControl = control, 
                  tuneGrid = expand.grid(mtry = 1:(ncol(training) - 1)))
rf_model
```

    ## Random Forest 
    ## 
    ## 1472 samples
    ##    8 predictor
    ## 
    ## Pre-processing: centered (13), scaled (13) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1177, 1177, 1179, 1177, 1178 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared     MAE     
    ##   1     7586.169  0.006679430  3300.900
    ##   2     7712.214  0.008337636  3323.908
    ##   3     7828.531  0.008338324  3373.902
    ##   4     8030.110  0.007156748  3432.893
    ##   5     8152.765  0.007106801  3460.842
    ##   6     8262.927  0.006602226  3484.894
    ##   7     8398.355  0.007539878  3501.446
    ##   8     8441.366  0.006237354  3508.179
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 1.

## Boosted tree model

``` r
gbm_model <- train(Shares ~ .,
                   data = training,
                   method = "gbm",
                   trControl = control,
                   preProcess = c("center", "scale"),
                   verbose = FALSE)
gbm_model
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 1472 samples
    ##    8 predictor
    ## 
    ## Pre-processing: centered (13), scaled (13) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1177, 1178, 1177, 1178, 1178 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   50      8019.673  0.002085941  3400.290
    ##   1                  100      8039.321  0.001504858  3425.814
    ##   1                  150      8029.943  0.001910256  3403.265
    ##   2                   50      8067.431  0.001081430  3415.035
    ##   2                  100      8066.735  0.002712858  3413.244
    ##   2                  150      8144.404  0.003573107  3473.736
    ##   3                   50      8104.459  0.008447615  3481.266
    ##   3                  100      8079.077  0.007754760  3451.902
    ##   3                  150      8267.039  0.008276244  3626.735
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

# Comparison

## Apply model for prediction

``` r
lasso_predict <- predict(lasso_model, newdata = testing)
fwdstep_predict <- predict(fwdstep_model, newdata = testing)
rf_predict <- predict(rf_model, newdata = testing)
gbm_predict <- predict(gbm_model, newdata = testing)
```

## Model performance

``` r
a <- postResample(lasso_predict, obs = testing$Shares)
b <- postResample(fwdstep_predict, obs = testing$Shares)
c <- postResample(rf_predict, obs = testing$Shares)
d <- postResample(gbm_predict, obs = testing$Shares)

table <- as_tibble(rbind(a, b, c, d))
Model <- c("Lasso", "Forward_Stepwise", "Random_Forest", "Boosted_Tree")
performance_table <- cbind(Model, table)
performance_table
```

    ##              Model     RMSE    Rsquared      MAE
    ## 1            Lasso 9650.057 0.005337779 3285.077
    ## 2 Forward_Stepwise 9646.664 0.004534141 3246.652
    ## 3    Random_Forest 9637.121 0.005537998 3189.949
    ## 4     Boosted_Tree 9695.017 0.004761231 3284.014

### Best model by RMSE criteria

``` r
performance_table %>% slice_min(RMSE)
```

    ##           Model     RMSE    Rsquared      MAE
    ## 1 Random_Forest 9637.121 0.005537998 3189.949

### Best model by Rsquared criteria

``` r
performance_table %>% slice_max(Rsquared)
```

    ##           Model     RMSE    Rsquared      MAE
    ## 1 Random_Forest 9637.121 0.005537998 3189.949
