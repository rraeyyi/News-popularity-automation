# Project3  
Rachel Hencher & Yi Ren

![](https://images.squarespace-cdn.com/content/v1/568f9ea70ab377cb54b16efb/067837d1-6426-457d-8872-76b54494fff1/news1-08746fa1.jpg?format=650w)
## Purpose
The purpose of this repository is to store the files related to ST 558 Project 3, as well as to record the history of our work for the assignment. The overall goal of the project was to read in and analyze an online [news popularity data set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) from Mashable in order to create predictive models for the number of article shares. Markdown reports were automated for each of the six channels of interest and can be found linked below.  

## R packages used
```{r}
library(readr)
library(dplyr)
library(knitr)
library(caret)
library(ggplot2)
library(GGally)
library(ggpubr)
library(rmarkdown)
```

## Automation for reports
```{r}
channel <- c("Lifestyle", "Entertainment", "Business", "SocialMedia", "Technology", "World")
report <- function(channel){
  render("Project3.Rmd", 
         params = list(channel = channel), 
         output_file = paste0(channel, ".md"))
}
lapply(channel, report)
```
## Links to each report
+ The analysis for [Lifestyle](Lifestyle.md) articles.
+ The analysis for [Entertainment](Entertainment.md) articles.
+ The analysis for [Business](Business.md) articles.
+ The analysis for [Social Media](SocialMedia.md) articles.
+ The analysis for [Technology](Technology.md) articles.
+ The analysis for [World](World.md) articles.
