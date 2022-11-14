# Project3

![](https://images.squarespace-cdn.com/content/v1/568f9ea70ab377cb54b16efb/067837d1-6426-457d-8872-76b54494fff1/news1-08746fa1.jpg?format=750w)
# Purpose of project
Our goal is to read in and analyze an online [news popularity data set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity) to create predictive models for the number of shares and automating Markdown reports based on each channel of interests.

# R packages used
```{r}
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(GGally)
library(rmarkdown)
```

# Automation for reports
```{r}
channel <- c("Lifestyle", "Entertainment", "Bus", "Socmed", "Tech", "World")
report <- function(channel){
  render("Project3.Rmd", 
         params = list(channel = channel), 
         output_file = paste0(channel, ".md"))
}
lapply(channel, report)
```
# Links to each reports
+ The analysis for [Lifestyle](Lifestyle.md).
+ The analysis for [Entertainment](Entertainment.md).
+ The analysis for [Bus](Bus.md).
+ The analysis for [Socmed](Socmed.md).
+ The analysis for [Tech](Tech.md).
+ The analysis for [World](World.md).
