# Project3

# Automation for reports
```{r}
channel <- c("Lifestyle", "Entertainment", "Bus", "Socmed", "Tech", "World")
report <- paste0(channel, ".md")
params <- lapply(channel, FUN = function(x){
  return(list(channel = x))
})
reports <- tibble(channel, report, params)
reports

library(rmarkdown)
apply(reports, MARGIN = 1, FUN=function(x){
  render(input = "Channel_Analysis.Rmd", output_file = x[[2]], params = x[[3]])
})
```
