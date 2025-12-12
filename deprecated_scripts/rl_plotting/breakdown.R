library("optparse")
library("data.table")
library(dplyr)
library(ggplot2)
library(Rmisc)
library(RColorBrewer)
library(ggh4x)

args=commandArgs(trailingOnly=TRUE)

data_summary <- function(data, varname, groupnames) {
  require(plyr)
  summary_func <- function(x, col) {
    c(mean = mean(x[[col]], na.rm=TRUE), sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum <- ddply(data, groupnames, .fun=summary_func, varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return (data_sum)
}

data <- read.csv(args[1], header=T, fileEncoding="UTF-8-BOM")
data <- data_summary(data, varname="time", groupnames=c("device", "type"))
data['time'] <- data['time'] / 1000000
print(data)

data$breakdown <- ifelse(data$type=="compute-active", "Computation",
                         ifelse(data$type=="data-active", "Data Transfer", "Idle"))
print(data)
time_plot <- ggplot(data=data, aes(x=device, fill=breakdown, y=time)) +
  geom_bar(mapping=aes(x=device, fill=breakdown, y=time),
           position="stack", stat="identity") +
  labs(fill="Time Breakdown", y="Execution time (s)", x="")+
  theme(axis.title = element_text(color="black", size=20),
        axis.text.y = element_text(color="black", size=20),
        axis.text.x = element_text(color="black", size=20, angle=90),
        axis.title.y = element_text(size=20),
        axis.title.x = element_text(size=10)) +
  # geom_text(aes(label=sprintf("%1.2f", time)),
  #           position=position_dodge(1), vjust=0) +
  geom_errorbar(aes(ymin=time-sd, ymax=time+sd), width=.2, position=position_dodge(.9))

time_plot
ggsave(args[2], height=6, width=10)

