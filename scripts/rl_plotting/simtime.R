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
print (data)
data <- data_summary(data, varname="ExecutionTime", groupnames=c("Mode"))

data$Mode <- factor(data$Mode, levels=c(
  "Serial", "BSP", "Independent", "HEFTTheory", "heft", "loadbalance", "eft_without_data", "eft_with_data", "random"))
levels(data$Mode)[levels(data$Mode) == "eft_without_data"] <- "EFT-NoData"
levels(data$Mode)[levels(data$Mode) == "eft_with_data"] <- "EFT"
levels(data$Mode)[levels(data$Mode) == "random"] <- "Random"
levels(data$Mode)[levels(data$Mode) == "loadbalance"] <- "LoadBalance"

data$vlines <- factor(data$Mode, labels=c(
  "Theory", "Theory", "Theory", "Offline", "Offline", "Online", "Online", "Online", "Online")) 

print(data$Mode)
print(data)
time_plot <- ggplot(data=data, aes(x=Mode, fill=Mode, y=ExecutionTime)) +
             # scale_x_discrete(labels=c("Serial", "BSP", "Independent", "HEFTTheory",  "HEFT", "LoadBalance", "EFT-NoData", "EFT", "Random"))+
             geom_bar(
                      #mapping=aes(x=factor(Mode), fill=Mode, y=ExecutionTime),
                      position=position_dodge(preserve='single'), stat="identity") +
             labs(fill="Methods", y="Execution time (s)", x="")+
             theme(axis.title = element_text(color="black", size=20),
                   axis.text.y = element_text(color="black", size=20),
                   axis.text.x = element_text(color="black", size=20, angle=90),
                   axis.title.y = element_text(size=20),
                   axis.title.x = element_text(size=20),
                   legend.position="none") +
             # facet_wrap(~vlines, scales="free_x") +
             facet_grid(. ~vlines, scales="free_x", space="free") +
             geom_text(aes(label=sprintf("%1.2f", ExecutionTime)),
             position=position_dodge(1), vjust=0) +
             geom_errorbar(aes(ymin=ExecutionTime-sd, ymax=ExecutionTime+sd),
             width=.2, position=position_dodge(.9))

time_plot
ggsave(args[2], height=6, width=10)
