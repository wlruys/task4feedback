library("optparse")
library("data.table")
library(dplyr)
library(ggplot2)
library(Rmisc)
library(RColorBrewer)
library(ggh4x)
library(grid)
library(cowplot)
library(gridExtra)
library(tidyr)

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
data <- data_summary(data, varname="ExecutionTime", groupnames=c("Mode", "BarLoc"))
data$Mode <- factor(data$Mode, levels=c(
  "heft", "loadbalance", "eft_without_data", "eft_with_data", "random"))
levels(data$Mode)[levels(data$Mode) == "eft_without_data"] <- "EFT-NoData"
levels(data$Mode)[levels(data$Mode) == "eft_with_data"] <- "EFT"
levels(data$Mode)[levels(data$Mode) == "heft"] <- "HEFT"
levels(data$Mode)[levels(data$Mode) == "random"] <- "Random"
levels(data$Mode)[levels(data$Mode) == "loadbalance"] <- "LoadBalance"

data$vlines <- factor(data$Mode, labels=c(
 "Offline", "Online", "Online", "Online", "Online"))

wide_data <- data %>%
  pivot_wider(names_from = BarLoc, values_from = ExecutionTime)
print(wide_data)


time_breakdown <- rbind(
  data.frame(wide_data$Mode, "time" = wide_data$bottom, "time_type" = "Total Execution Time"),
  data.frame(wide_data$Mode, "time" = wide_data$middle, "time_type" = "Computation Time"),
  data.frame(wide_data$Mode, "time" = wide_data$top, "time_type" = "Data Move Time")
)

print(time_breakdown)

plot <- ggplot(data = time_breakdown,
       mapping = aes(x=wide_data.Mode, y=time, fill=time_type)) + geom_bar(stat = "identity", position = position_dodge(width=0.3)) +
  theme(legend.position="top", panel.spacing = unit(0.3, "lines"),
        axis.text.y = element_text(color="black", size=10),
        axis.text.x = element_text(color="black", size=10),
        axis.title.y = element_text(color="black", size=10, face="bold"),
        axis.title.x = element_text(color="black", size=10, face="bold"),
  )+
  labs(x="Mapping Policies", y="Runtime (sec)", fill='Time Breakdown')


plot
ggsave(args[2], height=6, width=10)
