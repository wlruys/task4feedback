library("optparse")
library("data.table")
library(dplyr)
library(ggplot2)
library(Rmisc)
library(RColorBrewer)
library(ggh4x)
library(ggbreak)

args=commandArgs(trailingOnly=TRUE)

wallclock_data <- read.csv(args[1], header=T, fileEncoding="UTF-8-BOM")
simclock_data <- read.csv(args[2], header=T, fileEncoding="UTF-8-BOM")

print(paste("path1:", args[1]))
print(paste("path2:", args[2]))

# print(wallclock_data)
#print(simclock_data)

merged_data <- merge(wallclock_data, simclock_data, by='Iterations')
print(merged_data)

line_graph <- ggplot() +
  geom_point(data=merged_data, aes(x=WallClock, y=SimTime), color="blue") +
  geom_smooth(data=merged_data, aes(x=WallClock, y=SimTime), color="green") +
  labs(x="Wall clock (s)", y = "Simulated Time (s)")
ggsave("wall-sim_time.pdf", height=6, width=6 )
