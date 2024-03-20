library("optparse")
library("data.table")
library(dplyr)
library(ggplot2)
library(Rmisc)
library(RColorBrewer)
library(ggh4x)
library(ggbreak)

args=commandArgs(trailingOnly=TRUE)

loss_data <- read.csv(args[1], header=T, fileEncoding="UTF-8-BOM")
print(loss_data)
line_graph <- ggplot() +
  geom_point(data=loss_data, aes(x=Iterations, y=Loss), color="blue") +
  labs(x="Iterations", y = "Loss")
line_graph
ggsave("loss.pdf", height=6, width=6 )
