# ---------------------------------------------
# Required Packages
# ---------------------------------------------

library(reshape2)
library(ggplot2)
library(mlr)

# ---------------------------------------------
# Reading data
# ---------------------------------------------

data1 = read.csv("../results/dataset_1_results.csv")
df1   = melt(data1, id.vars = 1)
df1$dataset = "dataset1"


data2 = read.csv("../results/dataset_2_results.csv")
df2   = melt(data2, id.vars = 1)
df2$dataset = "dataset2"


data3 = read.csv("../results/dataset_3_results.csv")
df3   = melt(data3, id.vars = 1)
df3$dataset = "dataset3"


df.full = rbind(df1, df2, df3)
colnames(df.full) = c("rep", "algo", "acc", "dataset")

df.full$algo = gsub(x = df.full$algo, pattern = "Naive_Bayes", replacement = "NB") 
df.full$algo = gsub(x = df.full$algo, pattern = "Decision_Tree", replacement = "DT")
df.full$algo = gsub(x = df.full$algo, pattern = "Random_Forest", replacement = "RF")
df.full$algo = gsub(x = df.full$algo, pattern = "Euclidean", replacement = "E")
df.full$algo = gsub(x = df.full$algo, pattern = "Manhattan", replacement = "M")
df.full$algo = factor(df.full$algo)

# ---------------------------------------------
# Boxplot
# ---------------------------------------------

g = ggplot(df.full, aes(x = reorder(algo, -acc), y = acc, group = algo))
g = g + geom_violin() + geom_boxplot(width = .2)
g = g + facet_grid(~dataset) + theme_bw()
g = g + labs(x = "Algorithm", y = "Accuracy")
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g, filename = "boxplot.pdf", width = 7.55, height = 3.16)


# ---------------------------------------------
# Data frame for statistical validation
# ---------------------------------------------

df.friedman = rbind(data1, data2, data3)
df.friedman = df.friedman[,-1]
write.csv(df.friedman, file = "../data/dataForFriedman.csv")

# https://github.com/gabrieljaguiar/nemenyi
# python3 nemenyi.py examples/dataForFriedman.csv cropOutput.tex --h

# ---------------------------------------------
# ---------------------------------------------
