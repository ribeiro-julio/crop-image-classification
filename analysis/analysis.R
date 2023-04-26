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
# RF plot using the best dataset (dataset2)
# ---------------------------------------------


# TODO: generate dataset
# cat(" @ Plot: Random Forest (importance) \n")

# dataset = read.csv()
# mlrTask = mlr::makeClassifTask(df_all2_resc[,-1], id = "test", target = "rotulo")
# lrn     = mlr::makeLearner("classif.ranger", importance = "permutation")
# model   = mlr::train(task = mlrTask, learner = lrn)

# trueModel = model$learner.model

# importance = as.data.frame(trueModel$variable.importance)
# rf.df = cbind(rownames(importance), importance)
# rownames(rf.df) = NULL
# colnames(rf.df) = c("Feature", "Importance")

# g_importance = ggplot(rf.df, aes(x = reorder(Feature, Importance), y = Importance))
# g_importance = g_importance  + geom_col(width = 0.8, fill="lightblue", col="darkblue")
# g_importance = g_importance  + labs(y="Importance", x = "Feature") + coord_flip() + theme_bw() 
# # g_importance # for debug
# ggsave(g_importance, file = "plots/fig_randomForest.pdf", units = "in", width = 9, 
# 	height = 6, dpi = 300, pointsize = 20)


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
