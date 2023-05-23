# ---------------------------------------------
# Required Packages
# ---------------------------------------------

cat(" - Loading required packages\n")
set.seed(42)

library(reshape2)
library(ggplot2)
library(mlr)

# ---------------------------------------------
# Reading data
# ---------------------------------------------

cat(" - Reading results' data\n")

data1 = read.csv("../results/accuracies_dataset_1.csv")
ids = 1:nrow(data1)

data1 = cbind(ids, data1)
# ml does not work in the problem (it needs tuning)
data1$mlp = NULL 

df1   = melt(data1, id.vars = 1) 
df1$dataset = "dataset1"

data2 = read.csv("../results/accuracies_dataset_2.csv")
# ml does not work in the problem (it needs tuning)
data2$mlp = NULL 
data2 = cbind(ids, data2)
df2   = melt(data2, id.vars = 1)
df2$dataset = "dataset2"

data3 = read.csv("../results/accuracies_dataset_3.csv")
# ml does not work in the problem (it needs tuning)
data3$mlp = NULL 
data3 = cbind(ids, data3)
df3   = melt(data3, id.vars = 1)
df3$dataset = "dataset3"

df.full = rbind(df1, df2, df3)
colnames(df.full) = c("rep", "algo", "acc", "dataset")
df.full$algo = as.factor(toupper(as.character(df.full$algo)))

# ---------------------------------------------
# Boxplot
# ---------------------------------------------

cat(" - Plot: boxplot + violin plot\n")

g = ggplot(df.full, aes(x = reorder(algo, -acc), y = acc, group = algo))
g = g + geom_violin() + geom_boxplot(width = .2)
g = g + facet_grid(~dataset) + theme_bw() 
g = g + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g = g + labs(x = "Algorithm", y = "Accuracy")
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g, filename = "boxplot.pdf", width = 7.55, height = 3.16)

# ---------------------------------------------
# Predictions
# ---------------------------------------------

cat(" - Loading predictions obtained in dataset 1\n")

pred2 = read.csv("../results/predictions_dataset_1.csv")
pred2$dataset = "dataset1"
# ml does not work in the problem (it needs tuning)
pred2$MLP = NULL

pred.full = melt(pred2, id.vars = c(1, 8))
pred.full$value = as.factor(pred.full$value)
zero.ids = which(pred.full$value == 0)

tmp = rbind(pred.full[zero.ids, ], pred.full[-zero.ids, ])
tmp$df_index = factor(tmp$df_index, levels = unique(tmp$df_index))

g2 = ggplot(tmp, aes(x = df_index, y = variable, fill = value, colour = value))
g2 = g2 + geom_tile()
g2 = g2 + scale_fill_manual(values = c("lightgrey", "black"))
g2 = g2 + scale_colour_manual(values = c("lightgrey", "black"))
g2 = g2 + theme(axis.text.x = element_blank(), axis.ticks = element_blank())
g2 = g2 + labs(x = "Image index", y = "Algorithm", fill = "Class", colour = "Class")
ggsave(g2, filename = "predictionsPlot_dataset1.pdf", width = 7.55, height = 2.44)


# ---------------------------------------------
# Hard examples
# ---------------------------------------------

cat(" - Identifying missclassified examples (dataset1) \n")


aux = lapply(1:nrow(pred2), function(i) {
	example = pred2[i,]
	tmp = NULL
	if(all(example[3:7] != example$Y)) {
		tmp = example$df_index
	} 
	return (tmp)
})

hard.ids = unlist(aux)
sel.ids  = which(pred2$df_index %in% hard.ids)

hard.images = pred2[sel.ids, ]
hard.images = hard.images[order(hard.images$Y),]

table(hard.images$Y)
 # 0  1 
# 18 29

write.csv(hard.images, file = "../data/hardImages_dataset1.csv")

# ---------------------------------------------
# Problematic images
# ---------------------------------------------

# TODO: fachar imagens problematicas, onde todos os algoritmos erraram
df.error = read.csv("../data/hardImages_dataset1_withPath.csv")

# ---------------------------------------------
# Overall performance
# ---------------------------------------------

cat(" - Computing overall performance values \n")

avg1 = apply(data1[,-1], 2, mean)
sd1  = apply(data1[,-1], 2, sd)
med1 = apply(data1[,-1], 2, median)

avg2 = apply(data2[,-1], 2, mean)
sd2  = apply(data2[,-1], 2, sd)
med2 = apply(data2[,-1], 2, median)

avg3 = apply(data3[,-1], 2, mean)
sd3  = apply(data3[,-1], 2, sd)
med3 = apply(data3[,-1], 2, median)

obj = list(avg1, sd1, med1, avg2, sd2, med2, avg3, sd3, med3)
df.perf = data.frame(do.call("rbind", obj))
df.perf$perf = rep(c("mean", "sd", "media"), time = 3)
df.perf$dataset = rep(c("dataset1", "dataset2", "dataset3"),each = 3)
print(df.perf)
write.csv(df.perf, file = "../data/overallPerformances.csv")

# ---------------------------------------------
# Data frame for statistical validation
# ---------------------------------------------

cat(" - Exporting data for statistical tests (Friedman) \n")

df.friedman = rbind(data1, data2, data3)
df.friedman = df.friedman[,-1]
write.csv(df.friedman, file = "../data/dataForFriedman.csv", row.names = FALSE)

# https://github.com/gabrieljaguiar/nemenyi
# python3 nemenyi.py examples/dataForFriedman.csv cropOutput.tex --h

# ---------------------------------------------
# Wilcoxon per dataset
# ---------------------------------------------

cat(" - Statistical Evaluations (Wilxocon) \n")

obj1 = suppressWarnings(wilcox.test(x = data1$svc, y = data1$rf, conf.level = 0.95))
cat("P-value: ", obj1$p.value, "\n")

cat("@ SVM vs RF - Dataset1: ")
if(obj1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

obj2 = suppressWarnings(wilcox.test(x = data2$svc, y = data2$rf, conf.level = 0.95))
cat("P-value: ", obj2$p.value, "\n")
cat("@ SVM vs RF - Dataset2: ")
if(obj2$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

svc1 = suppressWarnings(wilcox.test(x = data1$svc, y = data2$svc, conf.level = 0.95))
cat("P-value: ", svc1$p.value, "\n")
cat("@ SVM - Dataset1 vs SVM - Dataset2: ")
if(svc1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

rf1  = suppressWarnings(wilcox.test(x = data1$rf, y = data2$rf, conf.level = 0.95))
cat("P-value: ", rf1$p.value, "\n")
cat("@ RF - Dataset1 vs RF - Dataset2: ")
if(rf1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

# ---------------------------------------------
# ---------------------------------------------

cat(" - Finished \n")

# ---------------------------------------------
# ---------------------------------------------
