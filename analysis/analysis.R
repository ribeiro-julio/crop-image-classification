# ---------------------------------------------
# Required Packages
# ---------------------------------------------

cat(" - Loading required packages\n")
set.seed(42)

library(reshape2)
library(ggplot2)
library(mlr)

# ---------------------------------------------
# Reading Traditional ML data
# ---------------------------------------------

cat(" - Reading results' data\n")

data1 = read.csv("../results/performances_dataset_1.csv")
ids = 1:nrow(data1)

data1 = cbind(ids, data1)
# ml does not work in the problem (it needs tuning)
# data1$mlp = NULL 

df1   = melt(data1, id.vars = c(1, 5, 6)) 
df1$dataset = "dataset1"

data2 = read.csv("../results/performances_dataset_2.csv")
# ml does not work in the problem (it needs tuning)
# data2$mlp = NULL 
data2 = cbind(ids, data2)
df2   = melt(data2, id.vars = c(1, 5, 6))
df2$dataset = "dataset2"

data3 = read.csv("../results/performances_dataset_3.csv")
# ml does not work in the problem (it needs tuning)
# data3$mlp = NULL 
data3 = cbind(ids, data3)
df3   = melt(data3, id.vars = c(1, 5, 6))
df3$dataset = "dataset3"

df.full = rbind(df1, df2, df3)
colnames(df.full) = c("ids", "rep", "algo", "measure", "value", "dataset")

# ---------------------------------------------
# Boxplot
# ---------------------------------------------

cat(" - Plot: boxplot + violin plot\n")

# filter results by bac
df.bac =  dplyr::filter(df.full, measure == "bac")

g = ggplot(df.full, aes(x = reorder(algo, -value), y = value, group = algo))
g = g + geom_violin() + geom_boxplot(width = .15)
g = g + facet_grid(~dataset) + theme_bw() 
g = g + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g = g + labs(x = "Algorithm", y = "Balanced\nAccuracy per Class (BAC)")
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g, filename = "tratidional_ml_boxplot.pdf", width = 7.55, height = 3.16)

# ---------------------------------------------
# Determining the best traditional ML algorithms
# ---------------------------------------------

rf.dataset1  = dplyr::filter(df.bac, dataset == "dataset1" & algo == "RF")
rf.dataset2  = dplyr::filter(df.bac, dataset == "dataset2" & algo == "RF")
svm.dataset1 = dplyr::filter(df.bac, dataset == "dataset1" & algo == "SVM")
svm.dataset2 = dplyr::filter(df.bac, dataset == "dataset2" & algo == "SVM")

# Algorithm      RF         SVM
# dataset 1 (0.8124209, 0.8166567)
# dataset 2 (0.8211814, 0.8145776)

# ---------------------------------------------
# Best algorithms in dataset2 x CNNs 
# ---------------------------------------------

cnn.data = read.csv("../results/performances_cnn.csv")
ids = 1:nrow(cnn.data)
cnn.data = cbind(ids, cnn.data)
cnn.data$algo = "CNN"
cnn.data$dataset = "dataset2"

# Algorithm      RF         SVM      CNN
# dataset 2 (0.8211814, 0.8145776, 0.8307914)

cnn.dataset2 = cnn.data[, c(1,5,6,3,7)]
rf.dataset2$measure = NULL 
svm.dataset2$measure = NULL

colnames(rf.dataset2)  = colnames(cnn.dataset2)
colnames(svm.dataset2) = colnames(cnn.dataset2)

df2 = rbind(cnn.dataset2, rf.dataset2, svm.dataset2)

g2 = ggplot(df2, aes(x = reorder(algo, -bac), y = bac, group = algo))
g2 = g2 + geom_violin() + geom_boxplot(width = .15)
g2 = g2 + theme_bw() 
g2 = g2 + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g2 = g2 + labs(x = "Algorithm", y = "Balanced\nAccuracy per Class (BAC)")
g2 = g2 + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g2, filename = "cnn_vs_traditional_ml_boxplot.pdf", width = 4.81, height = 3.16)

# ---------------------------------------------
# Wilcoxon per dataset
# ---------------------------------------------

cat(" - Statistical Evaluations (Wilxocon) \n")

obj1 = suppressWarnings(wilcox.test(
	x = svm.dataset1$value, y = rf.dataset1$value, conf.level = 0.95))
cat("P-value: ", obj1$p.value, "\n")

cat("@ SVM vs RF - Dataset1: ")
if(obj1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}

# P-value:  0.1680549
# There is no statistical difference between the methods!

obj2 = suppressWarnings(wilcox.test(
	x = svm.dataset2$value, y = rf.dataset2$value, conf.level = 0.95))
cat("P-value: ", obj2$p.value, "\n")
cat("@ SVM vs RF - Dataset2: ")
if(obj2$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.01628184 
# There is statistical difference between the methods!

svc1 = suppressWarnings(wilcox.test(
	x = svm.dataset1$value, y = svm.dataset2$value, conf.level = 0.95))
cat("P-value: ", svc1$p.value, "\n")
cat("@ SVM - Dataset1 vs SVM - Dataset2: ")
if(svc1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.6972963 
# There is no statistical difference between the methods!

rf1  = suppressWarnings(wilcox.test(
	x = rf.dataset1$value, y = rf.dataset2$value, conf.level = 0.95))
cat("P-value: ", rf1$p.value, "\n")
cat("@ RF - Dataset1 vs RF - Dataset2: ")
if(rf1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.004994396
# There is statistical difference between the methods!

obj4 = suppressWarnings(wilcox.test(
	x = rf.dataset2$value, y = cnn.data$bac, conf.level = 0.95))
cat("P-value: ", obj4$p.value, "\n")
cat("@ RF - Dataset2 vs CNN: ")
if(rf1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.07469582 
# There is statistical difference between the methods!

obj4 = suppressWarnings(wilcox.test(
	x = svm.dataset1$value, y = cnn.data$bac, conf.level = 0.95))
cat("P-value: ", obj4$p.value, "\n")
cat("@ SVM - Dataset1 vs CNN: ")
if(rf1$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.005769847 
# There is statistical difference between the methods!

# ##########
# Summary: 
#  - RF in dataset 2 is statistically better than SVM and 
# 		RF in any othern dataset
#  - CNN is statistically better than RF-dataset2 and SVM-dataset1
# Best overall: CNN

# #############################
# Até aqui funciona de boas
# #############################

# ---------------------------------------------
# Predictions Plots
# ---------------------------------------------

# TODO: Need real Y, to compare with the predictions
# TODO: Keep the image ID when doing the prediction 

cat(" - Loading predictions obtained in dataset 2\n")

# true labels
labels = read.csv("../data/folds_coffee_dataset.csv")
test.labels = dplyr::filter(labels, Fold == "Test")


pred2 = read.csv("../results/predictions_dataset_2.csv")
pred2$dataset = "dataset2"
rf.preds = dplyr::filter(pred2, Algo == "RF")
rf.preds$Algo    = NULL 
rf.preds$dataset = NULL


# filtering RF predictions by algorithhm
ALGOS = c("SVM", "RF", "MLP", "DT", "KNN", "NB")


aux.algos = lapply(ALGOS, function(alg) {
	pred.algo = dplyr::filter(pred2, Algo == alg)
	pred.algo$Seed    = NULL 
	pred.algo$Algo    = NULL 
	pred.algo$dataset = NULL

	preds = apply(pred.algo, 2, DescTools::Mode)
	preds = unlist(lapply(preds, function(pd) return (pd[1])))
	return(preds)
})


# -----------------------
# Joining Traditional ML and CNN predictions
# -----------------------

all.preds = cbind(agg.preds, cnn.agg.preds)
colnames(all.preds)[ncol(all.preds)] = "CNN"
all.preds$id = 1:nrow(all.preds)
rownames(all.preds) = NULL

# -----------------------
# -----------------------



# dataset3 = read.csv("../data/dataset_3.csv", sep = ";")
# Y = dataset3$Y

pred.full = melt(all.preds, id.vars = c(8))
pred.full$value = as.factor(pred.full$value)

# zero.ids = which(pred.full$value == 0)
# tmp = rbind(pred.full[zero.ids, ], pred.full[-zero.ids, ])
# tmp$df_index = factor(tmp$df_index, levels = unique(tmp$df_index))

# TODO: Add True labels
# TODO: order x axis according to the different labels

g3 = ggplot(pred.full, aes(x = id, y = variable, fill = value, colour = value))
g3 = g3 + geom_tile() + theme_bw()
g3 = g3 + scale_fill_manual(values = c("lightgrey", "black"))
g3 = g3 + scale_colour_manual(values = c("lightgrey", "black"))
g3 = g3 + theme(axis.text.x = element_blank(), axis.ticks = element_blank())
g3 = g3 + labs(x = "Image index", y = "Algorithm", fill = "Class", colour = "Class")
g3 

ggsave(g3, filename = "predictionsPlot_dataset2.pdf", width = 7.55, height = 2.44)


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

# https://cran.r-project.org/web/packages/imager/vignettes/gettingstarted.html#example-1-histogram-equalisation

coffee.images = list.files(path = "../dataset-brazilian_coffee_scenes/hardImages/coffee/", full.names=TRUE)
aux.img = lapply(coffee.images, function(image.name) {
	print(image.name)
	image = imager::load.image(image.name)
	bdf = as.data.frame(image)
	bdf = dplyr::mutate(bdf,channel=factor(cc,labels=c('R','G','NI')))
	bdf$image.name = image.name 
	bdf$target = "coffee"
	return(bdf)
})
df.coffee = do.call("rbind", aux.img)


non.coffee.images = list.files(path = "../dataset-brazilian_coffee_scenes/hardImages/noncoffee/", full.names=TRUE)
aux.img = lapply(non.coffee.images, function(image.name) {
	print(image.name)
	image = imager::load.image(image.name)
	bdf = as.data.frame(image)
	bdf = dplyr::mutate(bdf,channel=factor(cc,labels=c('R','G','NI')))
	bdf$image.name = image.name 
	bdf$target = "noncoffee"
	return(bdf)
})
df.non.coffee = do.call("rbind", aux.img)


# ---------------------
# histogram
# ---------------------

df.hist = rbind(df.coffee, df.non.coffee)

hist = ggplot(df.hist, aes(value,col=channel, fill=channel))
hist = hist + geom_histogram(bins=30) + facet_grid(target ~ channel, scales = "free")
hist = hist + labs(x = "", y = "Count") + theme_bw()
#hist
ggsave(hist, file = "hardImagesPixelDistribution.pdf", width = 6.76, height = 3.18)

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
# ---------------------------------------------

cat(" - Finished \n")

# ---------------------------------------------
# ---------------------------------------------
