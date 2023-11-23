# ---------------------------------------------
# Required Packages
# ---------------------------------------------

dir.create(path = "./plots/", showWarnings = FALSE)
set.seed(42)

cat(" - Loading required packages\n")
library(reshape2)
library(ggplot2)
library(mlr)

cat(" - Loading other R files \n")
R.files = list.files(path = "R", full.name = TRUE)
for(file in R.files) {
	print(file)
	source(file)
}

# ---------------------------------------------
# Reading Traditional ML data
# ---------------------------------------------

cat(" - Reading results' data\n")

df1 = loadData(datapath = "../results/performances_dataset_1.csv",
	dataname = "dataset1")
df2 = loadData(datapath = "../results/performances_dataset_2.csv",
	dataname = "dataset2")
df3 = loadData(datapath = "../results/performances_dataset_3.csv",
	dataname = "dataset3")

df.full = rbind(df1, df2, df3)
colnames(df.full) = c("ids", "rep", "algo", "measure", "value", "dataset")

# ---------------------------------------------
# Boxplot
# ---------------------------------------------

cat(" - Plot: boxplot + violin plot\n")

# filter results by bac
df.bac = dplyr::filter(df.full, measure == "bac")

g = ggplot(df.full, aes(x = reorder(algo, -value), y = value, group = algo))
g = g + geom_violin() + geom_boxplot(width = .15)
g = g + facet_grid(~dataset) + theme_bw() 
g = g + geom_hline(yintercept = 0.8, linetype = "dotted", colour = "red")
g = g + labs(x = "Algorithm", y = "Balanced\nAccuracy per Class (BAC)")
g = g + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g, filename = "plots/tratidional_ml_boxplot.pdf", width = 7.55, height = 3.16)

# ---------------------------------------------
# Determining the best traditional ML algorithms
# ---------------------------------------------

rf.dataset1  = dplyr::filter(df.bac, dataset == "dataset1" & algo == "RF")
rf.dataset2  = dplyr::filter(df.bac, dataset == "dataset2" & algo == "RF")
svm.dataset1 = dplyr::filter(df.bac, dataset == "dataset1" & algo == "SVM")
svm.dataset2 = dplyr::filter(df.bac, dataset == "dataset2" & algo == "SVM")

# Algorithm      RF         SVM
# dataset 1 (0.8113413, 0.8166567)
# dataset 2 (0.8228451, 0.8145776)

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

# P-value: 0.09349708
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
# P-value:  0.00550026 
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
# P-value:  0.0001748091
# There is statistical difference between the methods!

# ##########
# Summary: 
#  - RF in dataset 2 is statistically better than SVM and 
# 		RF in any othern dataset
#  - Choose dataset 2 as baseline (RF, SVM)

# ---------------------------------------------
# Best algorithms in dataset2 x CNNs x VGG16
# ---------------------------------------------

cnn.data = loadDLData(datapath = "../results/performances_cnn.csv", 
	algoname = "CNN")

vgg.data = loadDLData(datapath = "../results/performances_vgg16.csv", 
	algoname = "VGG16")

# Algorithm      RF         SVM      CNN        VGG
# dataset 2 (0.8211814, 0.8145776, 0.8307914, 0.834383)

rf.dataset2$measure  = NULL 
rf.dataset2$dataset  = NULL 
svm.dataset2$measure = NULL
svm.dataset2$dataset = NULL

cnn.dataset2 = cnn.data[, c(1,5,6,3)]
vgg.dataset2 = vgg.data[, c(1,5,6,3)]
colnames(rf.dataset2)  = colnames(cnn.dataset2)
colnames(svm.dataset2) = colnames(cnn.dataset2)

df2 = rbind(cnn.dataset2, vgg.dataset2, rf.dataset2, svm.dataset2)

g2 = ggplot(df2, aes(x = reorder(algo, -bac), y = bac, group = algo, fill = algo))
g2 = g2 + geom_violin() + geom_boxplot(width = .15, fill = "white")
g2 = g2 + theme_bw() + guides(fill = "none")
g2 = g2 + labs(x = "Algorithm", y = "Balanced\nAccuracy per Class (BAC)")
g2 = g2 + theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave(g2, filename = "plots/dl_vs_traditional_ml_boxplot.pdf", width = 4.81, height = 3.16)

# ---------------------------------------------
# Statistical tests comparing DL x Traditional ML
# ---------------------------------------------

obj4 = suppressWarnings(wilcox.test(
	x =  rf.dataset2$bac, y = cnn.data$bac, conf.level = 0.95))
cat("P-value: ", obj4$p.value, "\n")
cat("@ RF - Dataset2 vs CNN: ")
if(obj4$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.1058117 
# There is no statistical difference between the methods!

obj5 = suppressWarnings(wilcox.test(
	x = svm.dataset2$bac, y = cnn.data$bac, conf.level = 0.95))
cat("P-value: ", obj5$p.value, "\n")
cat("@ SVM - Dataset2 vs CNN: ")
if(obj5$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.002210278 
# There is statistical difference between the methods!

obj6 = suppressWarnings(wilcox.test(
	x = rf.dataset2$bac, y = vgg.data$bac, conf.level = 0.95))
cat("P-value: ", obj6$p.value, "\n")
cat("@ SVM - Dataset2 vs VGG: ")
if(obj6$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.0003181467 
# There is statistical difference between the methods!


obj7 = suppressWarnings(wilcox.test(
	x = vgg.data$bac, y = cnn.data$bac, conf.level = 0.95))
cat("P-value: ", obj7$p.value, "\n")
cat("@ VGG vs CNN: ")
if(obj7$p.value < 0.05) {
	cat("There is statistical difference between the methods!\n")
} else {
	cat("There is no statistical difference between the methods!\n")
}
# P-value:  0.2643196 
# There is no statistical difference between the methods!

# ##########
# Summary: 
#  - there is no statistical difference bewteen RF and CNN
#  - there is no statistical difference between VGG and CN
#  - but VGG is statistically better than RF and SVM

# ---------------------------------------------
# Deep Learning - Learning curves
# ---------------------------------------------

cat(" - Plotting DL learning curves \n")

cnn.learningCurves = loadDLLerningCurves(datapath = "../results/cnnLogs/",
	algoname = "CNN")

vgg.learningCurves = loadDLLerningCurves(datapath = "../results/vgg16Logs/",
	algoname = "VGG16")

cnn.epochs = data.frame(unlist(lapply(cnn.learningCurves, nrow)))
colnames(cnn.epochs) = c("epoch")
cnn.epochs$algo = "CNN"

vgg.epochs = data.frame(unlist(lapply(vgg.learningCurves, nrow)))
colnames(vgg.epochs) = c("epoch")
vgg.epochs$algo = "VGG"

df.epochs = rbind(cnn.epochs, vgg.epochs)

g3 = ggplot(df.epochs, aes(x = epoch, colour = algo, fill = algo)) 
g3 = g3 + geom_histogram(stat="count", alpha = 0.5) + theme_bw()
g3 = g3 + labs(x = "Number of epochs\n in the training step", y = "Count")
g3 = g3 + labs(fill = "Algorithm", colour = "Algorithm")
# g3 
ggsave(g3, file = "plots/dl_epochs.pdf", width = 4.46, height = 3.36)

# ---------------------------------------------
# ---------------------------------------------

df.lc.cnn = do.call("rbind", cnn.learningCurves)
g4 = ggplot(df.lc.cnn, mapping = aes(x = epoch, y = loss, 
	group = seed))
g4 = g4 + geom_line() + theme_bw()
ggsave(g4, file = "plots/cnn_loss_curves.pdf"), width = 3.47, height = 3.11)

df.lc.vgg = do.call("rbind", vgg.learningCurves)
g5 = ggplot(df.lc.vgg, mapping = aes(x = epoch, y = loss, 
	group = seed))
g5 = g5 + geom_line() + theme_bw()
ggsave(g5, file = "plots/vgg_loss_curves.pdf", width = 3.47, height = 3.11)


df.lc = rbind(df.lc.cnn, df.lc.vgg)
df.lc.melted = melt(df.lc, id.vars = c(1,5,6))

g6 = ggplot(df.lc.melted, mapping = aes(x = epoch, y = value, 
	group = seed, colour = variable)) 
g6 = g6 + facet_grid(algo~variable, scales = "free")
g6 = g6 + geom_line() + theme_bw() + guides(colour = "none")
ggsave(g6, file = "plots/dl_loss_curves_allmeasuers_grid.pdf", width = 5.61, height = 3.62)

g7 = ggplot(df.lc.melted, mapping = aes(x = epoch, y = value, 
	group = seed, colour = variable)) 
g7 = g7 + facet_wrap(algo~variable, scales = "free")
g7 = g7 + geom_line() + theme_bw() + guides(colour = "none")
ggsave(g7, file = "plots/dl_loss_curves_allmeasuers_wrap.pdf", width = 5.61, height = 3.62)


# ---------------------------------------------
# Predictions Plots (RF, VGG, CNN, SVM)
# ---------------------------------------------

cat(" - Loading predictions obtained in dataset 2\n")

pred2 = read.csv("../results/predictions_dataset_2.csv")
rf.preds  = dplyr::filter(pred2, algo == "RF")
svm.preds = dplyr::filter(pred2, algo == "SVM")

cnn.preds = read.csv("../results/predictions_cnn.csv")
cnn.preds$Fold = NULL
vgg.preds = read.csv("../results/predictions_vgg16.csv")
vgg.preds$Fold = NULL


# ---------------------
# Aggregating the predictions
# ---------------------
cat(" - Loading and aggregating RF predictions\n")
rf.agg.preds = aggregateMLPredictions(preds = rf.preds)

cat(" - Loading and aggregating SVM predictions\n")
svm.agg.preds = aggregateMLPredictions(preds = svm.preds)

cat(" - Loading and aggregating CNN predictions\n")
cnn.agg.preds = aggregateDLPredictions(preds = cnn.preds)

cat(" -Loading and aggregating VGG predictions\n")
vgg.agg.preds = aggregateDLPredictions(preds = vgg.preds)

# ---------------------------------------------
# ---------------------------------------------

all.preds = cbind(rf.agg.preds, svm.agg.preds$predictions, 
	cnn.agg.preds$predictions, vgg.agg.preds$predictions)
all.preds$algo = NULL
colnames(all.preds) = c("image", "Y", "RF", "SVM", "CNN", "VGG16")

all.preds.melted = melt(all.preds, id.vars = c(1))
all.preds.melted$value = as.factor(all.preds.melted$value)

# TODO: order x axis according to the different labels
zero.ids = which(all.preds.melted$value == 0)
tmp = rbind(all.preds.melted[zero.ids, ], all.preds.melted[-zero.ids, ])
tmp$image = factor(tmp$image, levels = unique(tmp$image))

g8 = ggplot(tmp, aes(x = image, y = variable, 
	fill = value, colour = value))
g8 = g8 + geom_tile() + theme_bw()
g8 = g8 + scale_fill_manual(values = c("lightgrey", "black"))
g8 = g8 + scale_colour_manual(values = c("lightgrey", "black"))
g8 = g8 + theme(axis.text.x = element_blank(), axis.ticks = element_blank())
g8 = g8 + labs(x = "Image index", y = "Algorithm", fill = "Class", colour = "Class")
ggsave(g8, filename = "plots/predictions.pdf", width = 7.55, height = 2.44)


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
