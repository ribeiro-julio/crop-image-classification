# ---------------------------------------------
# ---------------------------------------------

loadData = function(datapath, dataname) {
	
	data = read.csv(datapath)
	ids = 1:nrow(data)
	data = cbind(ids, data)
	df = reshape2::melt(data, id.vars = c(1, 5, 6))
	df$dataset = dataname

	return(df)
}

# ---------------------------------------------
# ---------------------------------------------

loadDLData = function(datapath, algoname) {

	dl.data = read.csv(datapath)
	ids = 1:nrow(dl.data)
	dl.data = cbind(ids, dl.data)
	dl.data$algo = algoname
	return(dl.data)

}

# ---------------------------------------------
# ---------------------------------------------

loadDLLerningCurves = function(datapath, algoname) {
	
	dl.files = list.files(path = datapath, full.names = TRUE)
	aux = lapply(1:30, function (i){
		dl.data = read.csv(dl.files[i])
		sub.data = dl.data[,c(1,2,5,4)]
		sub.data$seed = i-1
		sub.data$algo = algoname
		return(sub.data)
	})
	return(aux)
}

# ---------------------------------------------
# ---------------------------------------------

loadPixelData = function(images, target) {
	aux = lapply(1:nrow(images), function(i) {
		image.name = images[i, "image"]
		print(image.name)
		image = imager::load.image(image.name)
		bdf = as.data.frame(image)
		bdf = dplyr::mutate(bdf,channel=factor(cc,labels=c('R','G','NI')))
		bdf$image.name = image.name 
		bdf$target = target
		return(bdf)
	})
	df = do.call("rbind", aux)
	return(df)
}

# ---------------------------------------------
# ---------------------------------------------
