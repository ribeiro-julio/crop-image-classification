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
