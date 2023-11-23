# ---------------------------------------------
# ---------------------------------------------
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# ---------------------------------------------
# ---------------------------------------------
aggregateMLPredictions = function(preds) {

	imgs = unique(preds$im_path)
	aux.imgs = lapply(imgs, function(img) {
		# print(img)
		sub = dplyr::filter(preds, im_path == img)
		response = sub[1,]
		response['predictions'] = Mode(sub$predictions)
		return(response[1:4])
	})
	agg.preds = do.call("rbind", aux.imgs)
	return(agg.preds)
}

# ---------------------------------------------
# ---------------------------------------------

aggregateDLPredictions = function(preds) {

	imgs = unique(preds$im_path)
	aux.imgs = lapply(imgs, function(img) {
		# print(img)
		sub = dplyr::filter(preds, im_path == img)
		sub$Seed = NULL

		response = sub[1,]
		response['predictions'] = Mode(sub$predictions)
		return(response[1:4])
	})
	agg.preds = do.call("rbind", aux.imgs)
	return(agg.preds)

}

# ---------------------------------------------
# ---------------------------------------------
