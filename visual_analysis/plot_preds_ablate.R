library(ggplot2)
library(stringi)

op = par(no.readonly = TRUE)
args = commandArgs(trailingOnly = TRUE)

# load data
if (length(args) > 1) {
    csvPath = args[1]
    outPath = args[2]
} else {
    csvPath = '/Users/yahuishi/Documents/20171123_CF6_Blade/main/preds/ratio05_batch80/model-700-ablate/706983_706903.csv'
    outPath = stri_replace_all_fixed(csvPath, '.csv', '_zoom.pdf')
}
print(paste(c('loading...', csvPath), collapse = ' '))
preds = data.frame(read.table(csvPath, header = FALSE, sep = ',', fill = TRUE, na.strings = ''), stringsAsFactors = FALSE)

# plot
eng = preds[1, 1]
tag = preds[1, 2]
plot_data = data.frame(label = c(), value = c(), index = c())
for (i_row in 1:dim(preds)[1]) {
    ablated_parm = as.character(preds[i_row, 3])
    values_pred = unlist(preds[i_row, 4:dim(preds)[2]])
    plot_data = rbind(plot_data, data.frame(label = ablated_parm, value = values_pred, index = 1:length(values_pred)))
}
p = ggplot(data = plot_data, aes(x = index, y = value)) + geom_point(size = 1) + facet_wrap(~label, ncol=4) + coord_cartesian(xlim = c(1200, length(values_pred))) + labs(title = paste0(eng, '_', tag))

ggsave(outPath, width = 12, height = 8, units = "in")
par(op)
