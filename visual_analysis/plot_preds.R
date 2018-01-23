library(ggplot2)
library(grid)
library(gridExtra)
op <- par(no.readonly = TRUE)

plot_preds <- function(preds) {
    pdf("batch80_ratio50_step700.pdf", height = 4.8, width = 6)
    par(mfrow = c(2,3))
    for (i_esn in 1:dim(preds)[1]) {
        rcd = preds[i_esn,]
        esn = rcd[1,1]
        tag = rcd[1,2]
        if (tag == 'Fail') {
            plot_color = 'red'
            plot_shape = 4    # x
        } else {
            plot_color = 'blue'
            plot_shape = 20    # o
        }
        pred = rcd[3:length(rcd)]
        pred = pred[!is.na(pred)]
        plot(x = 1:length(pred), y = pred, pch = plot_shape, col = plot_color, xlab = "index", ylab = "prediction", main = paste(esn, "test", tag, sep = "_"))
    }
    plot(x = 1:length(pred), y = pred, pch = plot_shape, col = plot_color, type = "n", axes  =  FALSE, xlab = "", ylab = "", main = "Predictions on test set")
    legend("top", pch = c(4, 20), col = c('red', 'blue'), legend = c('Fail', 'Normal'), bty = "n")
    dev.off()
    par(op)
}

# Load data
load_pred <- function(pred_csv) {
    preds = read.table(pred_csv, header = FALSE, sep = ',', fill = TRUE, na.strings = '')
    return(preds)
}

# ---------------- Main --------------------
pred_csv = '/Users/yahuishi/Documents/20171123_CF6_Blade/main/preds/ratio05_batch80/model-700_test.csv'
preds = load_pred(pred_csv)

raw_data = read.csv('/Users/yahuishi/Documents/20171123_CF6_Blade/data/csv/for_MCDCNN/CF6_test.csv')

for (i_esn in 1:dim(preds)[1]) {
    rcd = preds[i_esn,]
    esn = rcd[1,1]
    tag = rcd[1,2]
    prdRlts = unlist(rcd[1,3:dim(rcd)[2]])
    prdRlts = prdRlts[!is.na(prdRlts)]
    plot_data = data.frame(label = "pred", value = c(rep(NA, 31), prdRlts), index = 1:(length(prdRlts)+31))
    for (param in names(raw_data)) {
        if (param %in% c("tk_egthdm", "tk_zt49", "cr_zvb1f", "cr_degt", "cr_gpcn25", "cr_gwfm")) {
            param_seq = raw_data[raw_data$tk_esn == esn, param]
            plot_data = rbind(plot_data, data.frame(label = param, value = param_seq, index = 1:length(param_seq)))
        }
    }
    p = ggplot(data = plot_data, aes(x = index, y = value)) + geom_point(size = 1) + facet_wrap(~label, ncol=3) + coord_cartesian(xlim = c(0, length(prdRlts))) + labs(title = paste0(esn, '_', tag))
    ggsave(paste0("./pred_rlts/", esn, "_pred_sigParams.pdf"), width = 12, height = 8, units = "in")
}
