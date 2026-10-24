#!/usr/bin/env Rscript
#
# Fit a single mgcv::bam model + predict on a held-out test set.
#
# Called by RGamModel._fit_predict_subprocess (use_bam=True);
# fit_loglik fits in-process via rpy2 instead.
# The Python side handles cross-validation, scaling, and
# baseline-vs-full bookkeeping; this script does one fit + one predict
# per invocation.
#
#
# Standalone usage (for debugging):
#   Rscript src/replica/analysis/models/additive/fit_bam.R \
#       --train /tmp/train.tsv \
#       --test  /tmp/test.tsv  \
#       --formula 'y ~ s(x0, bs="cr", k=6) + s(participant_id, bs="re")' \
#       --out   /tmp/preds.tsv \
#       --method fREML \
#       --discrete TRUE

suppressMessages(library(mgcv))

parse_args <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[i]
    if (!startsWith(key, "--")) {
      stop(sprintf("expected --flag, got %s", key))
    }
    key <- sub("^--", "", key)
    val <- argv[i + 1]
    out[[key]] <- val
    i <- i + 2
  }
  required <- c("train", "test", "formula", "out")
  missing <- setdiff(required, names(out))
  if (length(missing) > 0) {
    stop(sprintf("missing required args: %s", paste(missing, collapse = ", ")))
  }
  out
}

args <- parse_args()
# Python supplies the extracted package resource when necessary. Standalone
# invocations find the helper beside this script, independent of the cwd.
helper <- args$prediction_helpers
if (is.null(helper)) {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
  helper <- file.path(dirname(script), "gam_prediction.R")
}
source(helper)

method   <- if (is.null(args$method))   "fREML" else args$method
discrete <- if (is.null(args$discrete)) "TRUE"  else args$discrete
discrete <- as.logical(discrete)
prediction_discrete <- if (is.null(args$prediction_discrete)) FALSE else as.logical(args$prediction_discrete)
if (is.na(prediction_discrete) || (prediction_discrete && !discrete)) {
  stop("prediction_discrete must be TRUE or FALSE and requires discrete fitting")
}
nthreads <- if (is.null(args$nthreads)) 1L else as.integer(args$nthreads)
# re_predict: "exclude" drops bs='re' and every smooth involving participant_id
# or item_id (including fs, factor-by and tensor interactions); "conditional"
# keeps them at the true levels, which must be seen for groups in the formula.
# "shrink_unseen" drops only smooths depending on an unseen group in each row.
re_predict <- if (is.null(args$re_predict)) "exclude" else args$re_predict
if (!re_predict %in% c("exclude", "conditional", "shrink_unseen")) {
  stop(sprintf("re_predict must be 'exclude', 'conditional', or 'shrink_unseen', got '%s'", re_predict))
}
# sigma_mode: "sig2" reports mgcv's estimated scale; "train_mse" reports
# sqrt(mean(train residuals^2)) from the fit's own training residuals.
sigma_mode <- if (is.null(args$sigma_mode)) "sig2" else args$sigma_mode
if (!sigma_mode %in% c("sig2", "train_mse")) {
  stop(sprintf("sigma_mode must be 'sig2' or 'train_mse', got '%s'", sigma_mode))
}

train <- read.table(args$train, header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE)
test  <- read.table(args$test,  header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE)

for (col in c("participant_id", "item_id")) {
  if (!(col %in% colnames(train))) next
  train[[col]] <- factor(train[[col]])
}

f <- as.formula(args$formula)
m <- bam(f, data = train, method = method, discrete = discrete,
         nthreads = nthreads)
if (prediction_discrete && is.null(m$dinfo)) {
  stop("prediction_discrete requires a discretely fitted bam model")
}

if (re_predict == "shrink_unseen") {
  preds <- replica_gam_shrink_unseen(m, train, test, prediction_discrete)
} else if (re_predict == "conditional") {
  test <- replica_gam_prediction_data(m, train, test, re_predict)
  preds <- predict(m, newdata = test,
                   type = "response",
                   discrete = prediction_discrete)
} else {
  test <- replica_gam_prediction_data(m, train, test, re_predict)
  # Discrete prediction needs mgcv to construct its prediction frame.
  preds <- predict(m, newdata = test,
                   exclude = replica_gam_excluded_smooths(m),
                   newdata.guaranteed = !prediction_discrete,
                   type = "response",
                   discrete = prediction_discrete)
}

if (sigma_mode == "train_mse") {
  sigma <- sqrt(mean(residuals(m, type = "response")^2))
} else {
  sigma <- sqrt(m$sig2)
}

write.table(data.frame(preds = preds), args$out, sep = "\t",
            row.names = FALSE, quote = FALSE)

# Optional --coefs <path>: emit the parametric (linear fixed-effect)
# coefficients from summary(m)$p.table. Smoothed columns are reported in
# $s.table, so only linearly-entered predictors yield coefficients.
if (!is.null(args$coefs)) {
  p <- summary(m)$p.table
  if (!is.null(p) && nrow(p) > 0) {
    df_coef <- data.frame(
      term      = rownames(p),
      estimate  = p[, "Estimate"],
      std_error = p[, "Std. Error"],
      stringsAsFactors = FALSE
    )
  } else {
    df_coef <- data.frame(term = character(),
                          estimate = numeric(),
                          std_error = numeric())
  }
  write.table(df_coef, args$coefs, sep = "\t",
              row.names = FALSE, quote = FALSE)
}

# Single machine-readable sigma line on stdout. Python parses this.
cat(sprintf("SIGMA %.17g\n", sigma))
