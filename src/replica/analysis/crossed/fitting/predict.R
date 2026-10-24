# Predict genuinely unseen participants and materials from frozen mixed fits.
# Usage: Rscript predict.R observations.tsv specification.json fits_dir output.tsv
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4L) stop("Expected observations, specification, fitted models and output")
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
spec <- fromJSON(args[2], simplifyVector=FALSE)
data <- read.delim(args[1], check.names=FALSE)
train <- data[data$role == "train", , drop=FALSE]
test <- data[data$role == "test", , drop=FALSE]
if (!nrow(train) || !nrow(test)) stop("Empty independent training or test sample")
for (group in unique(c(unlist(spec$groups), "participant", "material", "item"))) {
  if (anyNA(data[[group]]) || length(intersect(train[[group]], test[[group]]))) {
    stop(paste("Independent holdout identity violated:", group))
  }
}
if (file.exists(args[4])) stop("Prediction output already exists")
scaling <- fromJSON(file.path(args[3], "scaling.json"), simplifyVector=FALSE)
for (entry in scaling) {
  column <- entry$column
  if (!isTRUE(all.equal(mean(train[[column]]), entry$center)) ||
      !isTRUE(all.equal(sd(train[[column]]), entry$scale))) stop("Training transform mismatch")
  test[[column]] <- (test[[column]] - entry$center) / entry$scale
}
normal_quadrature <- function(n) {
  jacobi <- matrix(0, n, n)
  jacobi[cbind(seq_len(n - 1L), 2:n)] <- sqrt(seq_len(n - 1L))
  eig <- eigen(jacobi + t(jacobi), symmetric=TRUE)
  list(nodes=eig$values, weights=eig$vectors[1, ]^2)
}
integrated_probability <- function(eta, variance, n) {
  q <- normal_quadrature(n)
  as.numeric(plogis(outer(eta, rep(1, n)) +
                   outer(sqrt(pmax(0, variance)), q$nodes)) %*% q$weights)
}
result <- test[c("participant", "material", "item", "response")]
for (name in names(spec$models)) {
  model <- readRDS(file.path(args[3], paste0(name, ".rds")))
  if (nobs(model) != nrow(train) ||
      !isTRUE(all.equal(as.numeric(model.response(model.frame(model))), train$response))) {
    stop("Fitted response frame differs from declared training frame")
  }
  fitted_frame <- model.frame(model)
  for (group in unlist(spec$groups)) {
    if (group %in% names(fitted_frame) &&
        !identical(as.character(fitted_frame[[group]]), as.character(train[[group]]))) {
      stop("Fitted grouping differs from declared training observations")
    }
  }
  for (entry in scaling) {
    column <- entry$column
    if (column %in% names(fitted_frame) &&
        !isTRUE(all.equal(as.numeric(fitted_frame[[column]]),
                         (train[[column]] - entry$center) / entry$scale))) {
      stop("Fitted predictor rows differ from declared training observations")
    }
  }
  eta <- as.numeric(predict(model, newdata=test, re.form=NA, allow.new.levels=TRUE))
  if (spec$family == "binomial") {
    # New reader/item/material effects are unknown. Integrate them rather than
    # equating the conditional inverse-link at zero with a marginal probability.
    variance <- rep(0, nrow(test))
    vc <- as.data.frame(VarCorr(model))
    value <- function(term) if (term == "(Intercept)") rep(1, nrow(test)) else test[[term]]
    for (i in seq_len(nrow(vc))) {
      variance <- variance + if (is.na(vc$var2[i])) {
        vc$vcov[i] * value(vc$var1[i])^2
      } else {
        2 * vc$vcov[i] * value(vc$var1[i]) * value(vc$var2[i])
      }
    }
    if (any(variance < -1e-10)) stop("Negative random-effect variance")
    probability <- integrated_probability(eta, variance, 40L)
    converged <- FALSE
    for (n in c(80L, 160L, 320L)) {
      refined <- integrated_probability(eta, variance, n)
      error <- abs(refined - probability)
      converged <- max(error) < 1e-7
      probability <- refined
      if (converged) break
    }
    if (!converged) {
      for (i in which(error >= 1e-7)) {
        probability[i] <- integrate(function(z) {
          plogis(eta[i] + sqrt(variance[i]) * z) * dnorm(z)
        }, -Inf, Inf, rel.tol=1e-10, abs.tol=1e-10, subdivisions=1000L)$value
      }
    }
    result[[paste0(name, "_latent_mean")]] <- eta
    result[[paste0(name, "_latent_variance")]] <- variance
    probability <- pmax(.Machine$double.eps, pmin(1 - .Machine$double.eps, probability))
    result[[paste0(name, "_prediction")]] <- probability
    result[[paste0(name, "_loss")]] <- -ifelse(test$response == 1, log(probability), log1p(-probability))
  } else if (spec$family == "gaussian") {
    result[[paste0(name, "_prediction")]] <- eta
    result[[paste0(name, "_loss")]] <- (test$response - eta)^2
  } else stop("Unsupported predictive response family")
}
if (!all(vapply(result[endsWith(names(result), "_loss")], function(x) all(is.finite(x)), logical(1)))) {
  stop("Invalid predictive loss")
}
write.table(result, args[4], sep="\t", quote=FALSE, row.names=FALSE)
