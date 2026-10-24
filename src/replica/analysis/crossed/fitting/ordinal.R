# Cumulative-logit crossed models; predict entirely new groups by integration.
# Usage: Rscript ordinal.R observations.tsv specification.json output_dir
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Expected observations, specification and output directory")
suppressPackageStartupMessages(library(ordinal))
suppressPackageStartupMessages(library(jsonlite))
spec <- fromJSON(args[2], simplifyVector=FALSE)
data <- read.delim(args[1], check.names=FALSE)
if (spec$family != "ordinal") stop("Expected declared ordinal family")
categories <- unlist(spec$categories)
if (!identical(categories, sort(unique(categories))) || length(categories) < 3L ||
    anyNA(data$response) || !all(data$response %in% categories)) stop("Invalid ordered categories")
if (!"role" %in% names(data)) data$role <- "train"
train <- data[data$role == "train", , drop=FALSE]
test <- data[data$role == "test", , drop=FALSE]
if (!all(categories %in% train$response)) stop("Training set lacks an outcome category")
groups <- unlist(spec$groups)
for (group in groups) {
  if (anyNA(train[[group]]) || length(unique(train[[group]])) < 2L) stop("Invalid grouping")
  train[[group]] <- factor(train[[group]])
}
if (nrow(test)) for (group in unique(c(groups, "participant", "material", "item"))) {
  if (anyNA(data[[group]]) || length(intersect(as.character(train[[group]]), as.character(test[[group]])))) {
    stop(paste("Independent holdout violated:", group))
  }
}
if (dir.exists(args[3])) stop("Output directory exists")
dir.create(args[3], recursive=TRUE)
writeLines(capture.output(sessionInfo()), file.path(args[3], "session.txt"))
columns <- unique(unlist(spec$models))
if (!all(vapply(data[columns], function(x) all(is.finite(x)), logical(1)))) stop("Nonfinite design")
constant <- columns[vapply(train[columns], function(x) length(unique(x)) == 1L, logical(1))]
columns <- setdiff(columns, constant)
scaling <- lapply(columns, function(column) {
  center <- mean(train[[column]])
  spread <- sd(train[[column]])
  train[[column]] <<- (train[[column]] - center) / spread
  test[[column]] <<- (test[[column]] - center) / spread
  list(column=column, center=center, scale=spread)
})
write_json(scaling, file.path(args[3], "scaling.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
train$response <- ordered(train$response, levels=categories)
diagnostics <- list()
result <- test[c("participant", "material", "item", "response")]
for (name in names(spec$models)) {
  fixed <- setdiff(unlist(spec$models[[name]]), constant)
  fixed_formula <- reformulate(c("1", fixed))
  design <- model.matrix(fixed_formula, train)
  if (qr(design)$rank < ncol(design)) stop("Rank-deficient fixed design")
  formula <- as.formula(paste("response ~", paste(c("1", fixed, unlist(spec$random_terms)), collapse=" + ")))
  warnings <- character()
  fit_once <- function(start) withCallingHandlers(
    clmm(formula, data=train, link="logit", threshold="flexible", nAGQ=1L, Hess=TRUE, start=start,
         control=clmm.control(method="nlminb", maxIter=1000L, gradTol=1e-6,
                              maxLineIter=100L, innerCtrl="warnOnly",
                              iter.max=2000L, eval.max=4000L, rel.tol=1e-10)),
    warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  model <- fit_once()
  attempts <- list(list(loglik=as.numeric(logLik(model)), code=model$optRes$convergence,
                        max_gradient=max(abs(model$gradient))))
  if (model$optRes$convergence != 0L || max(abs(model$gradient)) > 0.01) {
    restarted <- fit_once(start=list(model$coefficients, model$optRes$par[-seq_along(model$coefficients)]))
    attempts[[2]] <- list(loglik=as.numeric(logLik(restarted)), code=restarted$optRes$convergence,
                          max_gradient=max(abs(restarted$gradient)))
    if (as.numeric(logLik(restarted)) >= as.numeric(logLik(model)) - 1e-6) model <- restarted
  }
  variances <- VarCorr(model)
  gradient <- max(abs(model$gradient))
  # An optimizer may visit poorly converged candidate modes. Accept only a
  # final solution whose conditional modes independently converge from zero
  # under strict inner checks and recover the reported Laplace likelihood.
  final_env <- clmm(formula, data=train, link="logit", threshold="flexible",
                    nAGQ=1L, doFit=FALSE,
                    control=clmm.control(maxIter=1000L, maxLineIter=100L,
                                         gradTol=1e-6, innerCtrl="giveError"))
  final_nll <- ordinal:::getNLA(final_env, model$optRes$par)
  inner_gradient <- max(abs(final_env$gradient))
  if (!is.finite(final_nll) || inner_gradient > 1e-6 ||
      abs(final_nll + as.numeric(logLik(model))) > 1e-5) stop("Final ordinal modes or likelihood did not verify")
  diagnostics[[name]] <- list(
    formula=deparse(formula, width.cutoff=500L), nobs=nrow(train),
    loglik=as.numeric(logLik(model)), optimizer_code=model$optRes$convergence,
    optimizer_message=model$optRes$message, max_gradient=gradient,
    optimization_attempts=attempts,
    verified_inner_gradient=inner_gradient,
    verified_likelihood_error=abs(final_nll + as.numeric(logLik(model))),
    hessian_eigenvalues=eigen(model$Hessian, symmetric=TRUE, only.values=TRUE)$values,
    constant_fixed_terms=intersect(unlist(spec$models[[name]]), constant),
    boundary=any(vapply(variances, function(v) min(eigen(v, symmetric=TRUE)$values) < 1e-8, logical(1))),
    variances=unname(lapply(seq_along(variances), function(i) {
      list(group=names(variances)[i], terms=colnames(variances[[i]]),
           covariance=unname(as.matrix(variances[[i]])))
    })), thresholds=model$alpha, coefficients=as.list(model$beta), warnings=warnings
  )
  saveRDS(model, file.path(args[3], paste0(name, ".rds")))
  write_json(diagnostics, file.path(args[3], "diagnostics.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
  if (model$optRes$convergence != 0L || !is.finite(gradient) || gradient > 0.01 ||
      !is.finite(logLik(model))) stop(paste("Unresolved ordinal convergence:", name))
  if (!nrow(test)) next
  design <- model.matrix(fixed_formula, test)
  eta <- if (length(model$beta)) as.numeric(design[, names(model$beta), drop=FALSE] %*% model$beta) else rep(0, nrow(test))
  variance <- rep(0, nrow(test))
  for (v in variances) {
    z <- matrix(vapply(colnames(v), function(term) if (term == "(Intercept)") rep(1, nrow(test)) else test[[term]], numeric(nrow(test))), nrow=nrow(test))
    variance <- variance + rowSums((z %*% v) * z)
  }
  if (any(variance < -1e-10)) stop("Negative predictive variance")
  probability <- matrix(0, nrow(test), length(categories))
  for (i in seq_len(nrow(test))) {
    cumulative <- vapply(model$alpha, function(threshold) {
      if (variance[i] < 1e-14) return(plogis(threshold - eta[i]))
      integrate(function(z) plogis(threshold - eta[i] - sqrt(variance[i]) * z) * dnorm(z),
                -Inf, Inf, rel.tol=1e-10, abs.tol=1e-10, subdivisions=1000L)$value
    }, numeric(1))
    probability[i, ] <- diff(c(0, cumulative, 1))
  }
  if (any(probability < -1e-9) || any(abs(rowSums(probability) - 1) > 1e-9)) stop("Invalid ordinal probabilities")
  result[[paste0(name, "_latent_mean")]] <- eta
  result[[paste0(name, "_latent_variance")]] <- variance
  for (k in seq_along(categories)) result[[paste0(name, "_probability_", categories[k])]] <- probability[, k]
  observed <- probability[cbind(seq_len(nrow(test)), match(test$response, categories))]
  result[[paste0(name, "_loss")]] <- -log(pmax(.Machine$double.eps, observed))
}
if (nrow(test)) write.table(result, file.path(args[3], "predictions.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
