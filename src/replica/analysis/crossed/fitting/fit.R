# Fit declared crossed mixed models without changing the random-effects structure.
# Usage: Rscript fit.R observations.tsv.gz specification.json output_dir [partial_fits|-] [model]
args <- commandArgs(trailingOnly=TRUE)
if (!length(args) %in% c(3L, 4L, 5L)) stop("Expected observations, specification, output, optional partial fits and model")
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(jsonlite))
spec <- fromJSON(args[2], simplifyVector=FALSE)
requested <- if (length(args) == 5L) args[5] else names(spec$models)
if (!all(requested %in% names(spec$models))) stop("Unknown requested model")
partial <- !setequal(requested, names(spec$models))
if (partial && spec$bootstrap_samples > 0L) stop("Partial fitting cannot run paired bootstrap tests")
data <- read.delim(args[1], check.names=FALSE)
if (dir.exists(args[3])) stop("Output directory already exists")
dir.create(args[3], recursive=TRUE)
writeLines(capture.output(sessionInfo()), file.path(args[3], "session.txt"))
if (!spec$family %in% c("gaussian", "binomial")) stop("Unsupported declared family")
if (spec$family == "binomial" && !all(data$response %in% c(0, 1))) {
  stop("Bernoulli response must be observed binary outcomes")
}
groups <- unlist(spec$groups)
for (group in groups) {
  if (anyNA(data[[group]]) || length(unique(data[[group]])) < 2L) stop("Invalid grouping")
  data[[group]] <- factor(data[[group]])
}
fixed_columns <- unique(unlist(spec$models))
# Center random-only covariates too. With independent random intercepts/slopes,
# changing a slope's origin changes the covariance model, so a separately fitted
# control must use the same transformation as its larger comparison model.
random_columns <- setdiff(unique(unlist(lapply(spec$random_terms, function(term) {
  all.vars(as.formula(paste("~", term)))
}))), groups)
columns <- unique(c(fixed_columns, random_columns))
if (!all(vapply(data[c("response", columns)], function(x) all(is.finite(x)), logical(1)))) {
  stop("Nonfinite complete-case design")
}
if (any(!grepl("^[A-Za-z][A-Za-z0-9_]*$", c(groups, columns)))) stop("Unsafe column name")
# A constant fixed predictor aliases the intercept. Record that structural
# omission using only this fit's observations, before scaling or optimization.
constant <- fixed_columns[vapply(data[fixed_columns], function(x) length(unique(x)) == 1L, logical(1))]
columns <- setdiff(columns, constant)
write_json(list(constant_fixed_terms=constant), file.path(args[3], "fixed_design.json"),
           auto_unbox=TRUE, pretty=TRUE)
# Full-data association fits may scale all observations. CV callers must supply
# their own training transform and must not use these full-data coefficients.
scaling <- lapply(columns, function(column) {
  center <- mean(data[[column]])
  spread <- sd(data[[column]])
  if (!is.finite(spread) || spread == 0) stop(paste("Constant fixed term", column))
  data[[column]] <<- (data[[column]] - center) / spread
  list(column=column, center=center, scale=spread)
})
write_json(scaling, file.path(args[3], "scaling.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
fits <- list()
diagnostics <- list()
coefficients <- list()
# A declared numerical retry may tighten optimization or restart from retained
# parameters. The observations, fixed effects, random effects and convergence
# checks remain unchanged. Store these settings in the immutable specification.
optimizer_control <- modifyList(list(maxfun=200000),
                                if (is.null(spec$optimizer_control)) list() else spec$optimizer_control)
prior <- if (length(args) >= 4L && args[4] != "-") normalizePath(args[4], mustWork=TRUE) else NULL
prior_diagnostics <- if (!is.null(prior)) {
  fromJSON(file.path(prior, "diagnostics.json"), simplifyVector=FALSE)
} else list()
resumed <- list()
validate_resume <- function(model, formula) {
  # A saved model is reusable only for this exact transformed observation frame,
  # formula, likelihood and optimization contract. Never accept warnings by reuse.
  expected <- model.frame(subbars(formula), data=data, drop.unused.levels=TRUE)
  observed <- model.frame(model)
  if (!identical(deparse(formula), deparse(formula(model))) ||
      !identical(names(expected), names(observed)) || nrow(expected) != nrow(observed)) {
    stop("Resume model formula or observation inventory differs")
  }
  for (column in names(expected)) {
    a <- expected[[column]]; b <- observed[[column]]
    if (is.factor(a)) a <- as.character(a)
    if (is.factor(b)) b <- as.character(b)
    if (!isTRUE(all.equal(a, b, tolerance=0, check.attributes=FALSE))) {
      stop(paste("Resume observations differ:", column))
    }
  }
  if ((spec$family == "gaussian" && (!inherits(model, "lmerMod") || isREML(model))) ||
      (spec$family == "binomial" && (!inherits(model, "glmerMod") ||
       family(model)$family != "binomial" || family(model)$link != "logit" ||
       getME(model, "devcomp")$dims[["nAGQ"]] != 1L)) ||
      model@optinfo$optimizer != "bobyqa" || any(weights(model) != 1) ||
      !is.null(model.offset(observed))) stop("Resume likelihood or optimizer differs")
  for (setting in names(optimizer_control)) {
    if (!isTRUE(all.equal(model@optinfo$control[[setting]], optimizer_control[[setting]],
                          tolerance=0))) stop("Resume optimizer settings differ")
  }
}
for (name in requested) {
  terms <- c("1", setdiff(unlist(spec$models[[name]]), constant), unlist(spec$random_terms))
  formula <- as.formula(paste("response ~", paste(terms, collapse=" + ")))
  warnings <- character()
  starting_values <- spec$starting_values[[name]]
  if (!is.null(starting_values)) starting_values <- lapply(starting_values, unlist)
  saved <- if (!is.null(prior)) file.path(prior, paste0(name, ".rds")) else ""
  if (nzchar(saved) && file.exists(saved)) {
    if (is.null(prior_diagnostics[[name]])) stop("Resume model has no diagnostics")
    model <- readRDS(saved)
    validate_resume(model, formula)
    warnings <- unlist(prior_diagnostics[[name]]$warnings, use.names=FALSE)
    if (is.null(warnings)) warnings <- character()
    resumed[[name]] <- list(path=saved, md5=unname(tools::md5sum(saved)))
  } else model <- withCallingHandlers(
    if (spec$family == "gaussian") {
      lmer(formula, data=data, REML=FALSE, start=starting_values,
           control=lmerControl(optimizer="bobyqa", optCtrl=optimizer_control,
                               check.rankX="stop.deficient"))
    } else {
      glmer(formula, data=data, family=binomial(), nAGQ=1L, start=starting_values,
            control=glmerControl(optimizer="bobyqa", optCtrl=optimizer_control,
                                 check.rankX="stop.deficient"))
    },
    warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  messages <- model@optinfo$conv$lme4$messages
  # Singularity is retained as a boundary diagnostic, not used to drop slopes.
  convergence <- messages[!grepl("boundary.*singular", messages)]
  diagnostics[[name]] <- list(
    formula=deparse(formula, width.cutoff=500L), nobs=nobs(model),
    constant_fixed_terms=intersect(unlist(spec$models[[name]]), constant),
    loglik=as.numeric(logLik(model)), parameters=attr(logLik(model), "df"),
    singular=isSingular(model), optimizer_code=model@optinfo$conv$opt,
    convergence_messages=convergence, warnings=warnings,
    variances=as.data.frame(VarCorr(model))
  )
  write_json(diagnostics, file.path(args[3], "diagnostics.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
  saveRDS(model, file.path(args[3], paste0(name, ".rds")))
  if (length(convergence) || any(model@optinfo$conv$opt != 0) || !is.finite(logLik(model))) {
    stop(paste("Unresolved convergence:", name))
  }
  table <- as.data.frame(coef(summary(model)))
  table$term <- rownames(table)
  table$model <- name
  coefficients[[name]] <- table
  fits[[name]] <- model
}
if (!is.null(prior)) {
  write_json(list(source=prior, models=resumed,
                  observations_md5=unname(tools::md5sum(args[1])),
                  specification_md5=unname(tools::md5sum(args[2]))),
             file.path(args[3], "resumption.json"), auto_unbox=TRUE, pretty=TRUE)
}
if (partial) {
  write_json(list(requested=requested, declared=names(spec$models),
                  status="partial model batch; not a complete comparison family"),
             file.path(args[3], "model-selection.json"), auto_unbox=TRUE, pretty=TRUE)
}
write.table(do.call(rbind, coefficients), file.path(args[3], "coefficients.tsv"),
            sep="\t", quote=FALSE, row.names=FALSE)
# Nested fixed-effect contrasts only. Predictive superiority between nonnested
# models requires its separately declared held-out analysis.
tests <- list()
for (contrast in spec$contrasts) {
  small_terms <- setdiff(unlist(spec$models[[contrast$reduced]]), constant)
  large_terms <- setdiff(unlist(spec$models[[contrast$full]]), constant)
  if (spec$bootstrap_samples < 1L) next
  if (!all(small_terms %in% large_terms) || length(large_terms) <= length(small_terms)) {
    stop("Contrast must be a strict nested fixed-effect comparison")
  }
  small <- fits[[contrast$reduced]]
  large <- fits[[contrast$full]]
  statistic <- 2 * as.numeric(logLik(large) - logLik(small))
  if (!is.finite(statistic) || statistic < -1e-6) stop("Invalid observed likelihood ordering")
  statistic <- max(0, statistic)
  # Refit both models for each response drawn under the reduced model, including
  # fresh random effects. Inspect every optimizer; never discard failed draws.
  set.seed(spec$seed)
  draws <- vector("list", spec$bootstrap_samples)
  for (i in seq_len(spec$bootstrap_samples)) {
    warnings <- character()
    result <- tryCatch(withCallingHandlers({
      response <- simulate(small, nsim=1L, use.u=FALSE)[[1L]]
      refits <- list(refit(small, response), refit(large, response))
      diagnostics_i <- lapply(refits, function(model) {
        messages <- model@optinfo$conv$lme4$messages
        list(optimizer_code=model@optinfo$conv$opt,
             convergence=messages[!grepl("boundary.*singular", messages)],
             singular=isSingular(model))
      })
      ok <- all(vapply(diagnostics_i, function(d) {
        !length(d$convergence) && all(d$optimizer_code == 0)
      }, logical(1)))
      delta <- 2 * as.numeric(logLik(refits[[2L]]) - logLik(refits[[1L]]))
      list(draw=i, valid=ok && is.finite(delta) && delta >= -1e-6,
           statistic=if (is.finite(delta)) max(0, delta) else NA_real_,
           diagnostics=diagnostics_i)
    }, warning=function(w) {
      warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
    }), error=function(e) list(draw=i, valid=FALSE, error=conditionMessage(e)))
    result$warnings <- warnings
    draws[[i]] <- result
  }
  valid <- vapply(draws, function(draw) isTRUE(draw$valid), logical(1))
  extreme <- vapply(draws, function(draw) {
    !isTRUE(draw$valid) || draw$statistic >= statistic - 1e-10
  }, logical(1))
  saveRDS(draws, file.path(args[3], paste0(contrast$name, "-bootstrap.rds")))
  tests[[contrast$name]] <- list(
    name=contrast$name, reduced=contrast$reduced, full=contrast$full,
    statistic=statistic, p_value=(1 + sum(extreme)) / (1 + length(draws)),
    requested_samples=spec$bootstrap_samples, samples=sum(valid),
    failed_draws=sum(!valid), failure_policy="count every failed refit as extreme",
    seed=spec$seed, interpretation="unique fixed-effect contribution, not predictive superiority"
  )
}
write_json(tests, file.path(args[3], "tests.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
