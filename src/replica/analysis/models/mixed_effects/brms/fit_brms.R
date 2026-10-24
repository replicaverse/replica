# Gaussian brms adapter. Test responses are never supplied to this script.
args <- commandArgs(TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[1], mustWork = TRUE)
for (pkg in c("brms", "posterior", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(sprintf("BrmsModel requires the R package '%s'; see docs/guides/environment.md", pkg))
    }
}
cfg <- jsonlite::read_json(file.path(root, "config.json"), simplifyVector = TRUE)
stopifnot(cfg$backend == "cmdstanr")
if (!requireNamespace(cfg$backend, quietly = TRUE)) stop("Missing Stan backend: ", cfg$backend)
train <- read.csv(file.path(root, "train.csv"), check.names = FALSE,
                  colClasses = c(subject = "character", item = "character"))
test <- read.csv(file.path(root, "test.csv"), check.names = FALSE,
                 colClasses = c(subject = "character", item = "character"))
stopifnot(!"y" %in% names(test))
for (group in c("subject", "item")) train[[group]] <- factor(train[[group]])
p <- cfg$priors
normal <- function(mean, scale) sprintf("normal(%.17g, %.17g)", mean, scale)
priors <- c(brms::set_prior(normal(p$intercept_mean, p$intercept_scale), class = "b", coef = "Intercept"),
            brms::set_prior(normal(0, p$sigma_scale), class = "sigma"))
if (cfg$n_fixed > 0L) priors <- c(priors, brms::set_prior(normal(0, p$coefficient_scale), class = "b"))
if (cfg$has_group) priors <- c(priors, brms::set_prior(normal(0, p$group_scale), class = "sd"))
if (cfg$has_cor) priors <- c(priors, brms::set_prior(sprintf("lkj(%.17g)", p$correlation_shape), class = "cor"))
messages <- character()
started <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
    brms::brm(brms::bf(as.formula(cfg$formula), center = FALSE), data = train,
        family = stats::gaussian(), prior = priors, backend = cfg$backend,
        algorithm = "sampling", chains = cfg$chains, iter = cfg$iterations,
        warmup = cfg$warmup, cores = cfg$cores, seed = cfg$seed,
        control = list(adapt_delta = cfg$adapt_delta, max_treedepth = cfg$max_treedepth),
        refresh = 0, output_dir = root, save_model = file.path(root, "model.stan")),
    warning = function(w) { messages <<- c(messages, conditionMessage(w)) }
)
fit_seconds <- proc.time()[["elapsed"]] - started
# CmdStan CSVs are retained via output_dir; brms embeds draws in fit.rds.
saveRDS(fit, file.path(root, "fit.rds"))
draws <- posterior::as_draws_array(fit)
summary <- posterior::summarise_draws(draws)
np <- brms::nuts_params(fit)
energy <- split(np$Value[np$Parameter == "energy__"], np$Chain[np$Parameter == "energy__"])
ebfmi <- vapply(energy, function(x) mean(diff(x)^2) / var(x), numeric(1))
finite_extreme <- function(x, fn) if (length(x) && all(is.finite(x))) fn(x) else NA_real_
diag <- list(max_rhat = finite_extreme(summary$rhat, max),
             min_bulk_ess = finite_extreme(summary$ess_bulk, min),
             min_tail_ess = finite_extreme(summary$ess_tail, min),
             divergences = sum(np$Value[np$Parameter == "divergent__"]),
             treedepth_hits = sum(np$Value[np$Parameter == "treedepth__"] >= cfg$max_treedepth),
             min_ebfmi = finite_extreme(ebfmi, min))
# One predictive call: repeated new levels share effects within each draw.
set.seed(cfg$seed + 1L)
mu <- brms::posterior_epred(fit, newdata = test,
    re_formula = if (cfg$group_effects == "population") NA else NULL,
    allow_new_levels = cfg$allow_new_levels, sample_new_levels = "gaussian")
sigma <- posterior::as_draws_matrix(draws)[, "sigma"]
write.table(mu, file.path(root, "expected.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(sigma, file.path(root, "sigma.csv"), row.names = FALSE, col.names = FALSE)
coef_names <- c("b_Intercept", if (cfg$n_fixed > 0) paste0("b_x", seq_len(cfg$n_fixed) - 1L))
coefs <- summary[match(coef_names, summary$variable), ]
stopifnot(!anyNA(coefs$variable))
versions <- setNames(lapply(c("brms", "posterior", cfg$backend), function(p) as.character(packageVersion(p))),
                     c("brms", "posterior", cfg$backend))
jsonlite::write_json(list(diagnostics = diag, coefficients = coefs, warnings = messages,
                         versions = versions, r_version = R.version.string, cmdstan_version = as.character(cmdstanr::cmdstan_version()), fit_seconds = fit_seconds),
                    file.path(root, "summary.json"), pretty = TRUE, auto_unbox = TRUE, na = "null", digits = NA)
writeLines(capture.output(sessionInfo()), file.path(root, "session.txt"))
