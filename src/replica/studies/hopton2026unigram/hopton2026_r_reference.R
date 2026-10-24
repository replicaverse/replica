#!/usr/bin/env Rscript
# Regenerates the Hopton et al. (2026b) paper-code anchors.
#
#   --stage fit        fit GAMMs, write per-(lang, measure) partials  [default]
#   --stage folds      write the (obs_id -> fold) map, no fitting (seconds)
#   --stage combine    merge partials into the two fixture TSVs
#   --stage all        fit, then folds, then combine
#
#   --lang     comma-separated subset, or "all"        (default: all)
#   --measure  comma-separated subset, or "all"        (default: all)
#   --variant  "space" (main text) or "no_space" (appendix whitespace
#              ablation). Selects the default --rb-col and suffixes the
#              partials and fixtures, so both variants coexist.
#   --rb-col   Override the RB/MMC surprisal column --variant implies.
#   --data-dir directory holding <lang>.csv
#   --out-dir  fixture output directory                (default: fixtures dir)
#   --partial-dir  where per-(lang, measure) partials live
#   --folds    CV folds                                (default: 10)
#   --cv-seed  fold-map seed                           (default: 250)
#   --n-sim    permutation replicates                  (default: 5000)
#   --save-models  also saveRDS every fit (off by default; the original
#                  script wrote ~3300 .rds files per run)
#
# Port of meco_rep/../mixed_ef_gam.R ("FAIR VERSION"), run on the same
# per-language CSVs REPLICA loads. Emits:
#
#   hopton2026_r_reference.tsv      lang x psychometric x condition x comp
#                                   -> n, mean_dll, sd_dll, se_dll
#                                   (word-level DLL, plus lang = "All" rows)
#   hopton2026_sig_reference.tsv    psychometric x comp -> mean_diff, se_diff,
#                                   p_value, p_holm, p_bh, n
#                                   (RB_Estimate - Standard, across languages)
#
# Under --variant no_space both names gain a "_no_space" suffix. The two
# variants share a fold map: both estimate columns are complete in every
# language, so the complete-case intersection is identical and the ablation
# needs only the fit and combine stages.
#
# Word-level means are the anchor because the paper's figures and permutation
# tests both operate on the word-level DLL; opedal2024context_r_reference.R
# pools per-observation values.
#
# Faithful-to-code notes (mirroring mixed_ef_gam.R exactly):
# - obs_id is assigned by row_number() on the raw frame, before the
#   prev_freq > 0 filter, so it indexes rows of the unfiltered source CSV.
# - lowest_freq is likewise taken over the raw frame's positive freqs.
# - Standard prev_freq comes from the CSV's own prev_freq column, but
#   prev_freq_BW is constructed by lagging the RB column within
#   (subid, model, trialid) ordered by (ianum, obs_id). The two conditions
#   therefore derive the spillover term differently; this is the script's
#   behaviour and is preserved.
# - Both conditions are fit on the same rows: complete cases are computed
#   per condition, then intersected on obs_id ("FAIR VERSION"), so the fold
#   map and the DLL row set are shared.
# - The DV is raw reading time; there is no log transform (as in Opedal
#   et al. (2024) Experiment 2).
# - The baseline formula already carries the random-effect smooths, and
#   add_re_smooths() skips it, so the baseline is RE-smooths-only rather
#   than intercept-only.
# - z-scoring params include len / prev_len, which no formula references.
#   Harmless (intersect() with the frame's names) and preserved.
# - DROPPED: compute_orthogonal_slopes() / apply_orthogonals(). They add
#   orthogonal_surp / orthogonal_pmi columns that no formula in get_formula()
#   references, so they are dead code inherited from the Opedal et al. (2024) osurp
#   conditions and cannot affect any fit. (Same treatment as the dead lmer
#   branch dropped in opedal2024context_r_reference.R.)
# - DROPPED: the unused summarise_dll() helper (defined in the source but
#   never called; the summary tables are built with inline summarise()).
# - The aggregate significance table collapses across languages. The source
#   re-groups the already-word-level frame by everything except `lang` and
#   takes mean(dll) under a "# ensure unique rows" comment, but the frame is
#   already unique at that grain, so the step is not a no-op: MECO reuses the
#   same trialid numbering in every language, and ianum indexes words within
#   a trial, so DLLs for unrelated words at matching (trialid, ianum)
#   positions in different languages get averaged together. `n` in the
#   significance fixture is therefore the number of distinct (trialid, ianum)
#   positions, not the number of word tokens. Preserved as-is because the
#   paper's reported stars come from this computation; see README.md.
#
# Run from the repo root, e.g. one job per language:
#   Rscript src/replica/studies/hopton2026unigram/hopton2026_r_reference.R \
#       --stage fit --lang en --data-dir /path/to/merged_data_no_zero_disagg_retok
#   Rscript src/replica/studies/hopton2026unigram/hopton2026_r_reference.R \
#       --stage combine

shhh <- suppressPackageStartupMessages
shhh(library(dplyr))
shhh(library(tidyr))
shhh(library(readr))

args <- commandArgs(trailingOnly = TRUE)
arg_after <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
has_flag <- function(flag) flag %in% args

stage <- arg_after("--stage", "fit")
if (!stage %in% c("fit", "folds", "combine", "all")) {
  stop("--stage must be fit, folds, combine, or all")
}

ALL_LANGS <- c("du", "en", "fi", "ge", "gr", "he", "it", "ko", "sp", "tr", "ru")
ALL_MEASURES <- c("total_rt", "gaze_rt", "firstfix_rt")
MODELS <- c("mgpt_lc")

# regression_names order and comps are taken verbatim from mixed_ef_gam.R.
REGRESSION_NAMES <- c("surp", "freq", "pmi", "pmi+freq", "surp+freq", "baseline")
COMPS <- c("surp", "freq", "pmi", "surp+freq", "pmi+freq")

parse_subset <- function(flag, all) {
  raw <- arg_after(flag, "all")
  if (identical(raw, "all")) return(all)
  picked <- trimws(strsplit(raw, ",", fixed = TRUE)[[1]])
  bad <- setdiff(picked, all)
  if (length(bad)) {
    stop(sprintf("%s: unknown value(s) %s", flag, paste(bad, collapse = ", ")))
  }
  picked
}

langs <- parse_subset("--lang", ALL_LANGS)
measures <- parse_subset("--measure", ALL_MEASURES)

variant <- arg_after("--variant", "space")
if (!variant %in% c("space", "no_space")) {
  stop("--variant must be space or no_space")
}
# The main-text variant keeps the unsuffixed names the committed fixtures use.
suffix <- if (variant == "space") "" else "_no_space"
rb_estimate_col <- arg_after(
  "--rb-col", sprintf("rb_surp_N500_realigned_%s", variant)
)
data_dir <- arg_after("--data-dir", NULL)
if (stage != "combine" && is.null(data_dir)) {
  stop("--data-dir is required for --stage fit, folds, or all")
}
out_dir <- arg_after("--out-dir", "tests/studies/fixtures")
partial_dir <- arg_after("--partial-dir", file.path(out_dir, "hopton2026_partials"))
num_folds <- as.integer(arg_after("--folds", "10"))
cv_seed <- as.integer(arg_after("--cv-seed", "250"))
n_sim <- as.integer(arg_after("--n-sim", "5000"))
save_models <- has_flag("--save-models")
model_dir <- arg_after("--model-dir", "hopton2026_models")

options(digits = 4)
options(dplyr.summarise.inform = FALSE)

# ---------------------------------------------------------------------------
# Helpers (verbatim from mixed_ef_gam.R)
# ---------------------------------------------------------------------------

std_error <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

safe_sigma <- function(model) {
  out <- tryCatch(stats::sigma(model), error = function(e) NA_real_)
  if (!is.finite(out) || out <= 0) return(NA_real_)
  out
}

make_fold_map <- function(obs_ids, k = 10, seed = 250) {
  set.seed(seed)
  obs_ids <- sort(unique(obs_ids))
  shuffled <- sample(obs_ids, length(obs_ids), replace = FALSE)
  tibble(
    obs_id = shuffled,
    fold = rep(seq_len(k), length.out = length(shuffled))
  )
}

keep_complete_cases <- function(df, cols) {
  sub <- df %>% dplyr::select(all_of(cols))
  na_mat <- is.na(sub)
  fin_mat <- dplyr::mutate(sub, dplyr::across(where(is.numeric), is.finite)) %>%
    as.data.frame()
  has_na <- apply(na_mat, 1, any)
  has_nonfin <- apply(!fin_mat & !na_mat, 1, any)
  df[!(has_na | has_nonfin), , drop = FALSE]
}

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

prepare_dataset <- function(raw_df, rb_col) {
  if (!rb_col %in% names(raw_df)) {
    stop(sprintf("Column '%s' not found in raw data.", rb_col))
  }

  lowest_freq <- min(raw_df$freq[raw_df$freq > 0], na.rm = TRUE)

  raw_df %>%
    dplyr::mutate(
      obs_id = dplyr::row_number(),
      subid = as.factor(subid),
      trialid = as.factor(trialid)
    ) %>%
    dplyr::filter(prev_freq > 0) %>%
    dplyr::mutate(
      freq = -log2(dplyr::if_else(freq == 0, lowest_freq, freq)),
      prev_freq = -log2(prev_freq)
    ) %>%
    dplyr::group_by(subid, model, trialid) %>%
    dplyr::arrange(ianum, obs_id, .by_group = TRUE) %>%
    dplyr::mutate(
      freq_BW = .data[[rb_col]],
      prev_freq_BW = dplyr::lag(.data[[rb_col]], 1)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      pmi = freq - surp,
      prev_pmi = prev_freq - prev_surp,
      pmi_BW = freq_BW - surp,
      prev_pmi_BW = prev_freq_BW - prev_surp
    ) %>%
    dplyr::select(-starts_with("prev2"))
}

CONDITION_SPECS <- list(
  Standard = list(
    freq_col = "freq", prev_freq_col = "prev_freq",
    pmi_col = "pmi", prev_pmi_col = "prev_pmi"
  ),
  RB_Estimate = list(
    freq_col = "freq_BW", prev_freq_col = "prev_freq_BW",
    pmi_col = "pmi_BW", prev_pmi_col = "prev_pmi_BW"
  )
)

# ---------------------------------------------------------------------------
# Per-fold z-scoring (training stats applied to both splits)
# ---------------------------------------------------------------------------

compute_zscore_params <- function(train_df, spec) {
  cols <- unique(c(
    spec$freq_col, spec$prev_freq_col,
    spec$pmi_col, spec$prev_pmi_col,
    "surp", "prev_surp",
    "len", "prev_len"
  ))
  cols <- intersect(cols, names(train_df))
  params <- lapply(cols, function(col) {
    x <- train_df[[col]]
    list(mean = mean(x, na.rm = TRUE), sd = stats::sd(x, na.rm = TRUE))
  })
  names(params) <- cols
  params
}

apply_zscoring <- function(df, params) {
  for (col in names(params)) {
    if (!col %in% names(df)) next
    m <- params[[col]]$mean
    s <- params[[col]]$sd
    if (is.finite(m) && is.finite(s) && s > 0) {
      df[[col]] <- (df[[col]] - m) / s
    }
  }
  df
}

# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

add_re_smooths <- function(formula_string, reg_name) {
  if (reg_name == "baseline") return(formula_string)
  paste0(formula_string, " + s(subid, bs = 're') + s(trialid, bs = 're')")
}

get_formula <- function(reg_name, psychometric, spec) {
  f <- spec$freq_col
  pf <- spec$prev_freq_col
  p <- spec$pmi_col
  pp <- spec$prev_pmi_col

  switch(
    reg_name,
    "surp" = sprintf(
      "%s ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6)",
      psychometric
    ),
    "freq" = sprintf(
      "%s ~ s(%s, bs = 'cr', k = 6) + s(%s, bs = 'cr', k = 6)",
      psychometric, f, pf
    ),
    "pmi" = sprintf(
      "%s ~ s(%s, bs = 'cr', k = 6) + s(%s, bs = 'cr', k = 6)",
      psychometric, p, pp
    ),
    "pmi+freq" = sprintf(
      paste0(
        "%s ~ s(%s, bs = 'cr', k = 6) + s(%s, bs = 'cr', k = 6)",
        " + s(%s, bs = 'cr', k = 6) + s(%s, bs = 'cr', k = 6)"
      ),
      psychometric, p, pp, f, pf
    ),
    "surp+freq" = sprintf(
      paste0(
        "%s ~ s(surp, bs = 'cr', k = 6) + s(prev_surp, bs = 'cr', k = 6)",
        " + s(%s, bs = 'cr', k = 6) + s(%s, bs = 'cr', k = 6)"
      ),
      psychometric, f, pf
    ),
    "baseline" = sprintf(
      "%s ~ s(subid, bs = 're') + s(trialid, bs = 're')", psychometric
    ),
    stop(sprintf("Unknown regression name: %s", reg_name))
  )
}

fit_one_model <- function(formula_string, reg_name, train_df) {
  formula_string <- add_re_smooths(formula_string, reg_name)
  mgcv::bam(
    stats::as.formula(formula_string),
    data = train_df, method = "fREML", discrete = TRUE, nthreads = 4
  )
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

run_cv_for_regression <- function(df_cond, psychometric, reg_name, lang,
                                  llm_model, condition, spec) {
  formula_string <- get_formula(reg_name, psychometric, spec)
  fold_ids <- sort(unique(df_cond$fold))
  out <- vector("list", length(fold_ids))

  for (j in seq_along(fold_ids)) {
    fold_id <- fold_ids[j]

    train_df <- df_cond %>% dplyr::filter(fold != fold_id)
    test_df <- df_cond %>% dplyr::filter(fold == fold_id)
    if (nrow(train_df) == 0 || nrow(test_df) == 0) next

    zscore_params <- compute_zscore_params(train_df, spec)
    train_df <- apply_zscoring(train_df, zscore_params)
    test_df <- apply_zscoring(test_df, zscore_params)

    fit <- fit_one_model(formula_string, reg_name, train_df)

    if (save_models) {
      dir.create(file.path(model_dir, condition), recursive = TRUE,
                 showWarnings = FALSE)
      saveRDS(fit, file.path(
        model_dir, condition,
        sprintf("%s-%s-%s-%s-%02d.rds",
                lang, llm_model, psychometric, reg_name, fold_id)
      ))
    }

    sd_hat <- safe_sigma(fit)
    mu_hat <- tryCatch(
      predict(fit, newdata = test_df),
      error = function(e) rep(NA_real_, nrow(test_df))
    )
    ll <- if (is.finite(sd_hat)) {
      stats::dnorm(test_df[[psychometric]], mean = mu_hat, sd = sd_hat,
                   log = TRUE)
    } else {
      rep(NA_real_, nrow(test_df))
    }

    out[[j]] <- tibble(
      obs_id = test_df$obs_id,
      fold = fold_id,
      trialid = test_df$trialid,
      ianum = test_df$ianum,
      loglik = ll,
      names = reg_name,
      lang = lang,
      psychometric = psychometric,
      model = llm_model,
      condition = condition
    )
  }

  dplyr::bind_rows(out)
}

# ---------------------------------------------------------------------------
# Stage: fit
# ---------------------------------------------------------------------------

# Load and prepare one language. Shared by the fit and folds stages.
load_cell_frame <- function(lang) {
  csv_path <- file.path(data_dir, paste0(lang, ".csv"))
  if (!file.exists(csv_path)) {
    stop(sprintf("missing %s (pass --data-dir)", csv_path))
  }
  raw_df <- readr::read_csv(csv_path, show_col_types = FALSE)
  prepare_dataset(raw_df, rb_estimate_col)
}

# Complete-case intersection and fold map for one cell. Factored out so the
# folds stage emits exactly the assignment the fit stage used: any change here
# moves both together.
cell_common_and_folds <- function(df_model, psychometric, lang, llm_model) {
  needed_std <- c(psychometric, "surp", "prev_surp",
                  "freq", "prev_freq", "pmi", "prev_pmi")
  needed_rb <- c(psychometric, "surp", "prev_surp",
                 "freq_BW", "prev_freq_BW", "pmi_BW", "prev_pmi_BW")

  std_eval <- df_model %>% keep_complete_cases(needed_std)
  rb_eval <- df_model %>% keep_complete_cases(needed_rb)
  common_ids <- intersect(std_eval$obs_id, rb_eval$obs_id)

  if (length(common_ids) < min(nrow(std_eval), nrow(rb_eval))) {
    warning(sprintf(
      "Row mismatch in %s / %s / %s: Standard=%d, RB=%d, Intersection=%d",
      lang, llm_model, psychometric,
      nrow(std_eval), nrow(rb_eval), length(common_ids)
    ))
  }
  if (length(common_ids) < num_folds) {
    warning(sprintf("Skipping %s / %s / %s: only %d common rows.",
                    lang, llm_model, psychometric, length(common_ids)))
    return(NULL)
  }

  list(
    common_ids = common_ids,
    fold_map = make_fold_map(common_ids, k = num_folds, seed = cv_seed)
  )
}

fit_one_cell <- function(lang, psychometric) {
  df_all <- load_cell_frame(lang)

  all_logliks <- list()

  for (llm_model in MODELS) {
    df_model <- df_all %>% dplyr::filter(model == llm_model)

    cf <- cell_common_and_folds(df_model, psychometric, lang, llm_model)
    if (is.null(cf)) next
    fold_map <- cf$fold_map

    df_shared <- df_model %>%
      dplyr::filter(obs_id %in% cf$common_ids) %>%
      dplyr::inner_join(fold_map, by = "obs_id") %>%
      dplyr::arrange(fold, obs_id)

    for (condition in names(CONDITION_SPECS)) {
      spec <- CONDITION_SPECS[[condition]]
      for (reg_name in REGRESSION_NAMES) {
        cat(sprintf("  %s / %s / %s / %s\n",
                    lang, psychometric, condition, reg_name))
        res <- run_cv_for_regression(
          df_cond = df_shared, psychometric = psychometric,
          reg_name = reg_name, lang = lang, llm_model = llm_model,
          condition = condition, spec = spec
        )
        if (nrow(res) > 0) all_logliks[[length(all_logliks) + 1]] <- res
      }
    }
  }

  loglik_df <- dplyr::bind_rows(all_logliks)
  if (nrow(loglik_df) == 0) return(invisible(NULL))

  baseline_ll <- loglik_df %>%
    dplyr::filter(names == "baseline") %>%
    dplyr::select(obs_id, fold, lang, psychometric, model, condition,
                  baseline_loglik = loglik)

  target_ll <- loglik_df %>%
    dplyr::filter(names %in% COMPS) %>%
    dplyr::select(obs_id, fold, lang, psychometric, model, condition,
                  names, trialid, ianum, target_loglik = loglik)

  dll_by_row <- target_ll %>%
    dplyr::inner_join(
      baseline_ll,
      by = c("obs_id", "fold", "lang", "psychometric", "model", "condition")
    ) %>%
    dplyr::mutate(dll = target_loglik - baseline_loglik, comp = names)

  # Word-level aggregation (MECO style): average over participants.
  dll_word_level <- dll_by_row %>%
    dplyr::group_by(condition, lang, psychometric, model, comp,
                    trialid, ianum) %>%
    dplyr::summarise(dll = mean(dll, na.rm = TRUE), .groups = "drop")

  dir.create(partial_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(
    partial_dir, sprintf("dll_%s_%s%s.tsv", lang, psychometric, suffix)
  )
  readr::write_tsv(dll_word_level, path)
  cat(sprintf("wrote %d word-level rows -> %s\n", nrow(dll_word_level), path))
  invisible(NULL)
}

run_fit <- function() {
  for (lang in langs) {
    for (psychometric in measures) {
      cat(sprintf("\n=========== %s / %s ===========\n", lang, psychometric))
      fit_one_cell(lang, psychometric)
    }
  }
}

# ---------------------------------------------------------------------------
# Stage: folds
# ---------------------------------------------------------------------------
# Emits the (obs_id -> fold) assignment without fitting anything, so REPLICA
# study can replay R's folds. make_fold_map() draws through R's sample(), whose
# Mersenne-Twister stream numpy cannot reproduce, so the assignment has to be
# exported rather than recomputed on the Python side. Runs in seconds: no
# bam() calls, only the data prep and the complete-case intersection.

folds_one_cell <- function(lang, psychometric, df_all) {
  out <- list()
  for (llm_model in MODELS) {
    df_model <- df_all %>% dplyr::filter(model == llm_model)
    cf <- cell_common_and_folds(df_model, psychometric, lang, llm_model)
    if (is.null(cf)) next
    out[[length(out) + 1]] <- cf$fold_map %>%
      dplyr::mutate(lang = lang, psychometric = psychometric, model = llm_model) %>%
      dplyr::select(lang, psychometric, model, obs_id, fold) %>%
      dplyr::arrange(obs_id)
  }
  dplyr::bind_rows(out)
}

run_folds <- function() {
  fold_dir <- file.path(partial_dir, "folds")
  dir.create(fold_dir, recursive = TRUE, showWarnings = FALSE)

  for (lang in langs) {
    df_all <- load_cell_frame(lang)
    for (psychometric in measures) {
      fm <- folds_one_cell(lang, psychometric, df_all)
      if (!nrow(fm)) next
      path <- file.path(fold_dir, sprintf("folds_%s_%s.tsv", lang, psychometric))
      readr::write_tsv(fm, path)
      cat(sprintf("%-4s %-12s %7d rows -> %s\n",
                  lang, psychometric, nrow(fm), path))
    }
  }

  # Merge every fold partial present, so subset runs accumulate. The merged
  # map covers every cell, so it stays gzipped under the partials directory;
  # the committed artifact is the parquet the study's data-extraction step
  # writes from it.
  files <- list.files(fold_dir, pattern = "^folds_.*\\.tsv$", full.names = TRUE)
  merged <- dplyr::bind_rows(lapply(files, readr::read_tsv, show_col_types = FALSE))
  path <- file.path(fold_dir, "hopton2026_folds.tsv.gz")
  readr::write_tsv(merged, path)
  cat(sprintf("\nmerged %d fold rows across %d cells -> %s\n",
              nrow(merged), length(files), path))
}

# ---------------------------------------------------------------------------
# Stage: combine
# ---------------------------------------------------------------------------

run_combine <- function() {
  shhh(library(jmuOutlier))

  # Keep this variant's partials only: the space run leaves unsuffixed files
  # and the ablation leaves "_no_space" ones in the same directory.
  # list.files() takes an extended regex with no lookaround, so the split
  # happens after listing rather than in the pattern.
  files <- list.files(partial_dir, pattern = "^dll_.*\\.tsv$", full.names = TRUE)
  is_no_space <- grepl("_no_space\\.tsv$", basename(files))
  files <- if (variant == "space") files[!is_no_space] else files[is_no_space]
  if (!length(files)) stop(sprintf("no partials found in %s", partial_dir))
  dll_word_level <- dplyr::bind_rows(lapply(
    files, readr::read_tsv, show_col_types = FALSE
  ))

  # --- fixture 1: per-language and pooled word-level DLL -------------------
  per_lang <- dll_word_level %>%
    dplyr::group_by(lang, psychometric, model, condition, comp) %>%
    dplyr::summarise(
      n = dplyr::n(),
      mean_dll = mean(dll, na.rm = TRUE),
      sd_dll = stats::sd(dll, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(se_dll = sd_dll / sqrt(n))

  pooled <- dll_word_level %>%
    dplyr::group_by(psychometric, model, condition, comp) %>%
    dplyr::summarise(
      n = dplyr::n(),
      mean_dll = mean(dll, na.rm = TRUE),
      sd_dll = stats::sd(dll, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(se_dll = sd_dll / sqrt(n), lang = "All")

  ref <- dplyr::bind_rows(per_lang, pooled) %>%
    dplyr::select(lang, psychometric, model, condition, comp, n,
                  mean_dll, sd_dll, se_dll) %>%
    dplyr::arrange(lang, psychometric, model, condition, comp)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ref_path <- file.path(out_dir, sprintf("hopton2026_r_reference%s.tsv", suffix))
  readr::write_tsv(ref, ref_path)
  cat(sprintf("wrote %d rows -> %s\n", nrow(ref), ref_path))

  # --- fixture 2: RB vs Standard paired permutation tests ------------------
  # Aggregated across languages, matching frequency_metric_tests_agg.csv.
  sig <- dll_word_level %>%
    dplyr::group_by(condition, psychometric, model, comp, trialid, ianum) %>%
    dplyr::summarise(dll = mean(dll), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = condition, values_from = dll) %>%
    dplyr::mutate(diff = RB_Estimate - Standard) %>%
    dplyr::group_by(psychometric, model, comp) %>%
    dplyr::summarise(
      mean_diff = mean(diff, na.rm = TRUE),
      se_diff = std_error(diff),
      p_value = {
        x <- diff[is.finite(diff)]
        if (length(x) >= 2) {
          jmuOutlier::perm.test(x, num.sim = n_sim)$p.value
        } else {
          NA_real_
        }
      },
      n = sum(is.finite(diff)),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      p_holm = stats::p.adjust(p_value, method = "holm"),
      p_bh = stats::p.adjust(p_value, method = "BH")
    ) %>%
    dplyr::arrange(psychometric, comp)

  sig_path <- file.path(out_dir, sprintf("hopton2026_sig_reference%s.tsv", suffix))
  readr::write_tsv(sig, sig_path)
  cat(sprintf("wrote %d rows -> %s\n", nrow(sig), sig_path))
}

# ---------------------------------------------------------------------------

if (stage %in% c("fit", "all")) run_fit()
if (stage %in% c("folds", "all")) run_folds()
if (stage %in% c("combine", "all")) run_combine()
