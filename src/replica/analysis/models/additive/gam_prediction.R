# Prediction rules shared by the rpy2 gam and standalone bam paths.

replica_gam_smooth_variables <- function(s) {
  # mgcv stores expressions such as factor(participant_id) as strings.
  # Parse their variables without evaluating them; a non-syntactic column
  # name can also occur here literally, so retain it if parsing fails.
  unlist(lapply(c(s$term, s$fterm, s$by), function(term) {
    tryCatch(all.vars(str2lang(term)), error = function(e) term)
  }), use.names = FALSE)
}

replica_gam_excluded_smooths <- function(model) {
  grouping <- c("participant_id", "item_id")
  is_grouped <- vapply(model$smooth, function(s) {
    # Keep the existing bs='re' rule, including terms on other variables.
    inherits(s, "random.effect") ||
      any(replica_gam_smooth_variables(s) %in% grouping)
  }, logical(1))
  # Factor-by smooths have one label per level; exclude every such label.
  vapply(model$smooth, function(s) s$label, character(1))[is_grouped]
}

replica_gam_observed_groups <- function(model, train) {
  # Only rows retained by mgcv establish seen groups. Preserve the original
  # factor encoding for transformations that depend on its level order.
  rows <- seq_len(nrow(train))
  if (length(model$na.action)) rows <- rows[-as.integer(model$na.action)]
  grouping <- intersect(c("participant_id", "item_id"), names(train))
  setNames(lapply(grouping, function(col) {
    unique(as.character(train[[col]][rows]))
  }), grouping)
}

replica_gam_prediction_data <- function(model, train, test, re_predict) {
  # Both IDs are passed by Python even if only one occurs in this fit.
  required <- if (re_predict == "conditional") {
    all.vars(formula(model))
  } else {
    all.vars(delete.response(model$pterms))
  }
  observed <- replica_gam_observed_groups(model, train)
  for (col in names(observed)) {
    if (col %in% required) {
      test[[col]] <- factor(test[[col]], levels = levels(train[[col]]))
      if (anyNA(test[[col]]) || any(!test[[col]] %in% observed[[col]])) {
        stop(sprintf("re_predict=%s but test has unseen %s levels", re_predict, col))
      }
    } else if (re_predict == "exclude") {
      # A placeholder is inert only when every term using this ID is excluded.
      test[[col]] <- factor(observed[[col]][1], levels = levels(train[[col]]))
    }
  }
  test
}

replica_gam_shrink_unseen <- function(model, train, test, prediction_discrete = FALSE) {
  test <- replica_gam_prediction_data(model, train, test, "shrink_unseen")
  observed <- replica_gam_observed_groups(model, train)
  grouping <- intersect(names(observed), all.vars(formula(model)))
  unseen <- vector("list", length(grouping))
  pattern <- integer(nrow(test))
  for (i in seq_along(grouping)) {
    col <- grouping[i]
    if (anyNA(test[[col]])) {
      stop(sprintf("re_predict=shrink_unseen but test has missing %s levels", col))
    }
    test[[col]] <- factor(test[[col]], levels = levels(train[[col]]))
    unseen[[i]] <- !test[[col]] %in% observed[[col]]
    pattern <- pattern + 2^(i - 1L) * unseen[[i]]
    # A seen placeholder is inert after every dependent smooth is excluded.
    test[[col]][unseen[[i]]] <- observed[[col]][1]
  }
  variables <- lapply(model$smooth, replica_gam_smooth_variables)
  labels <- vapply(model$smooth, function(s) s$label, character(1))
  preds <- numeric(nrow(test))
  # At most four batches: each group is either seen or unseen. Preserve row
  # order, known-group effects, offsets and the model's response link.
  for (rows in split(seq_len(nrow(test)), pattern)) {
    missing <- grouping[vapply(unseen, function(x) x[rows[1]], logical(1))]
    exclude <- labels[vapply(variables, function(x) any(x %in% missing), logical(1))]
    preds[rows] <- predict(model, newdata = test[rows, , drop = FALSE],
                          exclude = exclude, type = "response",
                          discrete = prediction_discrete)
  }
  preds
}
