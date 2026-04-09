
## code for labelling behaviours in tracks using HMM ====================================
## the code is an adaptation of Hector's HMM code, but applied to the processed tracks data

## preliminaries ------------------------------------------------------------------------

## libraries
pacman::p_load(
    dplyr, data.table, moveHMM, pbapply
    )

## file paths
input_path  <- "./data/processed/processed_tracks.csv"
output_path <- "./data/processed/processed_tracks_hmm.csv"

## loading data
tracks <- read.csv(input_path)
tracks$datetime <- as.POSIXct(tracks$datetime, tz = "GMT")
cat("Loaded", nrow(tracks), "fixes across", length(unique(tracks$track_trip_id)), "trips.\n")

## HMM set up ---------------------------------------------------------------------------

## parameters
mu0 <- c(0.16, 0.64, 2.96)
sigma0 <- c(0.10, 0.72, 0.82)
stepPar0 <- c(mu0, sigma0)

angleMean0 <- c(0, 0, 0)
kappa0 <- c(28.16, 1.66, 12.32)
anglePar0 <- c(angleMean0, kappa0)

## state labels — order must match state numbers from model
behaviour_map <- c("Resting", "Foraging", "Commuting")

## fitting HMM per trip and assigning behaviour states -----------------------------------

run_hmm_on_trip <- function(trip_data, trip_id) {

  # need at least 4 fixes to fit a 3-state HMM
  if (nrow(trip_data) < 4) {
    trip_data$behaviour <- NA
    return(trip_data)
  }

  # prepare data for moveHMM — requires x, y, ID columns
  trip_data$x  <- trip_data$lon
  trip_data$y  <- trip_data$lat
  trip_data$ID <- factor(trip_id)

  df_mh <- tryCatch(
    moveHMM::prepData(trip_data[, c("x", "y", "ID")], type = "LL"),
    error = function(e) {
      message("prepData failed for trip: ", trip_id, " — ", e$message)
      NULL
    }
  )
  if (is.null(df_mh)) {
    trip_data$behaviour <- NA
    return(trip_data)
  }

  # quick diagnostics on step/angle distribution
  steps <- df_mh$step
  angles <- df_mh$angle
  finite_steps <- steps[is.finite(steps) & !is.na(steps)]
  if (length(finite_steps) < 5) {
    message("Trip ", trip_id, ": too few finite steps (", length(finite_steps), "). Skipping.")
    trip_data$behaviour <- NA
    return(trip_data)
  }

  prop_zero <- mean(finite_steps == 0)
  sd_steps <- stats::sd(finite_steps, na.rm = TRUE)

  # handle degenerate cases early
  if (prop_zero > 0.9 || sd_steps < 1e-6) {
    # almost no movement → label as Resting
    message("Trip ", trip_id, ": degenerate steps (prop_zero=", round(prop_zero,3),
            ", sd=", signif(sd_steps,3), ") — assigning Resting.")
    trip_data$behaviour <- "Resting"
    trip_data$x <- NULL; trip_data$y <- NULL; trip_data$ID <- NULL
    return(trip_data)
  }

  # trim extreme outliers to stabilise initial parameter estimates
  q <- stats::quantile(finite_steps, probs = c(0.001, 0.999), na.rm = TRUE)
  finite_trim <- finite_steps[finite_steps >= q[1] & finite_steps <= q[2]]
  if (length(finite_trim) < 5) {
    message("Trip ", trip_id, ": too few steps after trimming. Skipping.")
    trip_data$behaviour <- NA
    trip_data$x <- NULL; trip_data$y <- NULL; trip_data$ID <- NULL
    return(trip_data)
  }

  # sensible initial params from trip
  mu0_trip <- as.numeric(stats::quantile(finite_trim, probs = c(0.25, 0.5, 0.9), na.rm = TRUE))
  mu0_trip[mu0_trip <= 0] <- pmax(mu0_trip[mu0_trip <= 0], 1e-4)
  sigma0_trip <- rep(max(1e-3, stats::sd(finite_trim, na.rm = TRUE)), 3)
  anglePar0_trip <- c(0, 0, 0, 1, 1, 1)
  stepPar0_trip <- c(mu0_trip, sigma0_trip)

  # fitter with multiple randomized starts. return fitted object or NULL.
  try_fit <- function(df_mh, step_start, angle_start, nbStates = 3, attempts = 8) {
    for (attempt in seq_len(attempts)) {
      sp_try <- if (attempt == 1) step_start else step_start * runif(length(step_start), 0.5, 1.7)
      ap_try <- if (attempt == 1) angle_start else c(0,0,0, runif(3, 0.2, 8))[1:(length(angle_start))]
      sp_try[sp_try <= 0] <- 1e-4
      fit <- tryCatch(
        moveHMM::fitHMM(df_mh, nbStates = nbStates, stepPar0 = sp_try, anglePar0 = ap_try, verbose = 0),
        error = function(e) { message("  fit attempt ", attempt, " error: ", e$message); NULL },
        warning = function(w) { message("  fit attempt ", attempt, " warning: ", w$message); invokeRestart("muffleWarning") }
      )
      if (!is.null(fit) && inherits(fit, "moveHMM")) return(fit)
    }
    return(NULL)
  }

  # first try 3-state model
  hmmfit <- try_fit(df_mh, stepPar0_trip, anglePar0_trip, nbStates = 3, attempts = 8)

  # if 3-state fails, try 2-state fallback (more stable for short/degenerate trips)
  if (is.null(hmmfit)) {
    message("Trip ", trip_id, ": 3-state failed — attempting 2-state fallback.")
    # construct 2-state starts from quantiles
    mu0_2 <- as.numeric(stats::quantile(finite_trim, probs = c(0.25, 0.9), na.rm = TRUE))
    sigma0_2 <- rep(max(1e-3, stats::sd(finite_trim, na.rm = TRUE)), 2)
    stepPar0_2 <- c(mu0_2, sigma0_2)
    anglePar0_2 <- c(0,0, 1,1) # two angle means (0,0) and two kappas
    hmmfit <- try_fit(df_mh, stepPar0_2, anglePar0_2, nbStates = 2, attempts = 8)
    if (!is.null(hmmfit)) {
      # map 2-state results -> behaviour labels (no commuting)
      st <- moveHMM::viterbi(hmmfit)
      map2 <- c("Resting", "Foraging")
      trip_data$behaviour <- map2[st]
      trip_data$x <- NULL; trip_data$y <- NULL; trip_data$ID <- NULL
      return(trip_data)
    }
  }

  # if still NULL, give up and assign NA
  if (is.null(hmmfit) || !inherits(hmmfit, "moveHMM")) {
    message("Trip ", trip_id, ": all fit attempts failed. Assigning NA behaviour.")
    trip_data$behaviour <- NA
    trip_data$x <- NULL; trip_data$y <- NULL; trip_data$ID <- NULL
    return(trip_data)
  }

  # final viterbi decode and sanity check
  st <- tryCatch(moveHMM::viterbi(hmmfit), error = function(e) { message("viterbi error: ", e$message); NULL })
  if (is.null(st) || length(st) != nrow(df_mh)) {
    message("Trip ", trip_id, ": viterbi returned invalid states. Assigning NA behaviour.")
    trip_data$behaviour <- NA
    trip_data$x <- NULL; trip_data$y <- NULL; trip_data$ID <- NULL
    return(trip_data)
  }

  # map states to labels — defensive checks to avoid 'argument is of length zero'
  nbStates <- if (!is.null(hmmfit) && is.list(hmmfit) && !is.null(hmmfit$m) && !is.null(hmmfit$m$nbStates)) {
    as.integer(hmmfit$m$nbStates)
  } else {
    NA_integer_
  }

  if (!is.na(nbStates) && nbStates == 3) {
    trip_data$behaviour <- behaviour_map[st]
  } else if (!is.na(nbStates) && nbStates == 2) {
    # map two states -> first two labels
    trip_data$behaviour <- behaviour_map[st]  # behaviour_map[1:2] expected
  } else {
    # fallback: if nbStates unknown, attempt safe mapping, otherwise NA
    if (all(st %in% seq_along(behaviour_map))) {
      trip_data$behaviour <- behaviour_map[st]
    } else {
      message("Trip ", trip_id, ": unknown nbStates and states cannot be mapped. Assigning NA.")
      trip_data$behaviour <- NA
    }
  }

  # drop helper columns
  trip_data$x <- NULL
  trip_data$y <- NULL
  trip_data$ID <- NULL

  return(trip_data)
}

## only run HMM on actual trips (trip_id is not NA and out == TRUE)
trip_ids <- unique(tracks$track_trip_id[!is.na(tracks$trip_id) & tracks$out == TRUE])

cat("Fitting HMM on", length(trip_ids), "foraging trips...\n")

# initialise behaviour column
tracks$behaviour <- NA

results <- pblapply(trip_ids, function(tid) {
  idx <- which(tracks$track_trip_id == tid & !is.na(tracks$trip_id) & tracks$out == TRUE)
  trip_data <- tracks[idx, ]
  trip_result <- run_hmm_on_trip(trip_data, tid)
  list(idx = idx, behaviour = trip_result$behaviour)
})

# write results back to tracks
for (res in results) {
  if (is.null(res) || length(res) == 0) next
  idx <- res$idx
  beh <- res$behaviour
  if (is.null(beh)) {
    tracks$behaviour[idx] <- NA
  } else {
    # ensure lengths match
    if (length(idx) == length(beh)) {
      tracks$behaviour[idx] <- beh
    } else {
      tracks$behaviour[idx] <- NA
      message("Warning: mismatch length for a result; assigned NA for that trip.")
    }
  }
}

## DIAGNOSTICS

# 1) fraction of trips that failed (out==TRUE but behaviour NA)
failed_trips <- unique(tracks$track_trip_id[tracks$out == TRUE & is.na(tracks$behaviour)])
length_failed <- length(failed_trips)
total_trips   <- length(unique(tracks$track_trip_id[tracks$out == TRUE]))
cat("Failed trips:", length_failed, "/", total_trips, "(", round(100*length_failed/total_trips,2), "%)\n")

# 2) inspect one failing trip (replace failed_trips[1] if present)
inspect_trip <- function(tid) {
  idx <- which(tracks$track_trip_id == tid & tracks$out == TRUE)
  trip <- tracks[idx, ]
  df_mh <- moveHMM::prepData(data.frame(x = trip$lon, y = trip$lat, ID = factor(tid)), type = "LL")
  list(
    tid = tid,
    nfix = nrow(trip),
    step_summary = summary(df_mh$step),
    step_quantiles = quantile(df_mh$step, probs = c(0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 1), na.rm=TRUE),
    prop_zero = mean(df_mh$step == 0, na.rm=TRUE),
    sd_step = sd(df_mh$step, na.rm=TRUE),
    head = utils::head(df_mh)
  )
}
# example:
if (length_failed>0) print(inspect_trip(failed_trips[1]))

## number of fixes per behaviour state
cat("Fixes per behaviour state:\n")
print(table(tracks$behaviour, useNA = "ifany"))

## any remaining NAs should be classified as AtColony if trip_id is not NA, otherwise leave as NA

## saving output
write.csv(tracks, output_path, row.names = FALSE)
cat("\nDone. Output saved to:", output_path, "\n")