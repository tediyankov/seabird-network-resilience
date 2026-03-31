library(moveHMM)

### ---------------------------------------------------------
### 1. SET DIRECTORIES
### ---------------------------------------------------------
input_dir  <- "C:/Users/hkbai/Documents/PHD/Oxnav/R/Data/Resampled/Data/2025"
output_root <- "C:/Users/hkbai/Documents/PHD/Oxnav/R/Output/Checked/"

if (!dir.exists(output_root)) dir.create(output_root, recursive = TRUE)


### ---------------------------------------------------------
### 2. FIND ALL CSV FILES (RECURSIVE)
### ---------------------------------------------------------
csv_files <- list.files(
  path = input_dir,
  pattern = "\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

cat("Found", length(csv_files), "CSV files.\n")


### ---------------------------------------------------------
### 3. FUNCTION TO PROCESS ONE FILE AND MIRROR DIRECTORY
### ---------------------------------------------------------
run_hmm_on_file <- function(csv_path) {
  cat("\nProcessing:", csv_path, "\n")
  
  ### ---- READ DATA ----
  trips <- try(read.csv(csv_path), silent = FALSE)
  if (inherits(trips, "try-error")) {
    warning("Could not read:", csv_path)
    return(NULL)
  }
  
  ### ---- PREP FOR moveHMM ----
  trips <- trips[is.finite(trips$confidence), ]
  trips$x  <- trips$lon
  trips$y  <- trips$lat
  trips$ID <- factor(trips$trip_number)
  
  df <- trips[, c("x", "y", "ID", "datetime")]
  
  df_mh <- try(prepData(df, type = "LL"), silent = FALSE)
  if (inherits(df_mh, "try-error")) {
    warning("prepData failed:", csv_path)
    return(NULL)
  }
  
  ### ---------------------------------------------------------
  ### 4. HMM PARAMETERS (YOUR VALUES)
  ### ---------------------------------------------------------
  mu0    <- c(0.16, 0.64, 2.96)
  sigma0 <- c(0.1, 0.72, 0.82)
  stepPar0 <- c(mu0, sigma0)
  
  angleMean0 <- c(0, 0, 0)
  kappa0     <- c(28.16, 1.66, 12.32)
  anglePar0  <- c(angleMean0, kappa0)
  
  ### ---- FIT THE HMM ----
  hmmfit <- try(
    fitHMM(
      df_mh,
      nbStates = 3,
      stepPar0 = stepPar0,
      anglePar0 = anglePar0,
      verbose = 0
    ),
    silent = TRUE
  )
  
  if (inherits(hmmfit, "try-error")) {
    warning("HMM failed:", csv_path)
    return(NULL)
  }
  
  ### ---- ASSIGN BEHAVIOURAL STATES ----
  state_numbers <- viterbi(hmmfit)
  
  # Map numeric states to meaningful behaviour labels
  # **Adjust the mapping based on your model outputs (state characteristics)**
  # Example mapping based on step length & turning angle assumptions:
  # 1 = Resting, 2 = Foraging, 3 = Commuting
  behaviour_map <- c("Resting", "Foraging", "Commuting")
  trips$behaviour <- behaviour_map[state_numbers]
  
  
  ### ---------------------------------------------------------
  ### 5. MIRROR THE INPUT FOLDER STRUCTURE
  ### ---------------------------------------------------------
  
  rel_path <- substring(csv_path, nchar(input_dir) + 1) 
  rel_dir  <- dirname(rel_path)
  out_dir  <- file.path(output_root, rel_dir)
  
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  out_name <- gsub(".csv", "_HMM.csv", basename(csv_path))
  out_path <- file.path(out_dir, out_name)
  
  ### ---- SAVE OUTPUT ----
  write.csv(trips, out_path, row.names = FALSE)
  
  cat("Saved:", out_path, "\n")
  return(out_path)
}


### ---------------------------------------------------------
### 6. RUN THE HMM ON ALL CSV FILES
### ---------------------------------------------------------
results <- lapply(csv_files, run_hmm_on_file)

cat("\nFinished processing all files.\n")
