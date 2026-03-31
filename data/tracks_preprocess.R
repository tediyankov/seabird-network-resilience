## code for preprocessing Manx Shearwater GPS tracks ====================================
## To run this code, I used a ZIP file found in ./data which was provided by Paddy
## The code itself is an adaptation of Paddy's approach to cleaning the data

## preliminaries ------------------------------------------------------------------------

## libraries
pacman::p_load(
    dplyr, data.table, raster, stringr, geosphere, lubridate, mixtools, sf, 
    suncalc, ggplot2, circular, lme4, effects, rnaturalearth, rnaturalearthdata, 
    rnaturalearthhires, cowplot, AICcmodavg, ggspatial, pracma, scales, grid
    )

## file paths
dir <- "./data/all_tracks_2025/all_tracks_2025"
output_dir <- "./data/processed/"

## set-up --------------------------------------------------------------------------------

# getting names of csv files in directory, including subfolders
file_names <- list.files(dir, pattern = ".csv", recursive = TRUE)

# creating an empty list to be filled with processed tracks
all_metadata <- list()

# setting the desired interpolated fix interval
desired_interval_secs <- 300

## colony coordinates
colony_coords <- data.frame(
  colony = c("cp","sk","skok","lun","rum","ram","kilda"),
  lon = c(-5.525785, -5.287829, -5.276882, -4.668419, -6.331190, -5.339924, -8.586639),
  lat = c(54.695025, 51.737449, 51.697395, 51.176569, 57.005972, 51.866831, 57.815077),
  lon_min = c(-5.533334, -5.300001, -5.283334, -4.666667, -6.450001, -5.350001, -8.648728),
  lon_max = c(-5.516666, -5.283333, -5.266666, -4.649999, -6.249999, -5.283333, -8.471564),
  lat_min = c(54.616666, 51.733333, 51.700000, 51.166666, 56.589999, 51.866666, 57.88143),
  lat_max = c(54.683334, 51.750001, 51.716667, 51.200001, 57.066667, 51.900001, 57.79175),
  max_elevation_m = c(24, 79, 55, 143, 812, 136, 430)
)
colony_coords$max_visibility_km <- sqrt(13 * colony_coords$max_elevation_m)

## helper functions ---------------------------------------------------------------------

## standardise columb names for lon/lat 
standardise_lonlat_cols <- function(data, col_names) {
  lon_lat_map <- list(
    c("Latitude", "Longitude", "Latitude"),
    c("long", "long", "lat"),
    c("longitude", "longitude", "latitude"),
    c("Lon", "Lon", "Lat"),
    c("Long", "Long", "Lat"),
    c("x", "x", "y"),
    c("Latitude.deg.", "Longitude.deg.", "Latitude.deg."),
    c("Longitude.1", "Longitude.1", "Latitude.1"),
    c("Latitude..deg.", "Longitude..deg.","Latitude..deg."),
    c("londecd", "londecd", "latdecd")
  )
  
  for (entry in lon_lat_map) {
    check_col <- entry[1]; lon_col <- entry[2]; lat_col <- entry[3]
    if (check_col %in% col_names) {
      names(data)[names(data) == lon_col] <- "lon"
      names(data)[names(data) == lat_col] <- "lat"
      return(data)
    }
  }
  
  # special case: i/j columns without lat
  if ("i" %in% col_names && "j" %in% col_names && !("lat" %in% col_names)) {
    return(dplyr::rename(data, lon = j, lat = i))
  }
  
  return(data)
}

## standardise date-time
standardise_datetime <- function(data, col_names) {
  
  parse_iso_datetime <- function(dt_vec) {
    dt_vec <- as.character(unlist(strsplit(dt_vec, "Z")))
    date_time <- as.character(unlist(strsplit(dt_vec, "T")))
    dt_vec <- paste(
      date_time[seq(1, length(date_time) - 1, 2)],
      date_time[seq(2, length(date_time), 2)],
      sep = " "
    )
    as.POSIXct(dt_vec, format = "%Y-%m-%d %H:%M:%S", tz = "GMT")
  }
  
  if ("datetime" %in% col_names) {
    data$datetime <- as.character(data$datetime)
    if (grepl("Z|T", data$datetime[1])) {
      data$datetime <- parse_iso_datetime(data$datetime)
    } else {
      str_lengths <- stringr::str_length(data$datetime)
      data$datetime[str_lengths > 20] <- trimws(data$datetime[str_lengths > 20], "right")
      data$datetime[str_lengths < 20] <- paste0(data$datetime[str_lengths < 20], ":00")
      data$datetime <- as.POSIXct(data$datetime,
        tryFormats = c("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
                       "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
                       "%d/%m/%Y  %H:%M:%S"), tz = "GMT")
    }
    
  } else if ("time" %in% col_names && !("date" %in% col_names)) {
    data$datetime <- as.POSIXct(data$time, tryFormats = "%Y-%m-%dT%H:%M:%SZ", tz = "GMT")
    
  } else if ("date" %in% col_names && !("time" %in% col_names)) {
    data$date <- as.character(data$date)
    if (stringr::str_length(data$date[1]) < 19) {
      data$sec <- as.character(data$sec)
      sec_length <- stringr::str_length(data$sec)
      data$sec[sec_length < 2] <- paste0("0", data$sec[sec_length < 2])
      data$date <- paste0(data$date, ":", data$sec)
    }
    data$datetime <- as.POSIXct(data$date,
      tryFormats = c("%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"), tz = "GMT")
    
  } else if ("date" %in% col_names && "time" %in% col_names) {
    data$time <- paste0(data$time, ":00")
    data$datetime <- as.POSIXct(paste(data$date, data$time),
      format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
    
  } else if ("ime" %in% col_names) {
    data$datetime <- as.POSIXct(data$ime, tryFormats = "%d/%m/%Y %H:%M", tz = "GMT")
    
  } else if ("date.yymmdd" %in% col_names) {
    data$date.yymmdd <- as.character(data$date.yymmdd)
    data$time.hhmmss <- as.character(data$time.hhmmss)
    date_length <- stringr::str_length(data$date.yymmdd)
    time_length <- stringr::str_length(data$time.hhmmss)
    data$date.yymmdd[date_length == 5] <- paste0("0", data$date.yymmdd[date_length == 5])
    for (k in seq_len(nrow(data))) {
      if (time_length[k] < 6) {
        data$time.hhmmss[k] <- paste0(strrep("0", 6 - time_length[k]), data$time.hhmmss[k])
      }
    }
    data$datetime <- as.POSIXct(paste0(data$date.yymmdd, " ", data$time.hhmmss),
      tryFormats = "%y%m%d %H%M%S", tz = "GMT")
    
  } else if ("year" %in% col_names && !("Date" %in% col_names)) {
    if ("hours" %in% col_names) {
      data$datetime <- as.POSIXct(
        paste0(data$year,"/",data$month,"/",data$day," ",data$hours,":",data$minutes,":",data$seconds),
        tryFormats = "%Y/%m/%d %H:%M:%S", tz = "GMT")
    } else {
      data$datetime <- as.POSIXct(
        paste0(data$year,"/",data$month,"/",data$day," ",data$hour,":",data$min,":",data$second),
        tryFormats = "%y/%m/%d %H:%M:%S", tz = "GMT")
    }
    
  } else if ("UTCDate" %in% col_names) {
    data$datetime <- as.POSIXct(paste0(data$UTCDate, " ", data$UTCTime),
      tryFormats = c("%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"), tz = "GMT")
    
  } else if ("Date" %in% col_names && !("POSIXct" %in% col_names)) {
    data <- data[!is.na(data$Time), ]
    time_is_numeric <- is.numeric(data$Time)
    data$Time <- as.character(data$Time)
    if (time_is_numeric) {
      time_length <- stringr::str_length(data$Time)
      for (k in seq_len(nrow(data))) {
        if (time_length[k] < 6) {
          data$Time[k] <- paste0(strrep("0", 6 - time_length[k]), data$Time[k])
        }
      }
      # insert colons to convert e.g. "173100" -> "17:31:00"
      data$Time <- paste0(substr(data$Time, 1, 2), ":",
                          substr(data$Time, 3, 4), ":",
                          substr(data$Time, 5, 6))
      data$datetime <- as.POSIXct(paste0(data$Date, " ", data$Time),
        tryFormats = c("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"), tz = "GMT")
    } else if (grepl("Z", data$Time[1])) {
      data$datetime <- parse_iso_datetime(data$Time)
    } else {
      data$Time <- stringr::str_trunc(trimws(data$Time, "both"), 8, "right", ellipsis = "")
      data$datetime <- as.POSIXct(paste0(data$Date, " ", data$Time),
        tryFormats = c("%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"), tz = "GMT")
    }
    
  } else if ("Time" %in% col_names && "new_time" %in% col_names) {  # ← now correctly at top level
    if (!is.na(data$Time[1]) && stringr::str_length(data$Time[1]) > 10) {
      data$datetime <- parse_iso_datetime(as.character(data$Time))
    } else if ("DateTime" %in% col_names && !is.na(data$DateTime[1])) {
      if (grepl("Z", data$DateTime[1])) {
        data$datetime <- parse_iso_datetime(as.character(data$DateTime))
      } else {
        data$datetime <- as.POSIXct(data$DateTime, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
      }
    } else if ("POSIXct" %in% col_names) {
      if (stringr::str_length(as.character(data$POSIXct[1])) == 19) {
        data$datetime <- as.POSIXct(data$POSIXct,
          tryFormats = c("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"), tz = "GMT")
      } else if ("Date" %in% col_names && "Time" %in% col_names) {
        data$datetime <- as.POSIXct(paste(data$Date, data$Time),
          format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
      }
    }
    
  } else if ("Time" %in% col_names && !("Date" %in% col_names)) {
    data$datetime <- parse_iso_datetime(as.character(data$Time))
    
  } else if ("POSIXct" %in% col_names) {
    data$datetime <- as.POSIXct(data$POSIXct,
      tryFormats = c("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"), tz = "GMT")
    
  } else if ("comma_datetime" %in% col_names) {
    data$comma_datetime <- as.character(data$comma_datetime)
    a <- sapply(strsplit(data$comma_datetime, ","), `[`, 1)
    b <- sapply(strsplit(data$comma_datetime, ","), `[`, 2)
    data$datetime <- as.POSIXct(paste(a, b), format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
    
  } else if ("UTC_datetime" %in% col_names) {
    data$datetime <- as.POSIXct(data$UTC_datetime, format = "%Y-%m-%d %H:%M:%S", tz = "GMT")
  }
  
  return(data)
}

## iterative speed filter
filter_by_speed <- function(df, threshold = 80, max_iter = 20) {
  a <- 0
  while (any(na.omit(df$speed_kmph[-1]) > threshold)) {
    filtered_sections <- rle(df$speed_kmph[-1] > threshold)
    filter <- filtered_sections$values
    filter_positions <- filtered_sections$lengths
    filter_first_positions <- cumsum(filter_positions) + 1
    filter_first_positions <- filter_first_positions[which(!filter)] + 1
    if (filter[1]) filter_first_positions <- c(1, filter_first_positions)
    df <- df[-filter_first_positions, ]
    if (nrow(df) <= 2) return(NULL)
    df$distance_from_last_point_km <- c(NA,
      geosphere::distHaversine(df[1:(nrow(df)-1), c("lon","lat")],
                               df[2:nrow(df), c("lon","lat")]) / 1000)
    df$time_diff_h <- c(NA, diff(as.numeric(df$datetime)) / 3600)
    df$speed_kmph <- df$distance_from_last_point_km / df$time_diff_h
    a <- a + 1
    if (a > max_iter) return(df)  # return as-is after max iterations
  }
  return(df)
}

## recalculate distance, time diff and speed cols
recalc_kinematics <- function(df) {
  df$distance_from_last_point_km <- c(NA,
    geosphere::distHaversine(df[1:(nrow(df)-1), c("lon","lat")],
                             df[2:nrow(df), c("lon","lat")]) / 1000)
  df$time_diff_h <- c(NA, diff(as.numeric(df$datetime)) / 3600)
  df$speed_kmph <- df$distance_from_last_point_km / df$time_diff_h
  return(df)
}

## splitting dataframe into continuois sections based on time gaps
split_on_gaps <- function(df, resolution_secs, gap_multiplier = 10) {
  if (nrow(df) <= 1) return(list(df))
  gaps <- which(df$time_diff_h[-1] > gap_multiplier * resolution_secs / 3600)
  bounds <- c(1, gaps + 1, nrow(df) + 1)
  lapply(seq_len(length(bounds) - 1), function(i) df[bounds[i]:(bounds[i+1]-1), ])
}

## processing loop ---------------------------------------------------------------------

process_file <- function(file, names, colony_coords, desired_interval_secs, dir) {
  
  # read file
  data <- read.csv(paste(dir, file, sep = "/"), skipNul = TRUE)
  col_names <- colnames(data)
  
  # standardise VX column names
  if ("V1" %in% col_names) {
    data <- switch(as.character(length(col_names)),
      "8" = dplyr::rename(data, lon = V3, lat = V2, comma_datetime = V1),
      "9" = dplyr::rename(data, lon = V4, lat = V3, Date = V1, Time = V2),
      "5" = dplyr::rename(data, lon = V2, lat = V1, date = V4, time = V5),
      data
    )
    col_names <- colnames(data)
  }
  
  # parse metadata from filename
  parts <- strsplit(strsplit(file, "/")[[1]][2], "_")[[1]]
  colony <- parts[1]; burrow <- parts[2]; ring <- parts[3]
  gps <- parts[4]; gls    <- parts[5]; campaign <- parts[6]
  date <- as.Date(strsplit(parts[7], "\\.")[[1]][1], format = "%Y%m%d")
  year <- lubridate::year(date)
  month <- lubridate::month(date)
  
  data$colony <- colony; data$burrow <- burrow
  data$ring <- ring; data$gps <- gps
  data$gls <- gls; data$campaign <- campaign
  data$colony_lat <- colony_coords$lat[match(data$colony, colony_coords$colony)]
  data$colony_lon <- colony_coords$lon[match(data$colony, colony_coords$colony)]
  
  # standardise lon/lat columns
  data <- standardise_lonlat_cols(data, col_names)
  
  # ensure numeric lon/lat
  if (!is.numeric(data$lat) || !is.numeric(data$lon)) {
    data$lat <- as.numeric(as.character(data$lat))
    data$lon <- as.numeric(as.character(data$lon))
  }
  
  # filter invalid/out-of-range coordinates
  data <- data[data$lat > 0 & !is.na(data$lat) & !is.na(data$lon), ]
  if (nrow(data) == 0) return(NULL)
  if (data$lat[1] > 1000) { data$lat <- data$lat / 100; data$lon <- data$lon / 100 }
  data <- data[data$lat < 70 & data$lat > 45 & data$lon < 6 & data$lon > -35, ]
  if (nrow(data) == 0) return(NULL)
  
  # standardise datetime
  data <- standardise_datetime(data, col_names)
  
  # filter invalid datetimes
  data <- data[!is.na(data$datetime) &
               data$datetime > as.POSIXct("2004-01-01", tz = "GMT") &
               data$datetime < Sys.time(), ]
  if (nrow(data) == 0) return(NULL)
  
  # attempt to fix reversed month/day
  date_range <- as.numeric(difftime(data$datetime[nrow(data)], data$datetime[1], units = "days"))
  if (date_range > 30 || any(lubridate::month(data$datetime) < 4 | lubridate::month(data$datetime) > 8)) {
    new_datetime <- tryCatch(
      as.POSIXct(as.character(data$datetime), format = "%Y-%d-%m %H:%M:%S", tz = "GMT"),
      error = function(e) NULL
    )
    if (!is.null(new_datetime)) {
      new_range <- as.numeric(difftime(
        new_datetime[max(which(!is.na(new_datetime)))],
        new_datetime[min(which(!is.na(new_datetime)))],
        units = "days"
      ))
      if (new_range < date_range && !any(lubridate::month(new_datetime) < 4 | lubridate::month(new_datetime) > 8)) {
        data$datetime <- new_datetime
      }
    }
  }
  
  # remove remaining NAs and duplicates
  data <- data[!is.na(data$datetime), ]
  dupes <- which((duplicated(data$lat) & duplicated(data$lon)) | duplicated(data$datetime))
  if (length(dupes) > 0) data <- data[-dupes, ]
  if (nrow(data) == 0) return(NULL)
  
  # calculate time diffs and find gaps
  data$time_diff_h <- c(NA, diff(as.numeric(data$datetime)) / 3600)
  original_resolution_secs <- stats::median(data$time_diff_h, na.rm = TRUE) * 3600
  data_sections <- split_on_gaps(data, original_resolution_secs)
  
  # track-level error flags
  error_flags <- new.env()
  error_flags$speed_errors <- FALSE
  error_flags$time_errors  <- FALSE
  error_flags$no_sections  <- 0
  
  # process each continuous section
  processed_sections <- lapply(data_sections, function(sub_data) {
    if (nrow(sub_data) < 2) return(NULL)
    
    median_year <- stats::median(lubridate::year(sub_data$datetime))
    sub_data <- sub_data[lubridate::year(sub_data$datetime) == median_year | is.na(sub_data$datetime), ]
    
    if ("confidence" %in% col_names) {
      sub_data$confidence <- as.numeric(sub_data$confidence)
      sub_data <- sub_data[!is.na(sub_data$confidence) & !is.infinite(sub_data$confidence), ]
    }
    if (nrow(sub_data) <= 2) return(NULL)
    
    # remove time reversals
    sub_data$time_diff_h <- c(NA, diff(as.numeric(sub_data$datetime)) / 3600)
    a <- 0
    while (any(na.omit(sub_data$time_diff_h) <= 0)) {
      if (!is.na(sub_data$time_diff_h[2]) && sub_data$time_diff_h[2] <= 0) {
        sub_data <- sub_data[-1, ]
        sub_data$time_diff_h <- c(NA, diff(as.numeric(sub_data$datetime)) / 3600)
      }
      sub_data <- rbind(sub_data[1, ], sub_data[sub_data$time_diff_h > 0 & !is.na(sub_data$time_diff_h), ])
      if (nrow(sub_data) <= 2) { error_flags$time_errors <- TRUE; return(NULL) }
      sub_data$time_diff_h <- c(NA, diff(as.numeric(sub_data$datetime)) / 3600)
      a <- a + 1
      if (a > 20) { error_flags$time_errors <- TRUE; return(NULL) }
    }
    
    # calculate kinematics and speed filter
    sub_data <- recalc_kinematics(sub_data)
    sub_data <- filter_by_speed(sub_data, threshold = 80)
    if (is.null(sub_data)) { error_flags$speed_errors <- TRUE; return(NULL) }
    
    # re-check gaps after speed filtering
    sub_sections <- split_on_gaps(sub_data, original_resolution_secs)
    
    processed_sub <- lapply(sub_sections, function(sub_sub_data) {
      if (nrow(sub_sub_data) < 4 || sum(sub_sub_data$speed_kmph > 0, na.rm = TRUE) < 3) return(NULL)
      
      # prepare for interpolation: remove stationary points
      sub_sub_data$x1 <- c(abs(diff(sub_sub_data$lon)), 1)
      sub_sub_data$x2 <- c(abs(diff(sub_sub_data$lat)), 1)
      sub_sub_data$x3 <- c(-3600 * sub_sub_data$time_diff_h[-1], NA)
      sub_sub_data <- sub_sub_data[
        (sub_sub_data$x1 != 0 | sub_sub_data$x2 != 0) &
        !is.na(sub_sub_data$x3) & sub_sub_data$x3 < 0, ]
      sub_sub_data <- sub_sub_data[order(sub_sub_data$datetime), ]
      
      # pchip interpolation to desired interval
      sub_sub_data$datetime <- as.numeric(sub_sub_data$datetime)
      idt <- seq(sub_sub_data$datetime[1], sub_sub_data$datetime[nrow(sub_sub_data)], by = desired_interval_secs)
      ix  <- pracma::pchip(sub_sub_data$datetime, sub_sub_data$lon, idt)
      iy  <- pracma::pchip(sub_sub_data$datetime, sub_sub_data$lat, idt)
      Datetime <- as.POSIXct(idt, origin = "1970-01-01", tz = "GMT")
      
      sub_sub_data <- data.frame(
        filename = file,
        original_resolution_secs = original_resolution_secs,
        interpolated_resolution_secs = desired_interval_secs,
        track_id = which(names == file),
        colony = sub_sub_data$colony[1],
        colony_lon = sub_sub_data$colony_lon[1],
        colony_lat = sub_sub_data$colony_lat[1],
        campaign = campaign,
        burrow = burrow,
        ring = ring,
        gps = gps,
        gls = gls,
        datetime = Datetime,
        lon = ix,
        lat = iy
      )
      
      if (nrow(sub_sub_data) <= 1) return(NULL)
      
      # recalculate kinematics and re-filter
      sub_sub_data <- recalc_kinematics(sub_sub_data)
      sub_sub_data <- filter_by_speed(sub_sub_data, threshold = 80)
      if (is.null(sub_sub_data)) return(NULL)
      
      # bearings
      sub_sub_data$instantaneous_gc_bearing <- c(
        geosphere::bearing(sub_sub_data[1:(nrow(sub_sub_data)-1), c("lon","lat")],
                           sub_sub_data[2:nrow(sub_sub_data),     c("lon","lat")]), NA)
      sub_sub_data$gc_bearing_to_home <- geosphere::bearing(
        sub_sub_data[, c("lon","lat")],
        sub_sub_data[, c("colony_lon","colony_lat")])
      
      # rolling mean speed
      standardised_resolution_mins <- 5
      desired_interval_mins <- 15
      desired_interval_fixes <- desired_interval_mins / standardised_resolution_mins
      sub_sub_data$rolling_mean_speed_kmph <- NA
      
      if (nrow(sub_sub_data) > desired_interval_fixes) {
        half_win <- desired_interval_fixes / 2
        sub_sub_data$rolling_mean_speed_kmph <- sapply(
          seq_len(nrow(sub_sub_data)),
          function(i) {
            idx <- seq_len(nrow(sub_sub_data))
            idx <- idx[idx >= ceiling(i - half_win) & idx <= ceiling(i + half_win)]
            mean(sub_sub_data$speed_kmph[idx], na.rm = TRUE)
          }
        )
      }
      
      return(sub_sub_data)
    })
    
    error_flags$no_sections <- error_flags$no_sections + length(processed_sub)
    do.call("rbind", processed_sub)
  })
  
  data <- do.call("rbind", processed_sections)
  if (is.null(data) || nrow(data) < 60 * 30 / desired_interval_secs) return(NULL)

  # calculate distance from colony for each fix
  data$distance_from_home_km <- geosphere::distHaversine(
    data[, c("lon", "lat")],
    data[, c("colony_lon", "colony_lat")]
  ) / 1000
  
  # identify trips (>3km from colony)
  data$out <- data$distance_from_home_km >= 3
  sections <- rle(data$out)
  values <- sections$values
  lengths <- sections$lengths
  values[values & lengths < 24] <- FALSE
  data$out <- rep(values, lengths)
  
  trip_rle <- rle(data$out)
  data$trip_id <- rep(seq_along(trip_rle$lengths), trip_rle$lengths)
  data$track_trip_id <- paste(data$track_id, data$trip_id)
  
  track_trip_list <- lapply(seq_along(trip_rle$lengths), function(this_trip) {
    this_data <- data[data$trip_id == this_trip, ]
    if (any(!this_data$out)) this_data$trip_id <- NA
    this_data
  })
  
  # return both data and metadata as a named list
  metadata <- data.frame(colony, ring, burrow, gps, gls, campaign, year, month,
                         original_resolution_secs,
                         time_errors  = error_flags$time_errors,
                         speed_errors = error_flags$speed_errors,
                         no_sections  = error_flags$no_sections)
  
  list(trips = track_trip_list, metadata = metadata)
}

## run processing -----------------------------------------------------------------------

results <- lapply(file_names, function(f) {
  tryCatch(
    process_file(f, names = file_names, colony_coords = colony_coords,
                 desired_interval_secs = desired_interval_secs, dir = dir),
    error = function(e) { message("Error in file: ", f, "\n", e); NULL }
  )
})

results <- Filter(Negate(is.null), results)
gps_list <- lapply(results, `[[`, "trips")
all_metadata <- lapply(results, `[[`, "metadata")

all_metadata_df <- do.call("rbind", all_metadata)
all_gps <- do.call("rbind", lapply(gps_list, function(x) do.call("rbind", x)))

write.csv(all_gps, paste0(output_dir, "processed_tracks.csv"), row.names = FALSE)
write.csv(all_metadata_df, paste0(output_dir, "metadata.csv"), row.names = FALSE)


View (head(all_gps, 100))

## TODO: add logic for when a trip is complete (ie when bird transitions from OUT == T to OUT == F)
## TODO: add confidence column
## TODO: add logic to filter to confidence below 100m