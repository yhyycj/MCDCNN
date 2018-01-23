# load_data
setwd('/Users/yahuishi/Documents/20171123_CF6_Blade/main')

df_records = data.frame(read.csv('../data/csv/CF6.csv', stringsAsFactors = FALSE))
df_records$tk_flight_datetime <- as.POSIXct(df_records$tk_flight_datetime, format = "%Y-%m-%d %H:%M:%S")

df_events = data.frame(read.csv('../data/csv/events.csv', stringsAsFactors = FALSE))
df_events$Off.wing <- as.POSIXct(df_events$Off.wing, format = "%d/%m/%Y")
df_events$On.wing <- as.POSIXct(df_events$On.wing, format = "%d/%m/%Y")

df_params = data.frame(read.csv('../data/csv/params.csv', stringsAsFactors = FALSE))