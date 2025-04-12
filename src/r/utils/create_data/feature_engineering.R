# Load required libraries
library(here)
library(readr)
library(dplyr)
library(lubridate)
library(hms)
library(purrr)
library(glue)

# -----------------------------------------------------------------------------
# TEMPORAL FEATURES
# -----------------------------------------------------------------------------
prepare_dates <- function(df) {
  df %>% mutate(
    Check_In_Date = mdy(Check_In_Date),
    Check_In_Time = hms::as_hms(Check_In_Time)
  ) #%>%
  # arrange(Check_In_Date, Check_In_Time)
}

add_temporal_features <- function(df) {
  df %>% mutate(
    Check_In_Day = wday(Check_In_Date, label = TRUE),
    Is_Weekend = Check_In_Day %in% c("Sat", "Sun"),
    Check_In_Week = ceiling(day(Check_In_Date) / 7),
    Check_In_Month = month(Check_In_Date, label = TRUE),
    Check_In_Hour = hour(Check_In_Time)
  )
}

add_time_category <- function(df) {
  df %>% mutate(
    Time_Category = case_when(
      hour(Check_In_Time) < 6 ~ "Late Night",
      hour(Check_In_Time) < 12 ~ "Morning",
      hour(Check_In_Time) < 17 ~ "Afternoon",
      hour(Check_In_Time) < 22 ~ "Evening",
      TRUE ~ "Late Night"
    )
  )
}

convert_semester_to_date <- function(semester_str) {
  # Extract year and semester
  parts <- strsplit(semester_str, " ")[[1]]
  year <- parts[length(parts)]
  semester <- parts[1]
  
  # Map semesters to months
  month <- case_when(
    semester == "Fall" ~ "08",
    semester == "Spring" ~ "01",
    semester == "Summer" ~ "06",
    semester == "Winter" ~ "12",
    TRUE ~ NA_character_
  )
  
  # Combine into date
  paste0(month, "/", "01", "/", year)
}

add_date_features <- function(df) {
  df %>%
    mutate(
      # Convert semester to date
      Semester_Date = mdy(purrr::map(Semester, convert_semester_to_date)),
      # Convert expected graduation to date
      Expected_Graduation_Date = mdy(purrr::map(Expected_Graduation, convert_semester_to_date)),
    )
}

add_graduation_features <- function(df) {
  df %>% mutate(
    # Calculate months until graduation
    Months_Until_Graduation = as.numeric(
      difftime(Expected_Graduation_Date, Semester_Date, units = "days") / 30.44 # average days per month
    )
  )
}

# -----------------------------------------------------------------------------
# ACADEMIC & COURSE FEATURES
# -----------------------------------------------------------------------------
add_course_features <- function(df) {
  df %>% mutate(
    Course_Level = case_when(
      Course_Code_by_Thousands <= 100 ~ "Special",
      Course_Code_by_Thousands <= 3000 ~ "Lower Classmen",
      Course_Code_by_Thousands <= 4000 ~ "Upper Classmen",
      TRUE ~ "Graduate"
    )
  )
}

add_gpa_category <- function(df) {
  df %>% mutate(
    GPA_Category = case_when(
      Cumulative_GPA >= 3.5 ~ "Excellent",
      Cumulative_GPA >= 3.0 ~ "Good",
      Cumulative_GPA >= 2.0 ~ "Satisfactory",
      TRUE ~ "Needs Improvement"
    )
  )
}

add_credit_load_category <- function(df) {
  df %>% mutate(
    # Credit load features
    Credit_Load_Category = case_when(
      Term_Credit_Hours <= 6 ~ "Part Time",
      Term_Credit_Hours <= 12 ~ "Half Time",
      Term_Credit_Hours <= 18 ~ "Full Time",
      TRUE ~ "Overload"
    ),
  )
}

add_class_standing_category <- function(df) {
  df %>% mutate(
    # Renaming column and values for Class_Standing
    Class_Standing_Self_Reported = case_when(
      Class_Standing == "Freshman" ~ "First Year",
      Class_Standing == "Sophomore" ~ "Second Year",
      Class_Standing == "Junior" ~ "Third Year",
      Class_Standing == "Senior" ~ "Fourth Year",
      TRUE ~ Class_Standing
    ),
  )
}

add_class_standing_bgsu <- function(df) {
  df %>% mutate(
    # Class_standing by BGSU's definition
    # https://www.bgsu.edu/academic-advising/student-resources/academic-standing.html
    Class_Standing_BGSU = case_when(
      Total_Credit_Hours_Earned < 30 ~ "Freshman",
      Total_Credit_Hours_Earned < 60 ~ "Sophomore",
      Total_Credit_Hours_Earned < 90 ~ "Junior",
      Total_Credit_Hours_Earned <= 120 ~ "Senior",
      TRUE ~ "Extended"
    ),
  )
}

# -----------------------------------------------------------------------------
# COURSE NAME CATEGORIZATION
# -----------------------------------------------------------------------------
add_course_name_category <- function(df) {
  df %>% mutate(
    Course_Name_Category = case_when(
      # Handle Non-Course Entries First
      Course_Name %in% c("Course Enrollment") ~ "Administrative",

      # Specific Subject Categories
      grepl("Business|Finance|Accounting|Economics|Marketing|Management|Quantitative|
      |Taxation|Planning|Organizational|Behavior|Money|Banking|Auditing|Global Economy|
      |Financial Markets|Selling|Managing Change|Global Strategy",
            Course_Name, ignore.case = TRUE) ~ "Business",
      grepl("Computer|Programming|Data|Software|Network|Database|Algorithm|
      |Operating Systems|Analytics|Computing|Application",
            Course_Name, ignore.case = TRUE) ~ "Computer Science",
      grepl("Mathematics|Calculus|Statistics|Probability|Geometry|Discrete|Algebra|
      |\\bMath\\b|Quantitative|Analytics|Equations|\\bDesign\\b(?=.*Sample)|\\bDesign\\b(?=.*Experimental)|Game Theory",
            Course_Name, ignore.case = TRUE, perl=TRUE) ~ "Mathematics",
      grepl("Statics|Dynamics|Engineering|Structural|Manufacturing|Electrical|Electronic|Thermodynamics|
      |Machine Design|Fluid Power|CAD|BIM|Materials|Modeling|GIS|Geographic Information Systems|
      |Construction|Electric Circuits|Structures|Concrete|Surveying|Estimating|Cost Control",
            Course_Name, ignore.case = TRUE) ~ "Engineering/Technology",
      grepl("Physics|Chemistry|Biology|Astronomy|Earth|Environment|Science(?!.*Computer|.*Social|.*Political|.*Family|
      |.*Food)|Solar System|Sea|Marine|Mechanics|Weather|Climate|Limnology|Ecology|Cosmos|Evolution|Life Through Time|Electricity|Magnetism|Meteorology",
            Course_Name, ignore.case = TRUE, perl = TRUE) ~ "Natural Sciences",
      grepl("Psychology|Sociology|Anthropology|Social|Cultural|Society|Political|Development|
      |Sexuality|Government|Minority|Adolescent|Family|Geography|Organizational|Behavior|GIS|
      |Geographic Information Systems|Corrections|Poverty|Discrimination|Interaction|Women|
      |Juvenile|Delinquency|Interviewing|Observation|Personality|Victimology|Criminology",
            Course_Name, ignore.case = TRUE) ~ "Social Sciences",
      grepl("History|Philosophy|Ethics|Literature|Culture|Language|Art|Religion|Music|Moral|
      |Phonetics|Linguistics|Writing|Composition|Conversation|America|Roman|Drawing|Studio|
      |Performance|Stage|Recital|Civilizations|Thinking|Ideas|Cinematography|Translation|
      |Mythology|Hispanic|Modern World|Existentialism|Media|Strategic Communication",
            Course_Name, ignore.case = TRUE) ~ "Humanities",
      grepl("Education|Teaching|Learning|Childhood|Teacher|Curriculum|Child Development|
      |Families|Field Experience|Communication Development|Design",
            Course_Name, ignore.case = TRUE) ~ "Education",
      grepl("Anatomy|Physiology|Nutrition|Biomechanics|Exercise|Sport|Dietetics|Health|
      |Kinesiology|Nursing|Sexuality|Weight Training|Fitness|Food|Acoustics|Speech|Hearing|Epidemiology",
            Course_Name, ignore.case = TRUE) ~ "Health Sciences",

      # Course Types (Placed after specific subjects)
      grepl("Laboratory|\\bLab\\b", Course_Name, ignore.case = TRUE) ~ "Laboratory",
      grepl("Seminar|Workshop", Course_Name, ignore.case = TRUE) ~ "Seminar",
      grepl("Independent|Special|Practicum|Internship|Field Experience|Topics", # Added Topics here
            Course_Name, ignore.case = TRUE) ~ "Independent/Applied Study",

      # Course Levels (Applied last before Other/No Response)
      grepl("Advanced|III|3|Analysis|Senior|Graduate|Dissertation|Research|Capstone",
            Course_Name, ignore.case = TRUE) ~ "Advanced",
      grepl("Intermediate|II$|II |2|Applied",
            Course_Name, ignore.case = TRUE) ~ "Intermediate",
      grepl("Basic|Elementary|Intro|Introduction|Fundamental|General|Principles|Orientation|Success",
            Course_Name, ignore.case = TRUE) ~ "Introductory",
      grepl("No Response", Course_Name, ignore.case = TRUE) ~ "No Response",

      # Default case
      TRUE ~ "Other"
    )
  )
}

# -----------------------------------------------------------------------------
# COURSE TYPE CATEGORIZATION
# -----------------------------------------------------------------------------
add_course_type_category <- function(df) {
  df %>% mutate(
    Course_Type_Category = case_when(
      is.na(Course_Type) ~ "No Response", # Handle NA values
      Course_Type == "No Response" ~ "No Response", # Handle explicit "No Response"

      # Specific Categories based on Course_Type prefix
      Course_Type %in% c("ACCT", "BA", "ECON", "FIN", "LEGS", "MGMT", "MKT", "MIS", "BIZX", "MBA", "ORGD") ~ "Business & Economics",
      Course_Type %in% c("BIOL", "CHEM", "GEOL", "PHYS", "ASTR", "ENVS", "SEES") ~ "Natural Sciences",
      Course_Type %in% c("CS") ~ "Computer Science",
      Course_Type %in% c("MATH") ~ "Mathematics",
      Course_Type %in% c("STAT", "OR") ~ "Statistics",
      Course_Type %in% c("ART", "CDIS", "CLAS", "COMM", "ENG", "ETHN", "FREN", "GER", "GERM", "HIST", "HUM", "LAT", "PHIL", "POPC", "SPAN", "THFM", "CHIN", "JAPN", "ARTH", "MUCH", "MDIA", "JOUR", "AMPD", "GSW", "MUCT", "ID", "RUSN", "ITAL", "MUS", "CLCV") ~ "Humanities & Arts", # Added GSW, MUCT, ID, RUSN, ITAL, MUS, CLCV
      Course_Type %in% c("AFS", "ROTC") ~ "Military Science",
      Course_Type %in% c("EDAS", "EDCI", "EDEC", "EDFI", "EDHD", "EDIS", "EDL", "EDMS", "EDTL", "HIED", "ACEN", "EIEC") ~ "Education",
      Course_Type %in% c("SOC", "PSYC", "POLS", "GEOG", "JOUR", "WMST", "WS", "ACS", "CAST", "CRJU", "SOWK", "GERO") ~ "Social Sciences",
      Course_Type %in% c("HDFS", "HNRS", "UNIV", "RESC") ~ "Interdisciplinary/Honors",
      Course_Type %in% c("FDST", "NUTR", "EXSC", "PE", "PUBH", "ESHP", "FN", "SM", "NURS", "MLS", "PEG", "AHTH", "DHS", "HMSL") ~ "Health Sciences",
      Course_Type %in% c("EET", "IT", "MET", "QS", "CONS", "IS", "ENGT", "ARCH", "ECET") ~ "Engineering & Technology",

      # Default Case
      TRUE ~ "Other" # Should be empty or very few now
    )
  )
}

# -----------------------------------------------------------------------------
# MAJOR CATEGORIZATION
# -----------------------------------------------------------------------------
add_major_category <- function(df) {
  df %>% mutate(
    # Create cleaned Major string for multi-major check (remove comma-separated minors)
    Major_Cleaned = ifelse(is.na(Major), "", Major),
    Major_Cleaned = gsub(",[^,]+-MIN(:[^,]*)?", "", Major_Cleaned, ignore.case = TRUE),
    Major_Cleaned = gsub("[^,]+-MIN(:[^,]*)?,", "", Major_Cleaned, ignore.case = TRUE),
    
    # Check for multiple majors based on cleaned string
    Has_Multiple_Majors = ifelse(grepl(",", Major_Cleaned, fixed = TRUE), TRUE, FALSE),
    
    # Check for minor presence in the original string
    Has_Minor = ifelse(grepl("-MIN", Major, fixed = TRUE, ignore.case = TRUE), TRUE, FALSE),
    # Is_Pre_Professional = ifelse(grepl("Pre-Med|Pre-Vet|Pre-Law|Pre-Dent|Pre-Prof", Major, ignore.case = TRUE), TRUE, FALSE),
    
    Major_Category = case_when(
      # Handle Undecided/Non-Degree First
      grepl("Undecided|Deciding", Major, ignore.case = TRUE) ~ "Undecided/Deciding",

      # Subject Categories (Split CompSci/Eng, added keywords)
      grepl("Education|Teaching|EDTL|EDAS|EDCI|EDEC|EDFI|EDHD|EDIS|EDL|EDMS|BSED|MED|PHD|Childhood|Adolescent|Intervention|Integrated|College Student Personnel|CSP",
            Major, ignore.case = TRUE) ~ "Education",
      grepl("Business|Finance|Accounting|Economics|Marketing|Management|Supply Chain|SCM|ENTREP|ORGD|BA|BSBA|MBA|BIZX|Tourism|Hospitality|Event|Resort|Attraction|Selling|Human Resource|Sport Management|SPMGT",
            Major, ignore.case = TRUE) ~ "Business & Management",
      grepl("Computer Science|Software|Data|Information Systems|Computing|Analytics|CS", # Moved Analytics here for Busn Analytics
            Major, ignore.case = TRUE) ~ "Computer Science/IT",
      grepl("Engineering|ENGT|Technology|Electronics|ECET|EET|MET|CONS|Construction|Aviation|AVST|Architecture|ARCH|Mechatronics|EMST|Manufacturing|Quality Systems|Visual Communication|VCT",
            Major, ignore.case = TRUE) ~ "Engineering & Technology",
      grepl("Biology|BIOL|Chemistry|CHEM|Physics|PHYS|Geology|GEOL|Astronomy|ASTR|Environment|ENVS|SEES|Marine|Geospatial|Science(?!.*Computer|.*Social|.*Political|.*Family|.*Food|.*Health|.*Sport)",
            Major, ignore.case = TRUE, perl=TRUE) ~ "Natural Sciences",
      grepl("Mathematics|MATH|Statistics|STAT|Actuarial|ASOR",
            Major, ignore.case = TRUE) ~ "Mathematics & Statistics",
      grepl("Health|Nursing|NURS|BSNUR|Nutrition|Dietetics|FN|Food|FDST|Kinesiology|Exercise|EXSC|Sport|SM|HMSL|Athletic Training|MAT|Medical Lab|MEDTECH|MLS|Respiratory Care|RCT|Gerontology|GERO|LTCR|Allied Health|AHTH|DHS|Public Health|PUBH|Communication Disorders|CDIS",
            Major, ignore.case = TRUE) ~ "Health Sciences",
      grepl("Psychology|PSYC|Sociology|SOC|Anthropology|ANTH|Social Work|SOWK|Criminal Justice|CRJU|Political Science|POLS|Geography|GEOG|History|HIST|International|Public Admin|MPA|Family|HDFS|Liberal Arts|PPEL",
            Major, ignore.case = TRUE) ~ "Social Sciences & History", # Combined History here
      grepl("Art|BFA|Music|MUS|MUCT|Theatre|THFM|Film|Media|MDIA|Communication|COMM|JOUR|Literature|English|ENG|Philosophy|PHIL|Ethics|Language|World Languages|WL|Spanish|SPAN|French|FREN|German|GER|Latin|LAT|Russian|RUSN|Italian|ITAL|Chinese|CHIN|Japanese|JAPN|Classics|CLCV|Humanities|HUM|Popular Culture|POPC|Ethnic|ETHN|GSW|Women|Apparel|Merchandising|AMPD|Interior Design|ID",
            Major, ignore.case = TRUE) ~ "Arts & Humanities",

      # General Studies
      grepl("Liberal Studies|Individualized Studies", Major, ignore.case = TRUE) ~ "General Studies",
      grepl("Pre-Med|Pre-Vet|Pre-Law|Pre-Dent|Pre-Prof", Major, ignore.case = TRUE) ~ "Pre-Professional",
      grepl("-MIN|Certificate|Minor", Major, ignore.case = TRUE) ~ "Special Program (Minor/Cert)", # Catch minors explicitly
      is.na(Major) | Major %in% c("No Response", "", "Guest", "Non-Degree") ~ "Unknown/Non-Degree",

      # Default Case
      TRUE ~ "Other"
    )
  ) %>%
    select(-c(Major_Cleaned))
}

# -----------------------------------------------------------------------------
# VISITS FEATURES
# -----------------------------------------------------------------------------
add_visit_features <- function(df) {
  df %>%
    group_by(Student_IDs) %>%
    mutate(
      # Count visits per student
      Total_Visits = n(),
      # Count visits per student per semester
      Semester_Visits = n_distinct(Check_In_Date),
      # Average visits per week
      Avg_Weekly_Visits = Semester_Visits / max(Semester_Week)
    ) %>%
    ungroup()
}

add_week_volume_category <- function(df) {
  df %>%
    mutate(
      Week_Volume = case_when(
        Semester_Week %in% c(4:8, 10:13, 15:16) ~ "High Volume",
        Semester_Week %in% c(1:3, 9, 14, 17) ~ "Low Volume",
        TRUE ~ "Other"
      )
    )
}

# -----------------------------------------------------------------------------
# COURSE LOAD FEATURES
# -----------------------------------------------------------------------------
add_course_load_features <- function(df) {
  df %>%
    group_by(Student_IDs, Semester) %>%
    mutate(
      # Number of unique courses
      Unique_Courses = n_distinct(Course_Number),
      # Mix of course levels
      Course_Level_Mix = n_distinct(Course_Code_by_Thousands),
      # Proportion of advanced courses
      Advanced_Course_Ratio = mean(Course_Level == "Upper Classmen", na.rm = TRUE)
    ) %>%
    ungroup()
}

add_gpa_trend <- function(df) {
  df %>% mutate(
    # Calculate GPA trend (1 for positive, -1 for negative, 0 for no change)
    GPA_Trend = sign(Change_in_GPA),
  )
}

# -----------------------------------------------------------------------------
# RESPONSE/ESQUE FEATURES
# -----------------------------------------------------------------------------
ensure_duration <- function(df) {
  # Calculate duration in minutes
  df %>%
    mutate(
      Duration_In_Min = as.numeric(difftime(
        Check_Out_Time,
        Check_In_Time,
        units = "mins"
      )),
      # Filter out negative durations
      Duration_In_Min = if_else(Duration_In_Min < 0, NA_real_, Duration_In_Min),
    ) %>%
    filter(!is.na(Duration_In_Min))
}

add_session_length_category <- function(df) {
  df %>% mutate(
    # Add session length categories
    Session_Length_Category = case_when(
      Duration_In_Min <= 30 ~ "Short",
      Duration_In_Min <= 90 ~ "Medium",
      Duration_In_Min <= 180 ~ "Long",
      Duration_In_Min > 180 ~ "Extended",
      TRUE ~ NA_character_
    )
  )
}

calculate_occupancy <- function(df) {
  df %>%
    arrange(Check_In_Date, Check_In_Time) %>%
    group_by(Check_In_Date) %>%
    mutate(
      Cum_Arrivals = row_number(),
      Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
        sum(!is.na(Check_Out_Time[1:i]) & 
            Check_Out_Time[1:i] <= Check_In_Time[i])
      }),
      Occupancy = Cum_Arrivals - Cum_Departures
    ) %>%
    select(-c(Cum_Arrivals, Cum_Departures))
}

# -----------------------------------------------------------------------------
# GROUP SIZE FEATURES
# -----------------------------------------------------------------------------
add_group_features <- function(df) {
  df %>%
    mutate(
      Check_In_Timestamp = ymd_hms(paste(Check_In_Date, Check_In_Time))
    ) %>%
    add_count(Check_In_Timestamp, name = "Group_Size") %>%
    mutate(
      Group_Check_In = Group_Size > 1,
      Group_Size_Category = case_when(
        Group_Size == 1 ~ "Individual",
        Group_Size <= 3 ~ "Small Group",
        Group_Size <= 6 ~ "Medium Group",
        TRUE ~ "Large Group"
      )
    ) %>%
    select(-Check_In_Timestamp)
}

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
# Create a safe wrapper function
safely_mutate <- function(df, mutation_fn, required_cols) {
  # Check if all required columns exist
  missing_cols <- setdiff(required_cols, names(df))
  
  if (length(missing_cols) > 0) {
    warning(sprintf("Skipping mutation: Missing columns: %s", 
                   paste(missing_cols, collapse = ", ")))
    return(df)
  }
  
  tryCatch({
    mutation_fn(df)
  }, error = function(e) {
    warning(sprintf("Error in mutation: %s", e$message))
    return(df)
  })
}

# Modify the engineer_features function
engineer_features <- function(df, data_label = "data") {
  cat(glue("\n--- Starting feature engineering for {data_label} ({nrow(df)} rows) ---\n"))
  
  engineered_df <- df %>% 
    safely_mutate(prepare_dates, 
                 c("Check_In_Date", "Check_In_Time")) %>%
    safely_mutate(add_date_features, 
                 c("Semester", "Expected_Graduation")) %>%
    safely_mutate(add_temporal_features, 
                 c("Check_In_Date", "Check_In_Time")) %>%
    safely_mutate(add_time_category, 
                 c("Check_In_Hour")) %>%
    safely_mutate(add_course_features, 
                 c("Course_Code_by_Thousands")) %>%
    safely_mutate(add_course_name_category, 
                 c("Course_Name")) %>%
    safely_mutate(add_course_type_category, 
                 c("Course_Type")) %>%
    safely_mutate(add_major_category, 
                 c("Major")) %>%
    safely_mutate(add_gpa_category, 
                 c("Cumulative_GPA")) %>%
    safely_mutate(add_credit_load_category, 
                 c("Term_Credit_Hours")) %>%
    safely_mutate(add_class_standing_category, 
                 c("Class_Standing")) %>%
    safely_mutate(add_class_standing_bgsu, 
                 c("Total_Credit_Hours_Earned")) %>%
    safely_mutate(ensure_duration, 
                 c("Check_In_Time")) %>%
    safely_mutate(add_session_length_category, 
                 c("Duration_In_Min")) %>%
    safely_mutate(add_visit_features, 
                 c("Student_IDs", "Check_In_Date", "Semester_Week")) %>%
    safely_mutate(add_week_volume_category, 
                 c("Semester_Week")) %>%
    safely_mutate(add_graduation_features, 
                 c("Expected_Graduation_Date", "Semester_Date")) %>%
    safely_mutate(add_course_load_features, 
                 c("Student_IDs", "Semester", "Course_Number", 
                   "Course_Code_by_Thousands", "Course_Level")) %>%
    safely_mutate(add_gpa_trend, 
                 c("Change_in_GPA")) %>%
    safely_mutate(add_group_features, 
                 c("Check_In_Date", "Check_In_Time")) %>%
    safely_mutate(calculate_occupancy, 
                 c("Check_In_Date", "Check_In_Time", "Check_Out_Time")) %>%
    ungroup()
    
  cat(glue("--- Finished feature engineering for {data_label} ---\n"))
  return(engineered_df)
}

# -----------------------------------------------------------------------------
# MAIN EXECUTION (if run directly, usually sourced via make_data.R)
# -----------------------------------------------------------------------------
# Add a check to prevent direct execution if preferred
if (sys.nframe() == 0) { 
  cat("\nRunning feature engineering standalone...\n\n")
  # Create directories if they don't exist
  dir.create(here("data", "processed"), recursive = TRUE, showWarnings = FALSE)
  
  cat("Reading raw data...\n\n")
  data_raw_train <- readr::read_csv(here("data", "raw", "LC_train.csv"))
  data_raw_test <- readr::read_csv(here("data", "raw", "LC_test.csv"))
  
  lc_engineered_train <- engineer_features(data_raw_train, data_label="train")
  lc_engineered_test <- engineer_features(data_raw_test, data_label="test")
  
  cat("Saving engineered data...\n\n")
  # Use a different filename suffix to avoid overwriting external_data output if run standalone
  readr::write_csv(lc_engineered_train, here("data", "processed", "xxxtrain_engineered.csv")) 
  readr::write_csv(lc_engineered_test, here("data", "processed", "xxxtest_engineered.csv"))
  cat("\nFeature engineering standalone complete.\n\n")
}

# -----------------------------------------------------------------------------
# VIEW ENGINEERED DATA
# -----------------------------------------------------------------------------
# View(lc_engineered)

