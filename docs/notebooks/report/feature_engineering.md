# Feature Engineering

The complete feature engineering implementation can be found in our [source code](https://github.com/adabwana/sp25-m7560-final-project/blob/master/src/r/utils/create_data/feature_engineering.R).

## Temporal Feature Engineering

Our feature engineering process began with **_temporal data extraction_** using the `lubridate` and `hms` packages.

```r
prepare_dates <- function(df) {
 df %>% mutate(
   Check_In_Date = mdy(Check_In_Date),
   Check_In_Time = hms::as_hms(Check_In_Time)
 )
}
```

From these validated timestamps, we construct several temporal features:

```r
add_temporal_features <- function(df) {
 df %>% mutate(
    Check_In_Day = wday(Check_In_Date, label = TRUE),
    Is_Weekend = Check_In_Day %in% c("Sat", "Sun"),
    Check_In_Week = ceiling(day(Check_In_Date) / 7),
    Check_In_Month = month(Check_In_Date, label = TRUE),
    Check_In_Hour = hour(Check_In_Time)
 )
}
```

Analysis of visit patterns revealed a **_non-linear relationship_** between `Check_In_Hour` and `Duration` variables. This observation prompted the creation of a more nuanced `Time_Category` variable with distinct periods:

```r
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
```

The `Semester` and `Expected_Graduation` variables presented a **_dimensionality challenge_** due to their categorical semester format (e.g., "Fall 2016"). We first converted these strings into actual date objects representing the start of the semester/expected graduation semester.

```r
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
  
  # Combine into date (day is arbitrary, set to 01)
  paste0(month, "/", "01", "/", year)
}

add_date_features <- function(df) {
  df %>%
    mutate(
      # Convert semester to date
      Semester_Date = mdy(purrr::map_chr(Semester, convert_semester_to_date)),
      # Convert expected graduation to date
      Expected_Graduation_Date = mdy(purrr::map_chr(Expected_Graduation, convert_semester_to_date)),
    )
}
```

Using these dates, we then calculated a numeric **_'Months_Until_Graduation'_** metric, effectively reducing complexity while maintaining predictive potential.

```r
add_graduation_features <- function(df) {
  df %>% mutate(
    # Calculate months until graduation
    Months_Until_Graduation = as.numeric(
      difftime(Expected_Graduation_Date, Semester_Date, units = "days") / 30.44 # average days per month
    )
  )
}
```

## Course-Related Features

The `Course_Code_by_Thousands` variable was used to create a `Course_Level` feature:

```r
add_course_features <- function(df) {
  df %>% mutate(
    Course_Level = case_when(
      Course_Code_by_Thousands <= 100 ~ "Special",
      Course_Code_by_Thousands <= 3000 ~ "Lower Classmen",
      Course_Code_by_Thousands <= 4000 ~ "Upper Classmen",
      TRUE ~ "Graduate" # Includes codes > 4000
    )
  )
}
```

To capture academic performance context, we developed categorical features for `Cumulative_GPA` and `Term_Credit_Hours`:

```r
add_gpa_category <- function(df) {
  df %>% mutate(
    GPA_Category = case_when(
      Cumulative_GPA >= 3.5 ~ "Excellent",
      Cumulative_GPA >= 3.0 ~ "Good",
      Cumulative_GPA >= 2.0 ~ "Satisfactory",
      TRUE ~ "Needs Improvement" # Includes NA GPA values implicitly
    )
  )
}
```

```r
add_credit_load_category <- function(df) {
  df %>% mutate(
    # Credit load features
    Credit_Load_Category = case_when(
      Term_Credit_Hours <= 6 ~ "Part Time",
      Term_Credit_Hours <= 12 ~ "Half Time",
      Term_Credit_Hours <= 18 ~ "Full Time",
      TRUE ~ "Overload" # Includes > 18 hours
    ),
  )
}
```

## Student Classification Features

The dataset exhibited an **_unexpected concentration_** of '_Senior_' classifications in the original `Class_Standing` variable. Further investigation suggested this might stem from students accumulating excess credits for senior status without necessarily being in their final year. To address this potential ambiguity while preserving useful information, we implemented a **_dual classification approach_**.

First, we recoded the original `Class_Standing` variable, preserving potentially valuable self-reported information as `Class_Standing_Self_Reported`.

```r
add_class_standing_category <- function(df) {
  df %>% mutate(
    # Renaming column and values for Class_Standing
    Class_Standing_Self_Reported = case_when(
      Class_Standing == "Freshman" ~ "First Year",
      Class_Standing == "Sophomore" ~ "Second Year",
      Class_Standing == "Junior" ~ "Third Year",
      Class_Standing == "Senior" ~ "Fourth Year",
      TRUE ~ Class_Standing # Keeps 'Graduate', 'Other', etc.
    ),
  )
}
```

Complementing this, we developed a more **_objective BGSU Standing metric_** (`Class_Standing_BGSU`) based strictly on earned credit hours, following official university definitions. This dual approach provides both self-reported and objective perspectives.

```r
add_class_standing_bgsu <- function(df) {
  df %>% mutate(
    # Class_standing by BGSU's definition
    # https://www.bgsu.edu/academic-advising/student-resources/academic-standing.html
    Class_Standing_BGSU = case_when(
      Total_Credit_Hours_Earned < 30 ~ "Freshman",
      Total_Credit_Hours_Earned < 60 ~ "Sophomore",
      Total_Credit_Hours_Earned < 90 ~ "Junior",
      Total_Credit_Hours_Earned <= 120 ~ "Senior",
      TRUE ~ "Extended" # > 120 credits
    ),
  )
}
```

## Course Name and Type Features

The raw `Course_Name` variable presented **_significant challenges_** due to its high cardinality and free-text nature. We implemented a **_detailed keyword-based categorization system_** (`add_course_name_category`) to group courses into meaningful academic domains (e.g., Business, Computer Science, Natural Sciences, Humanities) and types (e.g., Introductory, Intermediate, Advanced, Laboratory, Seminar, Independent Study). This involved using `grepl` with extensive keyword lists and a structured `case_when` statement to prioritize specific subject areas before applying level or type classifications. This provides a more manageable and informative feature than the raw course names.

```r
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
      |\bMath\b|Quantitative|Analytics|Equations|\bDesign\b(?=.*Sample)|\bDesign\b(?=.*Experimental)|Game Theory",
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
      grepl("Laboratory|\bLab\b", Course_Name, ignore.case = TRUE) ~ "Laboratory",
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
```

Similarly, the `Course_Type` variable (e.g., "MATH", "HIST", "ENG") required **_consolidation_**. We grouped the original prefixes into broader academic categories (e.g., "Business & Economics", "Natural Sciences", "Humanities & Arts", "Engineering & Technology") using `case_when`, explicitly handling `NA` and "No Response" values first.

```r
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
      TRUE ~ "Other" # Catch-all for any unmapped types
    )
  )
}
```

## Major Categories

The `Major` variable demanded a similar, extensive **_keyword-based reduction strategy_**. We developed a detailed `case_when` structure using `grepl` to map numerous specific major names into broader categories like "Business & Management", "Computer Science/IT", "Natural Sciences", "Health Sciences", "Social Sciences & History", "Arts & Humanities", etc. This process explicitly handled undecided/non-degree students and pre-professional tracks.

Crucially, this function also identifies students with **_multiple majors_** (`Has_Multiple_Majors`) by detecting commas in the `Major` field (after cleaning out minor designations) and identifies the presence of **_minors_** (`Has_Minor`) using "-MIN".

```r
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
    # Is_Pre_Professional = ifelse(grepl("Pre-Med|Pre-Vet|Pre-Law|Pre-Dent|Pre-Prof", Major, ignore.case = TRUE), TRUE, FALSE), # Optional flag if needed elsewhere
    
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
    select(-c(Major_Cleaned)) # Remove the temporary cleaned column
}
```

## Visit Pattern Features

`Student_ID` analysis enabled the construction of several **_usage metrics_**. Beyond simple visit counts (`Total_Visits`), we examined **_temporal patterns_** at multiple scales, calculating visits per semester (`Semester_Visits`) and average weekly visits (`Avg_Weekly_Visits`).

```r
add_visit_features <- function(df) {
  df %>%
    group_by(Student_IDs) %>%
    mutate(
      # Count visits per student
      Total_Visits = n(),
      # Count distinct visit dates per student per semester
      Semester_Visits = n_distinct(Check_In_Date), # Assumes data is per semester, group_by(Student_IDs, Semester) might be safer if not
      # Average visits per week (approximate)
      Avg_Weekly_Visits = Semester_Visits / max(Semester_Week, na.rm = TRUE) # Use max week in semester
    ) %>%
    ungroup()
}
```

Examination of visit frequency throughout the semester revealed **_clear patterns_**. Weeks 1-3, 9, 14, and 17 consistently showed lower activity levels (likely start/end of term, breaks), while the remaining weeks demonstrated higher traffic. We encoded this insight through a categorical `Week_Volume` feature.

```r
add_week_volume_category <- function(df) {
  df %>%
    mutate(
      Week_Volume = case_when(
        Semester_Week %in% c(4:8, 10:13, 15:16) ~ "High Volume",
        Semester_Week %in% c(1:3, 9, 14, 17) ~ "Low Volume",
        TRUE ~ "Other" # Handle potential NAs or unexpected week numbers
      )
    )
}
```

![Week Visits](../../presentation/images/eda/week_visits.png)

## Course Load and Performance Features

For each student-semester combination, we developed metrics to capture **_academic context_**. We tracked the number of unique courses (`Unique_Courses`), the diversity of course levels taken (`Course_Level_Mix`), and the proportion of upper-division courses (`Advanced_Course_Ratio`).

```r
add_course_load_features <- function(df) {
  df %>%
    group_by(Student_IDs, Semester) %>%
    mutate(
      # Number of unique courses taken by student in that semester
      Unique_Courses = n_distinct(Course_Number),
      # Mix of course levels taken by student in that semester
      Course_Level_Mix = n_distinct(Course_Code_by_Thousands),
      # Proportion of advanced courses ('Upper Classmen' level) taken by student in that semester
      Advanced_Course_Ratio = mean(Course_Level == "Upper Classmen", na.rm = TRUE)
    ) %>%
    ungroup()
}
```

Additionally, we implemented a **_GPA trend indicator_** (`GPA_Trend`) using the `sign()` function on the `Change_in_GPA` column. This focuses on the direction of GPA change (positive, negative, or zero) rather than the magnitude.

```r
add_gpa_trend <- function(df) {
  df %>% mutate(
    # Calculate GPA trend (1 for positive, -1 for negative, 0 for no change/NA)
    GPA_Trend = sign(Change_in_GPA),
  )
}
```

## Group Dynamics

A final analytical step involved identifying **_potential group study patterns_**. By counting occurrences of identical `Check_In_Timestamp` values (combined date and time), we estimated `Group_Size`. This led to a boolean `Group_Check_In` flag and a categorical `Group_Size_Category`.

```r
add_group_features <- function(df) {
  df %>%
    mutate(
      Check_In_Timestamp = ymd_hms(paste(Check_In_Date, Check_In_Time))
    ) %>%
    # Count how many rows share the exact same check-in timestamp
    add_count(Check_In_Timestamp, name = "Group_Size") %>%
    mutate(
      # Flag if group size is greater than 1
      Group_Check_In = Group_Size > 1,
      # Categorize group size
      Group_Size_Category = case_when(
        Group_Size == 1 ~ "Individual",
        Group_Size <= 3 ~ "Small Group",
        Group_Size <= 6 ~ "Medium Group",
        TRUE ~ "Large Group" # Includes > 6
      )
    ) %>%
    # Remove the temporary timestamp column
    select(-Check_In_Timestamp)
}
```

While some simultaneous check-ins might be coincidental, this classification captures potential **_social patterns_** in Learning Commons usage.

## Data Quality & Response Variable Handling

Essential validation and processing steps were included for the target variables (`Duration_In_Min` and `Occupancy`) and related features.

We ensured `Duration_In_Min` was calculated correctly from check-in/out times and that negative durations (data errors) were handled (set to `NA` and filtered).

```r
ensure_duration <- function(df) {
  # Calculate duration in minutes
  df %>%
    mutate(
      Duration_In_Min = as.numeric(difftime(
        Check_Out_Time,
        Check_In_Time,
        units = "mins"
      )),
      # Handle negative durations (likely data errors)
      Duration_In_Min = if_else(Duration_In_Min < 0, NA_real_, Duration_In_Min),
    ) %>%
    # Remove rows where duration could not be calculated or was negative
    filter(!is.na(Duration_In_Min))
}
```

We also created a categorical version of duration, `Session_Length_Category`.

```r
add_session_length_category <- function(df) {
  df %>% mutate(
    # Add session length categories based on Duration_In_Min
    Session_Length_Category = case_when(
      Duration_In_Min <= 30 ~ "Short",
      Duration_In_Min <= 90 ~ "Medium",
      Duration_In_Min <= 180 ~ "Long",
      Duration_In_Min > 180 ~ "Extended",
      TRUE ~ NA_character_ # Handle cases where Duration_In_Min might be NA
    )
  )
}
```

The `Occupancy` variable (number of students present at check-in time) was calculated by tracking cumulative arrivals and departures within each day.

```r
calculate_occupancy <- function(df) {
  df %>%
    # Ensure data is ordered chronologically within each day
    arrange(Check_In_Date, Check_In_Time) %>%
    group_by(Check_In_Date) %>%
    mutate(
      # Cumulative arrivals on this day up to this point
      Cum_Arrivals = row_number(),
      # Cumulative departures on this day before or at this check-in time
      Cum_Departures = sapply(seq_along(Check_In_Time), function(i) {
        sum(!is.na(Check_Out_Time[1:i]) & 
            Check_Out_Time[1:i] <= Check_In_Time[i])
      }),
      # Occupancy is arrivals minus departures
      Occupancy = Cum_Arrivals - Cum_Departures
    ) %>%
    # Remove temporary cumulative columns
    select(-c(Cum_Arrivals, Cum_Departures)) %>%
    ungroup() # Ungroup after calculation
}
```

## Conclusion

Our feature engineering process addressed key challenges in the Learning Commons dataset through **_systematic transformation_** and **_enrichment_**. Temporal features capture cyclical patterns and academic calendar effects. Course-related variable treatments reduce dimensionality while preserving meaningful distinctions. The dual approach to student classification provides complementary perspectives.

The **_detailed keyword-based categorization_** for `Course_Name`, `Course_Type`, and `Major` balances granularity and practicality, enhancing interpretability despite some consolidation. Visit pattern features capture individual and facility-wide trends. The group dynamics features offer insights into collaborative usage.

Extensive **_validation and calculation steps_** for `Duration_In_Min` and `Occupancy` ensure data quality. These engineered features form a robust foundation for modeling, though further refinement is always possible.