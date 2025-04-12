# Load required libraries
import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# READ DATA
# -----------------------------------------------------------------------------
# Using Path for cross-platform compatibility
data_path = Path("data/LC_train.csv")
data_raw = pd.read_csv(data_path)

# -----------------------------------------------------------------------------
# TEMPORAL FEATURES
# -----------------------------------------------------------------------------
def prepare_dates(df):
    """Convert date and time columns to appropriate formats."""
    return df.assign(
        Check_In_Date=pd.to_datetime(df['Check_In_Date'], format='%m/%d/%y'),
        Check_In_Time=pd.to_datetime(df['Check_In_Time'], format='%H:%M:%S').dt.time,
        Check_Out_Time=pd.to_datetime(df['Check_Out_Time'], format='%H:%M:%S').dt.time,
    )

def add_temporal_features(df):
    """Add time-based features to the dataframe."""
    return df.assign(
        Day_of_Week=pd.to_datetime(df['Check_In_Date']).dt.day_name().str[:3],
        Is_Weekend=pd.to_datetime(df['Check_In_Date']).dt.dayofweek.isin([5, 6]),
        Week_of_Month=pd.to_datetime(df['Check_In_Date']).dt.day.apply(lambda x: np.ceil(x/7)),
        Month=pd.to_datetime(df['Check_In_Date']).dt.month_name().str[:3],
        Hour_of_Day=pd.to_datetime(df['Check_In_Time'], format='%H:%M:%S').dt.hour,
    )

def add_time_period(df):
    """Categorize hours into time periods."""
    return df.assign(
        Time_Period=pd.cut(
            df['Hour_of_Day'],
            bins=[-np.inf, 6, 12, 17, 22, np.inf],
            labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Late Night'],
            ordered=False
        )
    )

# -----------------------------------------------------------------------------
# ACADEMIC & COURSE FEATURES
# -----------------------------------------------------------------------------
def add_course_features(df):
    """Add course-related features."""
    df = df.copy()
    df['Course_Level'] = df['Course_Code_by_Thousands'].map({
        '1000': 'Introductory', 
        '2000': 'Intermediate',
    }).fillna('Advanced')
    df.loc[~df['Course_Level'].isin(['Introductory', 'Intermediate']), 'Course_Level'] = 'Other'
    return df

def add_performance_indicators(df):
    """Add student performance-related features."""
    return df.assign(
        GPA_Category=pd.cut(
            df['Cumulative_GPA'],
            bins=[-np.inf, 2.0, 3.0, 3.5, np.inf],
            labels=['Needs Improvement', 'Satisfactory', 'Good', 'Excellent']
        )
    )

def add_credit_load_features(df):
    """Add credit load-related features."""
    return df.assign(
        Credit_Load_Category=pd.cut(
            df['Term_Credit_Hours'],
            bins=[-np.inf, 6, 12, 18, np.inf],
            labels=['Part Time', 'Half Time', 'Full Time', 'Overload']
        )
    )

def add_standing_features(df):
    """Add class standing-related features."""
    standing_map = {
        'Freshman': 'First Year',
        'Sophomore': 'Second Year',
        'Junior': 'Third Year',
        'Senior': 'Fourth Year'
    }
    
    return df.assign(
        Class_Standing_Self_Reported=df['Class_Standing'].map(standing_map).fillna(df['Class_Standing']),
        Class_Standing_BGSU=pd.cut(
            df['Total_Credit_Hours_Earned'],
            bins=[-np.inf, 30, 60, 90, 120, np.inf],
            labels=['Freshman', 'Sophomore', 'Junior', 'Senior', 'Extended']
        )
    )

def add_course_load_features(df):
    """Add features related to course load and complexity."""
    return df.assign(
        # Number of different courses being taken
        Unique_Courses=df.groupby(['Student_IDs', 'Semester'])['Course_Number'].transform('nunique'),
        # Mix of course levels
        Course_Level_Mix=df.groupby(['Student_IDs', 'Semester'])['Course_Code_by_Thousands'].transform('nunique'),
        # Proportion of high-level courses
        Advanced_Course_Ratio=df.groupby(['Student_IDs', 'Semester'])['Course_Level'].transform(
            lambda x: (x == 'Advanced').mean()
        )
    )

def add_academic_features(df):
    """Add features related to academic progress and performance trends."""
    
    def convert_semester_to_date(semester_str):
        """Convert semester strings to approximate dates."""
        if pd.isna(semester_str):
            return pd.NaT
        
        # Handle format like "Fall Semester 2017"
        parts = semester_str.split()
        if len(parts) < 3:  # Need at least season and year
            return pd.NaT
            
        season, year = parts[0].lower(), parts[2]  # Take first and last parts
        
        # Map seasons to months
        season_month = {
            'fall': '09',
            'spring': '01',
            'summer': '06',
            'winter': '12'
        }
        
        month = season_month.get(season, '01')
        try:
            return pd.to_datetime(f"{year}-{month}-01")
        except:
            return pd.NaT

    # Convert graduation dates to datetime
    graduation_dates = df['Expected_Graduation'].apply(convert_semester_to_date)
    checkin_dates = pd.to_datetime(df['Check_In_Date'])

    return df.assign(
        # Percentage of degree completion (based on typical 120 credit hours)
        Degree_Progress=(df['Total_Credit_Hours_Earned'] / 120 * 100).clip(upper=100),
        # GPA trend (positive or negative)
        GPA_Trend=np.sign(df['Change_in_GPA']),
        # Distance from graduation (in months)
        Semesters_To_Graduation=(
            graduation_dates.dt.year * 12 +
            graduation_dates.dt.month -
            checkin_dates.dt.year * 12 -
            checkin_dates.dt.month
        )
    )

# -----------------------------------------------------------------------------
# STUDY SESSION & BEHAVIORAL FEATURES
# -----------------------------------------------------------------------------
def add_session_features(df):
    """Add study session-related features."""
    duration = ((pd.to_datetime(df['Check_Out_Time'].astype(str), format='%H:%M:%S') - 
                pd.to_datetime(df['Check_In_Time'].astype(str), format='%H:%M:%S'))
               .dt.total_seconds() / 60)
    
    # Filter out negative durations
    df = df[duration >= 0].copy()
    duration = duration[duration >= 0]
    
    return df.assign(
        Duration_In_Min=duration,
        Session_Length_Category=pd.cut(
            duration,
            bins=[-np.inf, 30, 90, 180, np.inf],
            labels=['Short', 'Medium', 'Long', 'Extended']
        )
    )

def calculate_occupancy(group):
    """Calculate occupancy for a group of check-ins."""
    # Specify format for datetime conversion
    check_in_times = pd.to_datetime(group['Check_In_Time'].astype(str), format='%H:%M:%S')
    check_out_times = pd.to_datetime(group['Check_Out_Time'].astype(str), format='%H:%M:%S')
    
    arrivals = range(1, len(group) + 1)
    departures = [
        sum(1 for j in range(i + 1)
            if not pd.isna(check_out_times.iloc[j]) 
            and check_out_times.iloc[j] <= check_in_times.iloc[i])
        for i in range(len(group))
    ]
    
    return [a - d for a, d in zip(arrivals, departures)]

def add_occupancy(df):
    """Add occupancy calculations to the dataframe."""
    df = df.sort_values(['Check_In_Date', 'Check_In_Time'])
    return df.assign(
        Occupancy=df.groupby('Check_In_Date', group_keys=False)
                    .apply(calculate_occupancy, include_groups=False)
                    .explode()
                    .values
    )

def add_previous_session_features(df):
    """Add features related to student's previous study patterns."""
    # Sort by student and datetime
    df = df.sort_values(['Student_IDs', 'Check_In_Date', 'Check_In_Time'])
    
    return df.assign(
        # Previous session duration for same student
        Previous_Duration=df.groupby('Student_IDs')['Duration_In_Min'].shift(1),
        # Average duration of student's past sessions
        Avg_Past_Duration=df.groupby('Student_IDs')['Duration_In_Min'].transform(
            lambda x: x.expanding().mean().shift(1)
        ),
        # Days since last visit
        Days_Since_Last_Visit=df.groupby('Student_IDs')['Check_In_Date'].diff().dt.days
    )

def add_behavioral_features(df):
    """Add features related to study patterns and behaviors."""
    return df.assign(
        # Consistency of visit time
        Visit_Time_Variance=df.groupby('Student_IDs')['Hour_of_Day'].transform('std'),
        # Preferred study time period
        Most_Common_Time_Period=df.groupby('Student_IDs')['Time_Period'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        ),
        # Study regularity (visits per week)
        Weekly_Visit_Frequency=df.groupby(['Student_IDs', 'Semester_Week'])['Check_In_Date'].transform('count')
    )

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------

def engineer_features(df):
    """Apply all feature engineering transformations."""
    return (df
            .pipe(prepare_dates)
            .pipe(add_temporal_features)
            .pipe(add_time_period)
            .pipe(add_course_features)
            .pipe(add_performance_indicators)
            .pipe(add_session_features)
            .pipe(add_credit_load_features)
            .pipe(add_standing_features)
            .pipe(add_occupancy)
            .pipe(add_previous_session_features)
            .pipe(add_course_load_features)
            .pipe(add_academic_features)
            .pipe(add_behavioral_features)
    )

# Process the data
lc_engineered = engineer_features(data_raw)

# -----------------------------------------------------------------------------
# SAVE ENGINEERED DATA
# -----------------------------------------------------------------------------
output_path = Path("data/LC_engineered_py.csv")
lc_engineered.to_csv(output_path, index=False)
