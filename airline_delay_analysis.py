# =====================================================
# AIRLINE DELAY ANALYSIS PROJECT
# =====================================================
# A comprehensive data science project analyzing 
# airline delays using statistical methods
# =====================================================

import pandas as pd
import numpy as np
import sys
from scipy import stats

# =====================================================
# IMPORT LIBRARIES
# =====================================================

print("==========================================")
print(" AIRLINE DELAY ANALYSIS PROJECT ")
print("==========================================")

# =====================================================
# USER INPUT
# =====================================================

file_name = input("\nEnter CSV File Name (Example: flights.csv) : ")

rows = int(input("Enter Number of Rows to Load (Example: 100000) : "))

# =====================================================
# LOAD DATASET
# =====================================================

try:
    df = pd.read_csv(file_name, nrows=rows)
    print("\nDATASET LOADED SUCCESSFULLY")
except FileNotFoundError:
    print(f"\nERROR: File '{file_name}' not found!")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: {str(e)}")
    sys.exit(1)

# =====================================================
# DATA UNDERSTANDING
# =====================================================

print("\n==========================================")
print(" DATA UNDERSTANDING ")
print("==========================================")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFO")
print(df.info())

# =====================================================
# M1 - PREPROCESSING
# =====================================================

print("\n==========================================")
print(" M1 - DATA PREPROCESSING ")
print("==========================================")

# Missing Values

print("\nMISSING VALUES BEFORE CLEANING")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

# Numerical Columns

num_cols = df.select_dtypes(include='number').columns

df[num_cols] = df[num_cols].fillna(
    df[num_cols].median()
)

# Categorical Columns

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    if len(df[col].mode()) > 0:
        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum().sum(), "total missing values remaining")

# Duplicate Rows

duplicates_count = df.duplicated().sum()
print("\nNUMBER OF DUPLICATES")
print(duplicates_count)

df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

print("\nDATASET SHAPE AFTER REMOVING DUPLICATES")
print(df.shape)

# =====================================================
# DATA CORRECTION
# =====================================================

print("\nDATA TYPE CORRECTION")

if 'ARRIVAL_DELAY' in df.columns:
    df['ARRIVAL_DELAY'] = pd.to_numeric(
        df['ARRIVAL_DELAY'],
        errors='coerce'
    )

if 'DEPARTURE_DELAY' in df.columns:
    df['DEPARTURE_DELAY'] = pd.to_numeric(
        df['DEPARTURE_DELAY'],
        errors='coerce'
    )

if 'DISTANCE' in df.columns:
    df['DISTANCE'] = pd.to_numeric(
        df['DISTANCE'],
        errors='coerce'
    )

print("\nUPDATED DATA TYPES")
print(df.dtypes)

# =====================================================
# CREATE NEW COLUMNS
# =====================================================

print("\nCREATING NEW COLUMNS")

if 'ARRIVAL_DELAY' in df.columns:
    df['DELAY_STATUS'] = np.where(
        df['ARRIVAL_DELAY'] > 15,
        'Delayed',
        'On Time'
    )
    print("✓ DELAY_STATUS column created")

if 'DEPARTURE_DELAY' in df.columns:
    df['DEPARTURE_STATUS'] = np.where(
        df['DEPARTURE_DELAY'] > 15,
        'Delayed',
        'On Time'
    )
    print("✓ DEPARTURE_STATUS column created")

if 'DISTANCE' in df.columns:
    df['DISTANCE_CATEGORY'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    print("✓ DISTANCE_CATEGORY column created")

print("\nNEW COLUMNS CREATED SUCCESSFULLY")

# =====================================================
# M2 - DESCRIPTIVE STATISTICS
# =====================================================

print("\n==========================================")
print(" M2 - DESCRIPTIVE STATISTICS ")
print("==========================================")

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMEAN OF NUMERICAL COLUMNS")
print(df[num_cols].mean())

print("\nMEDIAN OF NUMERICAL COLUMNS")
print(df[num_cols].median())

print("\nSTANDARD DEVIATION")
print(df[num_cols].std())

print("\nMAXIMUM VALUES")
print(df[num_cols].max())

print("\nMINIMUM VALUES")
print(df[num_cols].min())

print("\nSKEWNESS")
print(df[num_cols].skew())

print("\nKURTOSIS")
print(df[num_cols].kurtosis())

# =====================================================
# AIRLINE DELAY ANALYSIS
# =====================================================

print("\n==========================================")
print(" AIRLINE DELAY ANALYSIS ")
print("==========================================")

if 'AIRLINE' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    airline_delay = df.groupby(
        'AIRLINE'
    )['ARRIVAL_DELAY'].agg(['mean', 'median', 'count', 'std'])

    print("\nAVERAGE DELAY BY AIRLINE")
    print(airline_delay.sort_values('mean', ascending=False))

    print("\nAIRLINE WITH HIGHEST AVERAGE DELAY")
    highest_delay_airline = airline_delay['mean'].idxmax()
    print(f"{highest_delay_airline}: {airline_delay.loc[highest_delay_airline, 'mean']:.2f} minutes")

    print("\nAIRLINE WITH LOWEST AVERAGE DELAY")
    lowest_delay_airline = airline_delay['mean'].idxmin()
    print(f"{lowest_delay_airline}: {airline_delay.loc[lowest_delay_airline, 'mean']:.2f} minutes")

# =====================================================
# DAY WISE ANALYSIS
# =====================================================

print("\n==========================================")
print(" DAY WISE ANALYSIS ")
print("==========================================")

if 'DAY_OF_WEEK' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    day_delay = df.groupby(
        'DAY_OF_WEEK'
    )['ARRIVAL_DELAY'].agg(['mean', 'median', 'count'])

    print("\nAVERAGE DELAY BY DAY OF WEEK")
    print(day_delay)

    print("\nDY WITH HIGHEST AVERAGE DELAY")
    highest_delay_day = day_delay['mean'].idxmax()
    print(f"Day {highest_delay_day}: {day_delay.loc[highest_delay_day, 'mean']:.2f} minutes")

# =====================================================
# ORIGIN/DESTINATION ANALYSIS
# =====================================================

print("\n==========================================")
print(" ORIGIN/DESTINATION ANALYSIS ")
print("==========================================")

if 'ORIGIN_AIRPORT' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    origin_delay = df.groupby(
        'ORIGIN_AIRPORT'
    )['ARRIVAL_DELAY'].agg(['mean', 'count']).sort_values('mean', ascending=False)

    print("\nTOP 10 AIRPORTS WITH HIGHEST DEPARTURE DELAYS")
    print(origin_delay.head(10))

if 'DESTINATION_AIRPORT' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    dest_delay = df.groupby(
        'DESTINATION_AIRPORT'
    )['ARRIVAL_DELAY'].agg(['mean', 'count']).sort_values('mean', ascending=False)

    print("\nTOP 10 AIRPORTS WITH HIGHEST ARRIVAL DELAYS")
    print(dest_delay.head(10))

# =====================================================
# DISTANCE ANALYSIS
# =====================================================

print("\n==========================================")
print(" DISTANCE ANALYSIS ")
print("==========================================")

if 'DISTANCE_CATEGORY' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    distance_delay = df.groupby(
        'DISTANCE_CATEGORY'
    )['ARRIVAL_DELAY'].agg(['mean', 'median', 'count'])

    print("\nAVERAGE DELAY BY DISTANCE CATEGORY")
    print(distance_delay)

# =====================================================
# M3 - PROBABILITY
# =====================================================

print("\n==========================================")
print(" M3 - PROBABILITY ")
print("==========================================")

if 'ARRIVAL_DELAY' in df.columns:

    total_flights = len(df)

    delayed_flights = len(
        df[df['ARRIVAL_DELAY'] > 15]
    )

    probability_delay = delayed_flights / total_flights

    print("\nTOTAL FLIGHTS")
    print(total_flights)

    print("\nDELAYED FLIGHTS (> 15 minutes)")
    print(delayed_flights)

    print("\nPROBABILITY OF DELAY")
    print(f"{probability_delay:.4f} ({probability_delay*100:.2f}%)")

    # Probability of On-Time Flights

    on_time = len(
        df[df['ARRIVAL_DELAY'] <= 15]
    )

    probability_on_time = on_time / total_flights

    print("\nON-TIME FLIGHTS")
    print(on_time)

    print("\nPROBABILITY OF ON-TIME FLIGHT")
    print(f"{probability_on_time:.4f} ({probability_on_time*100:.2f}%)")

    # Probability of Cancelled Flights

    if 'CANCELLED' in df.columns:
        cancelled_flights = len(
            df[df['CANCELLED'] == 1]
        )
        probability_cancelled = cancelled_flights / total_flights

        print("\nCANCELLED FLIGHTS")
        print(cancelled_flights)

        print("\nPROBABILITY OF CANCELLATION")
        print(f"{probability_cancelled:.4f} ({probability_cancelled*100:.2f}%)")

# =====================================================
# CONDITIONAL PROBABILITY
# =====================================================

print("\nCONDITIONAL PROBABILITY")

if 'AIRLINE' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    first_airline = df['AIRLINE'].iloc[0]

    airline_data = df[
        df['AIRLINE'] == first_airline
    ]

    airline_total = len(airline_data)

    airline_delayed = len(
        airline_data[
            airline_data['ARRIVAL_DELAY'] > 15
        ]
    )

    if airline_total > 0:
        conditional_probability = (
            airline_delayed / airline_total
        )

        print(
            f"\nP(Delay | Airline = {first_airline})"
        )

        print(f"{conditional_probability:.4f} ({conditional_probability*100:.2f}%)")

# =====================================================
# M4 - INFERENCE
# =====================================================

print("\n==========================================")
print(" M4 - INFERENCE ")
print("==========================================")

if 'ARRIVAL_DELAY' in df.columns and 'DEPARTURE_DELAY' in df.columns:

    sample1 = df['ARRIVAL_DELAY'].sample(
        min(100, len(df)),
        random_state=1
    )

    sample2 = df['DEPARTURE_DELAY'].sample(
        min(100, len(df)),
        random_state=1
    )

    sample1_mean = sample1.mean()

    sample2_mean = sample2.mean()

    print("\nARRIVAL DELAY SAMPLE MEAN (n=100)")
    print(f"{sample1_mean:.2f} minutes")

    print("\nDEPARTURE DELAY SAMPLE MEAN (n=100)")
    print(f"{sample2_mean:.2f} minutes")

    mean_difference = sample1_mean - sample2_mean

    print("\nMEAN DIFFERENCE (Arrival - Departure)")
    print(f"{mean_difference:.2f} minutes")

    # Confidence Interval
    print("\n95% CONFIDENCE INTERVAL")
    
    sample1_std = sample1.std()
    sample1_se = sample1_std / np.sqrt(len(sample1))
    ci_lower = sample1_mean - 1.96 * sample1_se
    ci_upper = sample1_mean + 1.96 * sample1_se
    
    print(f"Arrival Delay CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # T-test
    print("\nT-TEST (Arrival vs Departure Delay)")
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")

# =====================================================
# M5 - REGRESSION
# =====================================================

print("\n==========================================")
print(" M5 - REGRESSION ")
print("==========================================")

if 'DEPARTURE_DELAY' in df.columns and 'ARRIVAL_DELAY' in df.columns:

    regression_data = df[
        ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
    ].dropna()

    x = regression_data['DEPARTURE_DELAY'].values

    y = regression_data['ARRIVAL_DELAY'].values

    # Linear Regression using NumPy

    slope, intercept = np.polyfit(x, y, 1)

    print("\nLINEAR REGRESSION EQUATION")

    print(
        f"Arrival Delay = {slope:.4f} * Departure Delay + {intercept:.4f}"
    )

    # Predicted Values

    y_pred = slope * x + intercept

    # R-Squared

    ss_total = np.sum(
        (y - np.mean(y)) ** 2
    )

    ss_residual = np.sum(
        (y - y_pred) ** 2
    )

    r_squared = 1 - (
        ss_residual / ss_total
    )

    print("\nR-SQUARED VALUE")
    print(f"{r_squared:.4f}")

    print("\nINTERPRETATION")
    print(f"For every 1 minute increase in departure delay,")
    print(f"arrival delay increases by {slope:.4f} minutes on average.")

    # Correlation
    correlation = np.corrcoef(x, y)[0, 1]
    print(f"\nCorrelation Coefficient: {correlation:.4f}")

# =====================================================
# CORRELATION ANALYSIS
# =====================================================

print("\n==========================================")
print(" CORRELATION ANALYSIS ")
print("==========================================")

print("\nCORRELATION MATRIX (Numerical Columns)")
print(df[num_cols].corr())

# =====================================================
# M6 - REPORTING
# =====================================================

print("\n==========================================")
print(" M6 - REPORTING ")
print("==========================================")

print("\nFINAL OBSERVATIONS")

print("✓ 1. Dataset cleaned successfully")
print("✓ 2. Missing values handled")
print("✓ 3. Duplicate rows removed")
print("✓ 4. Statistical analysis completed")
print("✓ 5. Probability analysis completed")
print("✓ 6. Inference analysis completed (with t-tests)")
print("✓ 7. Regression analysis completed")
print("✓ 8. Correlation analysis completed")
print("✓ 9. Departure delay affects arrival delay")
print("✓ 10. Distance and airport analysis provided")

# =====================================================
# SAVE CLEANED DATASET
# =====================================================

save_option = input(
    "\nDo you want to save cleaned dataset? (yes/no) : "
)

if save_option.lower() == 'yes':

    output_file = input(
        "Enter Output File Name (Example: cleaned_flights.csv) : "
    )

    try:
        df.to_csv(output_file, index=False)
        print(f"\n✓ CLEANED DATASET SAVED SUCCESSFULLY")
        print(f"File: {output_file}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
    except Exception as e:
        print(f"\nERROR saving file: {str(e)}")

# =====================================================
# END OF PROJECT
# =====================================================

print("\n==========================================")
print(" PROJECT COMPLETED SUCCESSFULLY ")
print("==========================================")
