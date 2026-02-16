import pandas as pd
from scipy.io import arff
import os

def wrangle_student_data(input_file):
    # 1. Load the ARFF file
    print(f"--- Loading {input_file} ---")
    data, meta = arff.loadarff(input_file)
    df = pd.DataFrame(data)

    # 2. Decode byte strings to regular strings
    # ARFF files often load as b'string'. This converts them to 'string'.
    df = df.apply(lambda x: x.str.decode('utf8') if x.dtype == "object" else x)

    # 3. Handle Ordinal Data (Ranked Categories)
    # We map these to numbers so the model understands Excellent > Vg > Good
    grade_map = {'Excellent': 4, 'Vg': 3, 'Good': 2, 'Average': 1}
    ordinal_features = ['Performance', 'Class_ X_Percentage', 'Class_XII_Percentage']
    
    for col in ordinal_features:
        df[col] = df[col].map(grade_map)

    # 4. Handle Nominal Data (Non-ranked Categories)
    # Using get_dummies creates binary columns (e.g., Gender_male, Gender_female)
    nominal_features = [
        'Gender', 'Caste', 'coaching', 'time', 
        'Class_ten_education', 'twelve_education', 
        'medium', 'Father_occupation', 'Mother_occupation'
    ]
    
    # drop_first=True prevents the "Dummy Variable Trap" (multicollinearity)
    df_final = pd.get_dummies(df, columns=nominal_features, drop_first=True)

    # 5. Final Check for Missing Values
    null_count = df_final.isnull().sum().sum()
    print(f"Missing values found: {null_count}")

    return df_final

if __name__ == "__main__":
    # Path based on your VS Code screenshot
    file_path = '../Student Performance/CEE_DATA.arff'
    
    if os.path.exists(file_path):
        cleaned_df = wrangle_student_data(file_path)
        
        # Save to CSV so your models can load it easily
        output_name = '../Student Performance/cleaned_student_data.csv'
        cleaned_df.to_csv(output_name, index=False)
        
        print(f"--- Success! Wrangled data saved to: {output_name} ---")
        print(f"New shape of data: {cleaned_df.shape}")
    else:
        print(f"Error: Could not find file at {file_path}")