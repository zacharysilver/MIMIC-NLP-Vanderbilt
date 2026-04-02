import pandas as pd

print("Environment working!")

# Load admissions from /hosp/admissions CSV file
admissions = pd.read_csv("data/admissions.csv", parse_dates=["admittime", "dischtime"])

admissions = admissions.sort_values(by=["subject_id", "admittime"]).reset_index(drop=True)

# Create readmission records
records = []

for subject_id, group in admissions.groupby("subject_id"):
    group = group.sort_values("admittime").reset_index(drop=True)
    
    for i in range(len(group) - 1):
        first = group.loc[i]
        second = group.loc[i + 1]
        
        # Calculate gap between visits
        interval_days = (second["admittime"] - first["dischtime"]).days

        # Skip invalid or overlapping records
        if pd.isnull(interval_days) or interval_days < 0:
            continue
        
        label = "yes" if interval_days <= 30 else "no"

        # Calculate length of each stay
        first_duration = (first["dischtime"] - first["admittime"]).days
        second_duration = (second["dischtime"] - second["admittime"]).days

        records.append({
            "subject_id": subject_id,
            "30_day_readmission": label,
            "first_visit": f"{first['admittime']} - {first['dischtime']}",
            "first_visit_length_days": first_duration,
            "second_visit": f"{second['admittime']} - {second['dischtime']}",
            "second_visit_length_days": second_duration,
            "interval_days": interval_days,
            "hadm_id": first["hadm_id"]           
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# Split into two groups
group_le_30 = df[df["30_day_readmission"] == "yes"]
group_gt_30 = df[df["30_day_readmission"] == "no"].head(len(group_le_30))


group_le_30.to_csv("data/readmission_less_30_result.csv", index=False)
group_gt_30.to_csv("data/readmission_greater_30_result.csv", index=False)

# Combine the two groups
result = pd.concat([group_le_30, group_gt_30])

# Export to CSV
result.to_csv("data/readmission.csv", index=False)
