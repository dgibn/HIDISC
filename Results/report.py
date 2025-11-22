import pandas as pd

# Load the Excel file
file_path = "HIDISC/Results/Office_Home/HIDISC_Results.xlsx"  # Replace with the actual file path
df = pd.read_excel(file_path)

# Find the best "All" accuracy for each (Source Domain, Target Domain) combination
best_entries = df.loc[df.groupby(["Source Domain", "Target Domain"])["All"].idxmax()]

# Calculate the average of the best "All," "Old," and "New" values
average_all = best_entries["All"].mean()
average_old = best_entries["Old"].mean()
average_new = best_entries["New"].mean()

# Print results
print("Selected Entries with Best 'All' Accuracy for Each (Source Domain, Target Domain) Combination:")
print(best_entries)
print("\nAverages:")
print(f"Average All: {average_all:.2f}")
print(f"Average Old: {average_old:.2f}")
print(f"Average New: {average_new:.2f}")

# Save the selected entries to a new Excel file
best_entries.to_excel("HIDISC/Results/Office_Home/best_selected_entries.xlsx", index=False)
print("\nSelected entries saved as 'best_selected_entries.xlsx'.")
