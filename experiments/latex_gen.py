import pandas as pd

# Load the CSV
df = pd.read_csv("/home/cc/task4feedback_torchrl/experiments/saved_models/sorted_pivoted_summary.csv")


# Function to convert cell like "1.04(1.06)" → stacked LaTeX
def format_cell(value):
    if isinstance(value, str) and "(" in value and ")" in value:
        main, secondary = value.split("(")
        secondary = secondary.strip(")")
        return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{main.strip()}\\\\ ({secondary.strip()})\\end{{tabular}}"
    return value


# Apply formatter to all numeric cells
for col in df.columns[3:]:
    df[col] = df[col].apply(format_cell)

# Start building LaTeX table
latex = []
latex.append("\\begin{table*}[htbp]")
latex.append("\\centering")
latex.append("\\renewcommand{\\arraystretch}{1.2}")
latex.append("\\setlength{\\tabcolsep}{3pt}")
latex.append("\\begin{tabular}{||c||c||c||c c c c c||c c c c c||c c c c c||c c c c c||}")
latex.append("\\hline\\hline")
latex.append(
    "\\multirow{2}{*}{Workload} & \\multirow{2}{*}{Interior} & \\multirow{2}{*}{Boundary} "
    "& \\multicolumn{5}{c||}{A} & \\multicolumn{5}{c||}{C} & "
    "\\multicolumn{5}{c||}{D} & \\multicolumn{5}{c||}{G} \\\\"
)
latex.append("\\cline{4-23}")
latex.append("& & & Worst & Q1 & Q2 & Q3 & Best & Worst & Q1 & Q2 & Q3 & Best " "& Worst & Q1 & Q2 & Q3 & Best & Worst & Q1 & Q2 & Q3 & Best \\\\")
latex.append("\\hline\\hline")

# Fill rows
for _, row in df.iterrows():
    line = f"{row['traj_type']} & {row['Interior']} & {row['Boundary']} "
    # A policy
    line += " & " + " & ".join([row["Apolicy_worst"], row["Apolicy_q1"], row["Apolicy_q2"], row["Apolicy_q3"], row["Apolicy_best"]])
    # C policy
    line += " & " + " & ".join([row["Cpolicy_worst"], row["Cpolicy_q1"], row["Cpolicy_q2"], row["Cpolicy_q3"], row["Cpolicy_best"]])
    # D policy
    line += " & " + " & ".join([row["Dpolicy_worst"], row["Dpolicy_q1"], row["Dpolicy_q2"], row["Dpolicy_q3"], row["Dpolicy_best"]])
    # G policy
    line += " & " + " & ".join([row["Gpolicy_worst"], row["Gpolicy_q1"], row["Gpolicy_q2"], row["Gpolicy_q3"], row["Gpolicy_best"]])
    line += " \\\\ \\hline"
    latex.append(line)

latex.append("\\hline\\hline")
latex.append("\\end{tabular}")
latex.append("\\end{table*}")

# Save to .tex file
with open("output_table.tex", "w") as f:
    f.write("\n".join(latex))

print("LaTeX table written to output_table.tex")
