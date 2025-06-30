import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

# Define tasks with start dates and durations
tasks = [
    {"Task": "1a. Image Classification", "Start": "2025-05-01", "Duration": 7},
    {"Task": "1b. Preprocessor (GAN + ROI)", "Start": "2025-05-01", "Duration": 7},
    {"Task": "2. Text Detection", "Start": "2025-05-08", "Duration": 14},
    {"Task": "3. Text Recognition", "Start": "2025-05-22", "Duration": 14},
    {"Task": "4. Layout Analysis", "Start": "2025-06-05", "Duration": 14},
    {"Task": "5. OCR Pipeline Integration", "Start": "2025-06-19", "Duration": 7},
    {"Task": "6. Evaluation & Tuning", "Start": "2025-06-26", "Duration": 7},
    {"Task": "7. Deployment", "Start": "2025-07-03", "Duration": 7},
    {"Task": "8. Finalization & CI/CD", "Start": "2025-07-10", "Duration": 7}
]

# Convert to DataFrame
df = pd.DataFrame(tasks)
df['Start'] = pd.to_datetime(df['Start'])

# Use a distinct color for each task using matplotlib's colormap
cmap = plt.cm.get_cmap('tab20', len(df))  # 'tab20' supports up to 20 distinct colors
df['Color'] = [cmap(i) for i in range(len(df))]

# Output path
output_path = "./dev/assets/doctane_gantt_chart.png"

# Plotting
fig, ax = plt.subplots(figsize=(18, 6))

for i, row in df.iterrows():
    ax.barh(y=row["Task"], width=row["Duration"], left=row["Start"], 
            color=row["Color"], edgecolor='black')
    label_x = row["Start"] + timedelta(days=row["Duration"] / 2)
    ax.text(label_x, i, row["Task"], va='center', ha='center', color='white',
            fontweight='bold', fontsize=5)

# X-axis formatting
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)

# Labels and title
plt.xlabel("Timeline", fontsize=12)
plt.ylabel("Tasks", fontsize=12)
plt.title("Doctane OCR Pipeline Development – Gantt Chart", fontsize=14, weight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and display
plt.savefig(output_path, dpi=300)
print(f"Gantt chart saved to: {output_path}")
plt.show()
