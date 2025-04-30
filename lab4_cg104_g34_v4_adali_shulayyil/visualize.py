import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("titanic/train.csv")


# Group by Ticket to get survival percentage and count
summary = df.groupby('Ticket').agg(
    Survival_Percentage=('Survived', 'mean'),
    Passenger_Count=('Survived', 'count')
).reset_index()

summary['Survival_Percentage'] *= 100  # Convert to percentage

# Get top 5 and bottom 5 by survival percentage
top_5 = summary.sort_values(by='Survival_Percentage', ascending=False)


# Combine for plotting
plot_data = top_5

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=plot_data, x='Ticket', y='Survival_Percentage', palette='viridis')

# Add count labels on top of bars
for i, row in plot_data.iterrows():
    ax.text(
        i - plot_data.index.min(),  # x position
        row['Survival_Percentage'] + 2,  # y position just above the bar
        f"n={row['Passenger_Count']}", 
        color='black', ha='center'
    )

plt.title('Top and Bottom 5 Survival % by Ticket (with Counts)')
plt.ylabel('Survival Percentage')
plt.xlabel('Ticket')
plt.ylim(0, 110)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()