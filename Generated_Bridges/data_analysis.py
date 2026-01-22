import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create output directory for plots
output_dir = Path("analysis_plots")
output_dir.mkdir(exist_ok=True)

# Load the data (from parent directory)
current_dir = Path(__file__).parent
json_path = current_dir / "bridge_summary.json"
df = pd.read_json(json_path)

print(f"Loaded {len(df)} bridge records")
print(f"\nDataset Overview:")
print(df.describe())

# ========== Graph 1: Bridge Type Distribution ==========
plt.figure(figsize=(10, 7))
bridge_types = df['bridge_type'].unique()
palette = sns.color_palette("Set1", len(bridge_types))

sns.scatterplot(
    data=df,
    x='span_m', 
    y='depth_of_girder', 
    hue='bridge_type', 
    palette=palette,
    s=80,
    edgecolor='black',
    alpha=0.85
)

plt.title('Span Length vs Girder Depth by Bridge Type', fontsize=14, fontweight='bold')
plt.xlabel('Span Length (m)', fontsize=12)
plt.ylabel('Girder Depth (m)', fontsize=12)
plt.legend(title='Bridge Type')
plt.tight_layout()
plt.savefig(output_dir / '01_span_vs_girder_depth_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_span_vs_girder_depth_scatter.png")
plt.close()


# ========== Graph 3: Total Length Distribution by Bridge Type ==========
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='bridge_type', y='total_length_m', palette='Set2')
plt.title('Bridge Length Distribution by Type', fontsize=14, fontweight='bold')
plt.xlabel('Bridge Type', fontsize=12)
plt.ylabel('Total Length (m)', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / '03_length_by_bridge_type.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_length_by_bridge_type.png")
plt.close()


# ========== Graph 5: Total Piers Distribution ==========
plt.figure(figsize=(10, 6))
df['total_piers'].hist(bins=20, color='#9b59b6', edgecolor='black')
plt.title('Distribution of Total Number of Piers', fontsize=14, fontweight='bold')
plt.xlabel('Total Piers', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / '05_total_piers_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_total_piers_distribution.png")
plt.close()

# ========== Graph 6: Average Total Piers by Pier Type ==========
plt.figure(figsize=(8, 6))
avg_piers = df.groupby('pier_type')['total_piers'].mean()
avg_piers.plot(kind='bar', color=['#1abc9c', '#e67e22'])
plt.title('Average Number of Piers by Pier Type', fontsize=14, fontweight='bold')
plt.xlabel('Pier Type', fontsize=12)
plt.ylabel('Average Total Piers', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '06_avg_piers_by_type.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_avg_piers_by_type.png")
plt.close()

# ========== Graph 7: Correlation Heatmap ==========
plt.figure(figsize=(10, 8))
numerical_cols = ['total_length_m', 'width_m', 'depth_of_girder', 'span_m', 
                  'num_spans', 'lanes', 'total_piers', 'number_of_piers_along_length',
                  'number_of_piers_across_width']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Correlation Matrix of Bridge Dimensions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_correlation_heatmap.png")
plt.close()

# ========== Graph 8: Span vs Depth Relationship ==========
plt.figure(figsize=(10, 6))
for bridge_type in df['bridge_type'].unique():
    subset = df[df['bridge_type'] == bridge_type]
    plt.scatter(subset['span_m'], subset['depth_of_girder'], 
               label=bridge_type, alpha=0.6, s=100)
plt.title('Span vs Girder Depth by Bridge Type', fontsize=14, fontweight='bold')
plt.xlabel('Span (m)', fontsize=12)
plt.ylabel('Depth of Girder (m)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '08_span_vs_depth.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_span_vs_depth.png")
plt.close()

# ========== Graph 9: Number of Spans Distribution ==========
plt.figure(figsize=(8, 6))
span_counts = df['num_spans'].value_counts().sort_index()
span_counts.plot(kind='bar', color='#34495e')
plt.title('Distribution of Number of Spans', fontsize=14, fontweight='bold')
plt.xlabel('Number of Spans', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '09_num_spans_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_num_spans_distribution.png")
plt.close()

# ========== Graph 10: Pier Cross Section Distribution ==========
plt.figure(figsize=(8, 6))
pier_section_counts = df['pier_cross_section'].value_counts()
colors = ['#e74c3c', '#3498db']
plt.pie(pier_section_counts, labels=pier_section_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors)
plt.title('Pier Cross Section Distribution', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig(output_dir / '10_pier_cross_section_pie.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 10_pier_cross_section_pie.png")
plt.close()

# ========== Graph 11: Bridge Type vs Pier Type Heatmap ==========
plt.figure(figsize=(8, 6))
cross_tab = pd.crosstab(df['bridge_type'], df['pier_type'])
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('Bridge Type vs Pier Type', fontsize=14, fontweight='bold')
plt.xlabel('Pier Type', fontsize=12)
plt.ylabel('Bridge Type', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / '11_bridge_vs_pier_type.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 11_bridge_vs_pier_type.png")
plt.close()

# ========== Graph 12: Multi-panel Summary ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Bridge Type
df['bridge_type'].value_counts().plot(kind='bar', ax=axes[0, 0], color='#3498db')
axes[0, 0].set_title('Bridge Type Distribution')
axes[0, 0].set_xlabel('Bridge Type')
axes[0, 0].set_ylabel('Count')

# Panel 2: Total Length
df['total_length_m'].hist(bins=15, ax=axes[0, 1], color='#2ecc71', edgecolor='black')
axes[0, 1].set_title('Total Length Distribution')
axes[0, 1].set_xlabel('Length (m)')
axes[0, 1].set_ylabel('Frequency')

# Panel 3: Lanes
df['lanes'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='#e74c3c')
axes[1, 0].set_title('Number of Lanes Distribution')
axes[1, 0].set_xlabel('Lanes')
axes[1, 0].set_ylabel('Count')

# Panel 4: Total Piers vs Bridge Type
df.groupby('bridge_type')['total_piers'].mean().plot(kind='bar', ax=axes[1, 1], color='#9b59b6')
axes[1, 1].set_title('Avg Piers by Bridge Type')
axes[1, 1].set_xlabel('Bridge Type')
axes[1, 1].set_ylabel('Avg Total Piers')

plt.suptitle('Bridge Dataset Summary', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / '12_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 12_summary_dashboard.png")
plt.close()

# ========== Print Summary Statistics ==========
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Bridges: {len(df)}")
print(f"\nBridge Types:")
print(df['bridge_type'].value_counts())
print(f"\nPier Types:")
print(df['pier_type'].value_counts())
print(f"\nAverage Total Length: {df['total_length_m'].mean():.2f} m")
print(f"Average Width: {df['width_m'].mean():.2f} m")
print(f"Average Number of Piers: {df['total_piers'].mean():.2f}")
print(f"\nAll plots saved to: {output_dir.absolute()}")
print("="*60)