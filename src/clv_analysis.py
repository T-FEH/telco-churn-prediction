# src/clv_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_processed_data(processed_dir):
    """Load processed train, val, and test splits."""
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))['Churn']
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv'))['Churn']
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))['Churn']
    
    # Combine features and target for analysis
    train = X_train.copy()
    train['Churn'] = y_train
    val = X_val.copy()
    val['Churn'] = y_val
    test = X_test.copy()
    test['Churn'] = y_test
    
    # Concatenate all data for CLV analysis
    df = pd.concat([train, val, test], ignore_index=True)
    return df

def segment_clv(df):
    """Segment customers into CLV quartiles."""
    df['CLV_quartile'] = pd.qcut(df['CLV'], q=4, labels=['Low', 'Med', 'High', 'Premium'])
    return df

def compute_churn_rates(df):
    """Calculate churn rate by CLV quartile."""
    churn_rates = df.groupby('CLV_quartile', observed=True)['Churn'].mean().reset_index()
    churn_rates['Churn'] = churn_rates['Churn'] * 100  # Convert to percentage
    return churn_rates

def plot_clv_distribution(df, output_dir):
    """Plot and save CLV distribution histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['CLV'], bins=30, color='#1f77b4', edgecolor='black')
    plt.title('Customer Lifetime Value (CLV) Distribution')
    plt.xlabel('CLV ($)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clv_distribution.png'))
    plt.close()

def plot_churn_by_quartile(churn_rates, output_dir):
    """Plot and save churn rate by CLV quartile."""
    plt.figure(figsize=(8, 5))
    plt.bar(churn_rates['CLV_quartile'], churn_rates['Churn'], color='#ff7f0e')
    plt.title('Churn Rate by CLV Quartile')
    plt.xlabel('CLV Quartile')
    plt.ylabel('Churn Rate (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'churn_by_quartile.png'))
    plt.close()

def generate_insights(df, churn_rates):
    """Generate 2–3 business insights based on CLV and churn analysis."""
    insights = [
        f"High-value customers (Premium CLV quartile) have a churn rate of {churn_rates[churn_rates['CLV_quartile'] == 'Premium']['Churn'].iloc[0]:.1f}%, indicating a priority for retention strategies targeting this segment.",
        f"Low CLV quartile customers churn at {churn_rates[churn_rates['CLV_quartile'] == 'Low']['Churn'].iloc[0]:.1f}%, suggesting they may be less engaged and could benefit from upselling or engagement campaigns.",
        "Customers with higher CLV tend to have longer expected tenure, emphasizing the importance of long-term contracts to boost retention."
    ]
    return insights

def save_insights(insights, output_dir):
    """Save insights to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'clv_insights.txt'), 'w') as f:
        for insight in insights:
            f.write(f"- {insight}\n")

def main():
    # File paths
    processed_dir = 'data/processed/'
    output_dir = 'data/processed/plots/'
    
    # Load data
    df = load_processed_data(processed_dir)
    
    # Segment CLV into quartiles
    df = segment_clv(df)
    
    # Compute churn rates by quartile
    churn_rates = compute_churn_rates(df)
    
    # Generate plots
    plot_clv_distribution(df, output_dir)
    plot_churn_by_quartile(churn_rates, output_dir)
    
    # Generate and save insights
    insights = generate_insights(df, churn_rates)
    save_insights(insights, output_dir)
    
    # Save churn rates for app
    churn_rates.to_csv(os.path.join(output_dir, 'churn_rates.csv'), index=False)
    
    print(f"CLV analysis complete. Plots and insights saved to {output_dir}")

if __name__ == "__main__":
    main()