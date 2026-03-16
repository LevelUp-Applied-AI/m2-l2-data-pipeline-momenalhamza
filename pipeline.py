"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Complete each function below. Remove the TODO: comments and pass statements
as you implement each function. Do not change the function signatures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load sales records from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Raw sales records DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def clean_data(df):
    """Handle missing values and fix data types.

    Args:
        df (pd.DataFrame): Raw DataFrame from load_data().

    Returns:
        pd.DataFrame: Cleaned DataFrame (do not modify the input in place).
    """
    df = df.copy()
    df['quantity'] = df['quantity'].fillna(df['quantity'].median())
    df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"Cleaned data: {len(df)} records")
    return df


def add_features(df):
    """Compute derived columns.

    Args:
        df (pd.DataFrame): Cleaned DataFrame from clean_data().

    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    df = df.copy()
    df['revenue'] = df['quantity'] * df['unit_price']
    df['day_of_week'] = df['date'].dt.day_name()
    return df


def generate_summary(df):
    """Compute summary statistics.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().

    Returns:
        dict: Summary statistics.
    """
    top_category = df.groupby('product_category')['revenue'].sum().idxmax()
    return {
        'total_revenue': df['revenue'].sum(),
        'avg_order_value': df['revenue'].mean(),
        'top_category': top_category,
        'record_count': len(df)
    }


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create and save 3 charts as PNG files.

    Args:
        df (pd.DataFrame): Enriched DataFrame from add_features().
        output_dir (str): Directory to save PNG files (create if needed).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1 — Bar chart: total revenue by product category
    revenue_by_cat = df.groupby('product_category')['revenue'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(revenue_by_cat.index, revenue_by_cat.values, color='steelblue', edgecolor='white')
    ax.set_title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Product Category', fontsize=12)
    ax.set_ylabel('Total Revenue (JOD)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/revenue_by_category.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 2 — Line chart: daily revenue trend
    daily_revenue = df.groupby('date')['revenue'].sum().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_revenue.index, daily_revenue.values, color='steelblue', linewidth=1.5)
    ax.set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Revenue (JOD)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/daily_revenue_trend.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 3 — Horizontal bar chart: avg order value by payment method
    avg_by_payment = df.groupby('payment_method')['revenue'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(avg_by_payment.index, avg_by_payment.values, color='steelblue', edgecolor='white')
    ax.set_title('Average Order Value by Payment Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Order Value (JOD)', fontsize=12)
    ax.set_ylabel('Payment Method', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/avg_order_by_payment.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Run the full data pipeline end-to-end."""
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)
    summary = generate_summary(df)
    print("\n=== Summary ===")
    print(f"Total Revenue:       {summary['total_revenue']:.2f}")
    print(f"Average Order Value: {summary['avg_order_value']:.2f}")
    print(f"Top Category:        {summary['top_category']}")
    print(f"Record Count:        {summary['record_count']}")
    create_visualizations(df)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()