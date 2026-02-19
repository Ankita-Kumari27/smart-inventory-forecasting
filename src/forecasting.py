import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

OUTPUT_DIR = 'output/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass


def plot_sales_overview(df, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sales Overview Dashboard', fontsize=16, fontweight='bold')
    ds = df.groupby('date')['quantity_sold'].sum()
    axes[0,0].plot(ds.index, ds.values, alpha=0.3, linewidth=0.5, color='steelblue')
    axes[0,0].plot(ds.rolling(30).mean(), color='red', linewidth=2, label='30d MA')
    axes[0,0].plot(ds.rolling(90).mean(), color='green', linewidth=2, linestyle='--', label='90d MA')
    axes[0,0].set_title('Daily Sales')
    axes[0,0].legend(fontsize=7)
    mr = df.groupby([pd.Grouper(key='date', freq='M'), 'category'])['revenue'].sum().reset_index()
    for cat in mr['category'].unique():
        cd = mr[mr['category']==cat]
        axes[0,1].plot(cd['date'], cd['revenue'], label=cat)
    axes[0,1].set_title('Revenue by Category')
    axes[0,1].legend(fontsize=7)
    dwo = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dws = df.groupby('day_of_week')['quantity_sold'].mean().reindex(dwo)
    colors = ['#F44336' if i >= 5 else '#2196F3' for i in range(7)]
    axes[1,0].bar(range(7), dws.values, color=colors)
    axes[1,0].set_xticks(range(7))
    axes[1,0].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    axes[1,0].set_title('Avg by Day')
    tp = df.groupby('product_name')['quantity_sold'].sum().sort_values(ascending=True)
    axes[1,1].barh(range(len(tp)), tp.values, color='#00BCD4')
    axes[1,1].set_yticks(range(len(tp)))
    axes[1,1].set_yticklabels(tp.index, fontsize=7)
    axes[1,1].set_title('Sales by Product')
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'sales_overview.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_product_analysis(product_daily, product_name, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Analysis: {product_name}', fontsize=16, fontweight='bold')
    axes[0,0].plot(product_daily['date'], product_daily['quantity_sold'], alpha=0.4, linewidth=0.5)
    axes[0,0].plot(product_daily['date'], product_daily['quantity_sold'].rolling(30).mean(), color='red', linewidth=2)
    axes[0,0].set_title('Daily Sales')
    pdf = product_daily.copy()
    pdf['month'] = pdf['date'].dt.month
    md = [pdf[pdf['month']==m]['quantity_sold'].values for m in range(1,13)]
    bp = axes[0,1].boxplot(md, labels=['J','F','M','A','M','J','J','A','S','O','N','D'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#00BCD4')
        patch.set_alpha(0.6)
    axes[0,1].set_title('Monthly')
    mean_val = product_daily['quantity_sold'].mean()
    axes[1,0].hist(product_daily['quantity_sold'], bins=50, color='steelblue', edgecolor='white')
    axes[1,0].axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    axes[1,0].set_title('Distribution')
    axes[1,0].legend()
    pdf['year'] = pdf['date'].dt.year
    for y in sorted(pdf['year'].unique()):
        yd = pdf[pdf['year']==y]
        axes[1,1].plot(yd.groupby('month')['quantity_sold'].mean(), marker='o', label=str(y))
    axes[1,1].set_title('YoY')
    axes[1,1].legend(fontsize=7)
    plt.tight_layout()
    if save:
        sn = product_name.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'product_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_model_comparison(results_df, save=True):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    x = range(len(results_df))
    m = results_df['model'].values
    metrics = [('MAE', 'salmon'), ('RMSE', 'lightskyblue'), ('MAPE', 'lightyellow'), ('R2', 'lightgreen')]
    for ax, (metric, color) in zip(axes, metrics):
        vals = results_df[metric].values
        bars = ax.bar(x, vals, color=color, edgecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(m, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric)
        if metric == 'R2':
            best_idx = vals.argmax()
        else:
            best_idx = vals.argmin()
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_predictions_vs_actual(test_df, predictions_dict, best_model_name, save=True):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    dates = test_df['date'].values
    actual = test_df['quantity_sold'].values
    axes[0].plot(dates, actual, 'k-', linewidth=2, label='Actual')
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(predictions_dict)))
    for (name, pred), color in zip(predictions_dict.items(), colors_list):
        lw = 2.5 if name == best_model_name else 1.0
        alpha = 1.0 if name == best_model_name else 0.4
        axes[0].plot(dates, pred, linewidth=lw, alpha=alpha, color=color, label=name)
    axes[0].set_title('All Models vs Actual')
    axes[0].legend(fontsize=7)
    bp = predictions_dict[best_model_name]
    axes[1].plot(dates, actual, 'k-', linewidth=2, label='Actual')
    axes[1].plot(dates, bp, 'r-', linewidth=2, label=f'Best: {best_model_name}')
    axes[1].fill_between(dates, actual, bp, alpha=0.2, color='red')
    errors = np.abs(actual - bp)
    high_error = errors > np.percentile(errors, 90)
    axes[1].scatter(dates[high_error], actual[high_error], color='orange', s=30, zorder=5, label='High Error')
    axes[1].set_title(f'Best: {best_model_name}')
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'predictions_vs_actual.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_forecast(historical_df, forecast_df, product_name, save=True):
    fig, ax = plt.subplots(figsize=(16, 6))
    recent = historical_df.tail(90)
    ax.plot(recent['date'], recent['quantity_sold'], 'b-', linewidth=1.5, label='Historical')
    ax.plot(forecast_df['date'], forecast_df['predicted_demand'], 'r--', linewidth=2, label='Forecast', marker='o', markersize=3)
    ax.fill_between(forecast_df['date'], forecast_df['predicted_demand']*0.8, forecast_df['predicted_demand']*1.2, alpha=0.15, color='red', label='±20% Band')
    ax.axvline(x=forecast_df['date'].iloc[0], color='gray', linestyle=':', label='Forecast Start')
    ax.set_title(f'Forecast: {product_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    plt.tight_layout()
    if save:
        sn = product_name.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'forecast_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_feature_importance(importance_df, model_name, top_n=15, save=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(top_n)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), top['importance'].values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=8)
    ax.set_title(f'Top {top_n} Features ({model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_inventory_dashboard(forecast_df, recommendations, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    pn = recommendations['product']
    fig.suptitle(f'Inventory: {pn}', fontsize=16, fontweight='bold')

    # Panel 1: Demand
    axes[0,0].bar(forecast_df['date'], forecast_df['predicted_demand'], color='steelblue', alpha=0.7)
    axes[0,0].axhline(y=recommendations['avg_daily_demand'], color='red', linestyle='--', label=f'Avg: {recommendations["avg_daily_demand"]}')
    axes[0,0].set_title('Forecast Demand')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)

    # Panel 2: Stock projection
    stk = recommendations['current_stock']
    sl = [stk]
    for d in forecast_df['predicted_demand'].values:
        stk = max(0, stk-d)
        sl.append(stk)
    axes[0,1].plot(sl, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0,1].axhline(y=recommendations['reorder_point'], color='orange', linestyle='--', label='Reorder')
    axes[0,1].axhline(y=recommendations['safety_stock'], color='red', linestyle='--', label='Safety')
    axes[0,1].fill_between(range(len(sl)), 0, recommendations['safety_stock'], alpha=0.1, color='red')
    axes[0,1].set_title('Stock Projection')
    axes[0,1].legend()

    # Panel 3: Risk info
    risk = recommendations['stockout_risk']
    risk_val = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}.get(risk, 1)
    risk_color = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'orangered', 'CRITICAL': 'red'}.get(risk, 'gray')
    axes[1,0].barh(['Risk'], [risk_val], color=risk_color, height=0.4)
    axes[1,0].set_xlim(0, 4.5)
    axes[1,0].set_xticks([1, 2, 3, 4])
    axes[1,0].set_xticklabels(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    axes[1,0].set_title(f'Risk: {risk}')
    info = f"Stock: {recommendations['current_stock']}\nDays Left: {recommendations['days_of_stock_remaining']:.0f}\nSafety: {recommendations['safety_stock']}\nReorder: {recommendations['reorder_point']}\nOrder: {recommendations['recommended_order_qty']}\nEOQ: {recommendations.get('economic_order_qty', 'N/A')}"
    axes[1,0].text(4.3, 0, info, fontsize=8, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Panel 4: Cumulative demand vs stock
    cum_demand = forecast_df['predicted_demand'].cumsum()
    axes[1,1].plot(cum_demand.values, color='red', linewidth=2, label='Cumulative Demand')
    axes[1,1].axhline(y=recommendations['current_stock'], color='blue', linestyle='--', linewidth=2, label='Current Stock')
    if recommendations.get('stockout_day'):
        axes[1,1].axvline(x=recommendations['stockout_day']-1, color='red', linestyle=':', linewidth=2, label=f'Stockout Day {recommendations["stockout_day"]}')
    axes[1,1].set_title('Cumulative Demand vs Stock')
    axes[1,1].legend(fontsize=7)

    plt.tight_layout()
    if save:
        sn = pn.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'dashboard_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_scenario_comparison(scenarios, product_name, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Scenarios: {product_name}', fontsize=16, fontweight='bold')
    scenario_colors = {'optimistic': 'green', 'normal': 'steelblue', 'pessimistic': 'red'}
    for name, data in scenarios.items():
        fc = data['forecast']
        axes[0].plot(fc['date'], fc['predicted_demand'], color=scenario_colors.get(name, 'gray'), linewidth=2, label=name.title())
    axes[0].set_title('Demand Scenarios')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    names = list(scenarios.keys())
    totals = [s['recommendation']['total_forecasted_demand'] for s in scenarios.values()]
    orders = [s['recommendation']['recommended_order_qty'] for s in scenarios.values()]
    x = np.arange(len(names))
    axes[1].bar(x - 0.175, totals, 0.35, label='Total Demand', color='steelblue')
    axes[1].bar(x + 0.175, orders, 0.35, label='Order Qty', color='orange')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.title() for n in names])
    axes[1].set_title('Scenario Metrics')
    axes[1].legend()
    plt.tight_layout()
    if save:
        sn = product_name.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'scenarios_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()