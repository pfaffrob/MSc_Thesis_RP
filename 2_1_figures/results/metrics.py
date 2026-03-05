"""
Generate PUBLICATION-READY markdown tables for thesis from UNet results.
Designed for compact, readable tables suitable for academic papers.
"""
import pandas as pd
from IPython.display import Markdown

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_loss(loss):
    """Format loss function names for Quarto acronyms."""
    loss_map = {
        'BCE': '{{< acr WBCE >}}',
        'DICE': '{{< acr DICE >}}',
        'BCEDICE': '{{< acr WBCE-D >}}'
    }
    return loss_map.get(loss, loss)

def format_band(band):
    """Format band combination names with LaTeX subscripts."""
    band_map = {
        'MS_REL': 'MS$_{rel}$',
        'MS_ABS': 'MS$_{abs}$',
        'IND_REL': 'IND$_{rel}$',
        'IND_ABS': 'IND$_{abs}$',
        'MS_REL_RENDVI': 'MS+IND$_{rel}$',
        'MS_ABS_RENDVI': 'MS+IND$_{abs}$'
    }
    return band_map.get(band, band)

def format_lr(lr):
    """Format learning rate for display."""
    if lr < 0.001:
        # Convert scientific notation to markdown format (e.g., 5e-05 -> 5x10^-5^)
        exp = int(f"{lr:.0e}".split('e')[1])
        coef = lr / (10 ** exp)
        return f"{coef:.0f}x10^{exp}^"
    return f"{lr}"

def format_reserve(reserve):
    """Format reserve names for display."""
    reserve_names = {
        'ESK': 'A1',
        'KAU': 'A2',
        'BUS': 'A3',
        'HAM': 'H1',
        'KAU_ESK_BUS_HAM': 'Combined'
    }
    return reserve_names.get(reserve, reserve)

# =============================================================================
# SINGLE-SITE TABLES (Compact, publication-ready)
# =============================================================================

def table_loss_function_comparison(df, bands=['RGB', 'MS_REL'], weight=10, lr=0.02, include_dice=True):
    """
    Table: Loss function comparison (compact).
    Shows mean ± std across reserves for each loss function.
    Fixed hyperparams: weight and lr (except for DICE which uses weight=1).
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    
    # Filter by bands and lr
    single = single[
        (single['Band Combination'].isin(bands)) &
        (single['Learning Rate'] == lr)
    ]
    
    # For BCE and BCEDICE, use specified weight; for DICE, use weight=1
    if include_dice:
        single = single[
            ((single['Loss Function'].isin(['BCE', 'BCEDICE'])) & (single['Maire Weight'] == weight)) |
            ((single['Loss Function'] == 'DICE') & (single['Maire Weight'] == 1))
        ]
    else:
        single = single[single['Maire Weight'] == weight]
    
    # Group by bands and loss function, calculate mean/std across reserves
    grouped = single.groupby(['Band Combination', 'Loss Function']).agg({
        'Maire IoU': ['mean', 'std'],
        'Maire F1': ['mean', 'std'],
        'Maire Precision': ['mean', 'std'],
        'Maire Recall': ['mean', 'std']
    }).round(3)
    
    # Sort by band then loss function
    loss_order = {'BCE': 0, 'DICE': 1, 'BCEDICE': 2}
    grouped = grouped.reset_index()
    grouped['loss_order'] = grouped['Loss Function'].map(loss_order)
    grouped = grouped.sort_values(['Band Combination', 'loss_order']).set_index(['Band Combination', 'Loss Function'])
    grouped = grouped.drop(columns='loss_order')
    
    lines = [
        "| Bands | Loss | F1 | IoU | Precision | Recall |",
        "|-------|------|-----|-----|-----------|--------|"
    ]
    
    for (band, loss), row in grouped.iterrows():
        iou_mean, iou_std = row['Maire IoU']['mean'], row['Maire IoU']['std']
        f1_mean, f1_std = row['Maire F1']['mean'], row['Maire F1']['std']
        prec_mean, prec_std = row['Maire Precision']['mean'], row['Maire Precision']['std']
        rec_mean, rec_std = row['Maire Recall']['mean'], row['Maire Recall']['std']
        
        lines.append(
            f"| {format_band(band)} | {format_loss(loss)} | {f1_mean:.2f}±{f1_std:.2f} | {iou_mean:.2f}±{iou_std:.2f} | "
            f"{prec_mean:.2f}±{prec_std:.2f} | {rec_mean:.2f}±{rec_std:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


def table_weight_comparison(df, bands=['RGB', 'MS_REL'], loss='BCEDICE', lr=0.02):
    """
    Table: Class weight comparison (compact).
    Shows mean ± std across reserves for each weight value.
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    single = single[
        (single['Band Combination'].isin(bands)) &
        (single['Loss Function'] == loss) &
        (single['Learning Rate'] == lr)
    ]
    
    grouped = single.groupby(['Band Combination', 'Maire Weight']).agg({
        'Maire IoU': ['mean', 'std'],
        'Maire F1': ['mean', 'std'],
        'Maire Precision': ['mean', 'std'],
        'Maire Recall': ['mean', 'std']
    }).round(3)
    
    lines = [
        "| Bands | Weight | F1 | IoU | Precision | Recall |",
        "|-------|--------|-----|-----|-----------|--------|"
    ]
    
    for (band, weight), row in grouped.iterrows():
        iou_mean, iou_std = row['Maire IoU']['mean'], row['Maire IoU']['std']
        f1_mean, f1_std = row['Maire F1']['mean'], row['Maire F1']['std']
        prec_mean, prec_std = row['Maire Precision']['mean'], row['Maire Precision']['std']
        rec_mean, rec_std = row['Maire Recall']['mean'], row['Maire Recall']['std']
        
        lines.append(
            f"| {format_band(band)} | {int(weight)} | {f1_mean:.2f}±{f1_std:.2f} | {iou_mean:.2f}±{iou_std:.2f} | "
            f"{prec_mean:.2f}±{prec_std:.2f} | {rec_mean:.2f}±{rec_std:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


def table_learning_rate_comparison(df, bands=['RGB', 'MS_REL'], loss='BCEDICE', weight=10, include_dice=True):
    """
    Table: Learning rate comparison (compact).
    Shows mean ± std across reserves for each LR.
    Includes DICE loss (with weight=1) if include_dice=True.
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    single = single[single['Band Combination'].isin(bands)]
    
    if include_dice:
        # Include BCEDICE with specified weight AND DICE with weight=1
        single = single[
            ((single['Loss Function'] == loss) & (single['Maire Weight'] == weight)) |
            ((single['Loss Function'] == 'DICE') & (single['Maire Weight'] == 1))
        ]
    else:
        single = single[
            (single['Loss Function'] == loss) &
            (single['Maire Weight'] == weight)
        ]
    
    grouped = single.groupby(['Band Combination', 'Loss Function', 'Learning Rate']).agg({
        'Maire IoU': ['mean', 'std'],
        'Maire F1': ['mean', 'std'],
        'Maire Precision': ['mean', 'std'],
        'Maire Recall': ['mean', 'std']
    }).round(3)
    
    lines = [
        "| Bands | Loss | LR | F1 | IoU | Precision | Recall |",
        "|-------|------|----|-----|-----|-----------|--------|"
    ]
    
    for (band, loss_fn, lr), row in grouped.iterrows():
        iou_mean, iou_std = row['Maire IoU']['mean'], row['Maire IoU']['std']
        f1_mean, f1_std = row['Maire F1']['mean'], row['Maire F1']['std']
        prec_mean, prec_std = row['Maire Precision']['mean'], row['Maire Precision']['std']
        rec_mean, rec_std = row['Maire Recall']['mean'], row['Maire Recall']['std']
        
        lines.append(
            f"| {format_band(band)} | {format_loss(loss_fn)} | {format_lr(lr)} | {f1_mean:.2f}±{f1_std:.2f} | {iou_mean:.2f}±{iou_std:.2f} | "
            f"{prec_mean:.2f}±{prec_std:.2f} | {rec_mean:.2f}±{rec_std:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


def table_band_comparison_pivot(df, weight=10, lr=0.02, loss='BCEDICE', metric='Maire F1'):
    """
    Table: Band comparison as pivot table (reserves as rows, bands as columns).
    Very compact - shows one metric only.
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    single = single[
        (single['Maire Weight'] == weight) &
        (single['Learning Rate'] == lr) &
        (single['Loss Function'] == loss)
    ]
    
    # Pivot
    pivot = single.pivot_table(
        index='Reserve(s)',
        columns='Band Combination',
        values=metric,
        aggfunc='first'
    )
    
    # Order columns sensibly
    col_order = ['RGB', 'MS_REL', 'MS_ABS', 'IND_REL', 'IND_ABS', 'MS_REL_RENDVI', 'MS_ABS_RENDVI']
    cols = [c for c in col_order if c in pivot.columns]
    pivot = pivot[cols]
    
    # Format column headers
    formatted_cols = [format_band(c) for c in cols]
    
    # Order rows by reserve order from format_reserve
    reserve_order = ['ESK', 'KAU', 'BUS', 'HAM']
    ordered_reserves = [r for r in reserve_order if r in pivot.index]
    
    lines = [
        "| Reserve | " + " | ".join(formatted_cols) + " |",
        "|---------|" + "|".join(["-------" for _ in cols]) + "|"
    ]
    
    for reserve in ordered_reserves:
        vals = [f"{pivot.loc[reserve, c]:.2f}" if pd.notna(pivot.loc[reserve, c]) else "-" for c in cols]
        lines.append(f"| {format_reserve(reserve)} | " + " | ".join(vals) + " |")
    
    # Mean row
    means = [f"**{pivot[c].mean():.2f}**" if c in pivot.columns else "-" for c in cols]
    lines.append(f"| **Mean** | " + " | ".join(means) + " |")
    
    return Markdown('\n'.join(lines))


def table_best_models_summary(df, top_n=5):
    """
    Table: Top N best performing single-site model configurations.
    Ranked by F1, aggregated across reserves.
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    
    # Average across reserves for each config
    grouped = single.groupby(['Band Combination', 'Loss Function', 'Maire Weight', 'Learning Rate']).agg({
        'Maire IoU': ['mean', 'std', 'count'],
        'Maire F1': ['mean', 'std']
    }).round(3)
    
    grouped.columns = ['IoU_mean', 'IoU_std', 'n_reserves', 'F1_mean', 'F1_std']
    grouped = grouped.reset_index().sort_values('F1_mean', ascending=False).head(top_n)
    
    lines = [
        "| Bands | Loss | W | LR | F1 | IoU |",
        "|--------------|------|---|-----|-----|-----|"
    ]
    
    for _, row in grouped.iterrows():
        lines.append(
            f"| {format_band(row['Band Combination'])} | {format_loss(row['Loss Function'])} | "
            f"{int(row['Maire Weight'])} | {format_lr(row['Learning Rate'])} | "
            f"{row['F1_mean']:.2f}±{row['F1_std']:.2f} | {row['IoU_mean']:.2f}±{row['IoU_std']:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


def table_top_individual_models(df, top_n=15, metric='Maire F1'):
    """
    Table: Top N individual models (no aggregation).
    Shows each reserve/config combination separately.
    """
    single = df[~df['Reserve(s)'].str.contains('_', na=False)].copy()
    single = single.sort_values(metric, ascending=False).head(top_n)
    
    lines = [
        "| Reserve | Bands | Loss | W | LR | F1 | IoU | Prec | Rec |",
        "|---------|--------------|------|---|-----|-----|-----|------|-----|"
    ]
    
    for _, row in single.iterrows():
        lines.append(
            f"| {format_reserve(row['Reserve(s)'])} | {format_band(row['Band Combination'])} | {format_loss(row['Loss Function'])} | "
            f"{int(row['Maire Weight'])} | {format_lr(row['Learning Rate'])} | "
            f"{row['Maire F1']:.2f} | {row['Maire IoU']:.2f} | "
            f"{row['Maire Precision']:.2f} | {row['Maire Recall']:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


# =============================================================================
# MULTI-SITE TABLES
# =============================================================================

def table_multi_site_summary(df):
    """
    Table: Multi-site model comparison (compact).
    One row per band combination, showing best config.
    """
    multi = df[df['Reserve(s)'].str.contains('_', na=False)].copy()
    
    if multi.empty:
        return Markdown("*No multi-site results available*")
    
    # Get best config per band combination
    idx = multi.groupby('Band Combination')['Maire F1'].idxmax()
    best = multi.loc[idx].sort_values('Maire F1', ascending=False)
    
    lines = [
        "| Bands | Loss | Weight | F1 | IoU | Precision | Recall |",
        "|-------|------|--------|-----|-----|-----------|--------|"
    ]
    
    for _, row in best.iterrows():
        lines.append(
            f"| {format_band(row['Band Combination'])} | {format_loss(row['Loss Function'])} | "
            f"{int(row['Maire Weight'])} | "
            f"{row['Maire F1']:.2f} | {row['Maire IoU']:.2f} | "
            f"{row['Maire Precision']:.2f} | {row['Maire Recall']:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


def table_multi_site_all(df, max_rows=None):
    """
    Table: All multi-site models (if few enough to show).
    """
    multi = df[df['Reserve(s)'].str.contains('_', na=False)].copy()
    
    if multi.empty:
        return Markdown("*No multi-site results available*")
    
    multi = multi.sort_values('Maire F1', ascending=False)
    
    # Limit rows if specified
    if max_rows is not None:
        multi = multi.head(max_rows)
    
    lines = [
        "| Bands       | Loss | Weight | F1 | IoU | Prec | Rec |",
        "|-------------|------|--------|-----|-----|------|-----|"
    ]
    
    for _, row in multi.iterrows():
        lines.append(
            f"| {format_band(row['Band Combination'])} | {format_loss(row['Loss Function'])} | "
            f"{int(row['Maire Weight'])} | "
            f"{row['Maire F1']:.2f} | {row['Maire IoU']:.2f} | "
            f"{row['Maire Precision']:.2f} | {row['Maire Recall']:.2f} |"
        )
    
    return Markdown('\n'.join(lines))


# =============================================================================
# DATASET TABLES
# =============================================================================

def table_class_distribution(datasets_dir='/Users/robinpfaff/Library/CloudStorage/OneDrive-AUTUniversity/MA/aa566b206b36b985ac2ad0e73eedfc197cc8d2ffc/5_3_unet/datasets', band='MS_REL', 
                              reserves=['ESK', 'KAU', 'BUS', 'HAM', 'KAU_ESK_BUS_HAM']):
    """
    Table: Class distribution across reserves from class_distribution.txt files.
    Shows training and validation set statistics for each reserve.
    """
    import re
    from pathlib import Path
    
    datasets_path = Path(datasets_dir)
    results = []
    
    for reserve in reserves:
        folder = datasets_path / f'{band}_{reserve}'
        txt_file = folder / 'class_distribution.txt'
        
        if not txt_file.exists():
            continue
        
        with open(txt_file, 'r') as f:
            content = f.read()
        
        # Parse train set
        train_match = re.search(
            r'TRAIN_MASKS SET:.*?Number of images: (\d+).*?Total pixels: ([\d,]+).*?'
            r'Class 0.*?: *([\d,]+) pixels.*?Class 1.*?: *([\d,]+) pixels \( *(\d+\.\d+)%\).*?'
            r'Class imbalance ratio.*?: *([\d.]+):1',
            content, re.DOTALL
        )
        
        # Parse valid set
        valid_match = re.search(
            r'VALID_MASKS SET:.*?Number of images: (\d+).*?Total pixels: ([\d,]+).*?'
            r'Class 0.*?: *([\d,]+) pixels.*?Class 1.*?: *([\d,]+) pixels \( *(\d+\.\d+)%\).*?'
            r'Class imbalance ratio.*?: *([\d.]+):1',
            content, re.DOTALL
        )
        
        if train_match and valid_match:
            results.append({
                'reserve': reserve,
                'reserve_name': format_reserve(reserve),
                'train_images': int(train_match.group(1)),
                'train_maire_pct': float(train_match.group(5)),
                'valid_images': int(valid_match.group(1)),
                'valid_maire_pct': float(valid_match.group(5)),
            })
    
    lines = [
        "| Reserve | Train Images | Train Maire (%) | Val Images | Val Maire (%) |",
        "|---------|--------------|-----------------|------------|---------------|"
    ]
    
    for r in results:
        lines.append(
            f"| {r['reserve_name']} | {r['train_images']} | {r['train_maire_pct']:.2f} |"
            f"{r['valid_images']} | {r['valid_maire_pct']:.2f} |"
        )
    
    return Markdown('\n'.join(lines))
