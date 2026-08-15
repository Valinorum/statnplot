import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols

def get_pvalues(df, x_col, y_col, hue_col, hue_order=None):
    """
    Calculates pairwise p-values using independent t-tests.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis variable.
        y_col (str): The name of the column for the y-axis variable.
        hue_col (str): The name of the column for the hue grouping.
        hue_order (list, optional): The specific order for the hue categories.

    Returns:
        pd.DataFrame: A DataFrame with columns [x_col, 'group1', 'group2', 'p_value'].
    """
    x_categories = df[x_col].unique()
    
    # Check if the column data type is 'object'
    if df[y_col].dtype == 'object':
    # Convert text to numbers and change the type
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    if hue_order is None:
        hue_order = sorted(df[hue_col].unique())
    
    results = []
    if len(df[hue_col].unique()) == 2:
        data1 = df[(df[x_col] == x_categories[0])][y_col]
        data2 = df[(df[x_col] == x_categories[1])][y_col]
        if len(data1) > 1 and len(data2) > 1:
            _, p_value = ttest_ind(data1, data2, equal_var=False)
        else:
            p_value = float('nan')

        results.append({
            'group1': df[hue_col].unique()[0],
            'group2': df[hue_col].unique()[1],
            'p_value': p_value
        })
    else:

        for x_cat in x_categories:
            for hue1, hue2 in combinations(hue_order, 2):
                data1 = df[(df[x_col] == x_cat) & (df[hue_col] == hue1)][y_col]
                data2 = df[(df[x_col] == x_cat) & (df[hue_col] == hue2)][y_col]
    
                if len(data1) > 1 and len(data2) > 1:
                    _, p_value = ttest_ind(data1, data2, equal_var=False)
                else:
                    p_value = float('nan')
    
                results.append({
                    x_col: x_cat,
                    'group1': hue1,
                    'group2': hue2,
                    'p_value': p_value
                })
            
    return pd.DataFrame(results)

def get_pvalues_anova(df, x_col, y_col, hue_col):
    """
    Update: 2025-08-23
    Use the perform_anova_tukey(df, x_col, y_col, group_col, tukey_on_sig_only=False) function to calculate the detailed anova
    Calculates p-values using ANOVA and Tukey's HSD post-hoc test.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis variable.
        y_col (str): The name of the column for the y-axis variable.
        hue_col (str): The name of the column for the hue grouping.

    Returns:
        pd.DataFrame: A DataFrame with p-values from Tukey's HSD, with
                      columns [x_col, 'group1', 'group2', 'p_adj'].
    """
    x_categories = df[x_col].unique()
    hue_categories = df[hue_col].unique()
    
    all_results = []

    for x_cat in x_categories:
        sub_df = df[df[x_col] == x_cat]
        
        # Prepare data for ANOVA
        samples = [sub_df[sub_df[hue_col] == cat][y_col] for cat in hue_categories]
        
        # Check if there are at least 2 groups with data to compare
        if len([s for s in samples if len(s) > 1]) < 2:
            continue
            
        # Perform one-way ANOVA
        f_val, p_val_anova = f_oneway(*samples)

        # If ANOVA is significant, perform Tukey's HSD post-hoc test
        if p_val_anova < 0.05:
            tukey_results = pairwise_tukeyhsd(endog=sub_df[y_col], groups=sub_df[hue_col], alpha=0.05)
            results_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
            results_df[x_col] = x_cat
            all_results.append(results_df)
            
    if not all_results:
        return pd.DataFrame(columns=[x_col, 'group1', 'group2', 'p-adj'])
        
    return pd.concat(all_results, ignore_index=True)


def _add_significance_bar(ax, p_value, x1, x2, y, h, col='k'):
    """Internal function to add a single significance bar."""
    if p_value < 0.001: text = '***'
    elif p_value < 0.01: text = '**'
    elif p_value < 0.05: text = '*'
    else: return

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color=col)

def add_pvalue_annotations(df, x_col, y_col, hue_col, ax, hue_order=None, method='ttest'):
    """
    Adds significance bars to an existing plot, with a choice of statistical method.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col, y_col, hue_col (str): Column names.
        ax (matplotlib.axes.Axes): The Axes object to draw on.
        hue_order (list, optional): Order for the hue categories.
        method (str): 'ttest' for independent t-tests or 'anova' for ANOVA + Tukey's HSD.
    """
    # Check if the column data type is 'object'
    if df[y_col].dtype == 'object':
    # Convert text to numbers and change the type
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    
    if hue_order is None:
        hue_order = sorted(df[hue_col].unique())
    
    x_categories = df[x_col].unique()
    hue_positions = {cat: i for i, cat in enumerate(hue_order)}
    dodge_width = 0.8
    bar_width = dodge_width / len(hue_order)
    
    if len(df[hue_col].unique()) == 2:
        comparisons = []
        data1 = df[(df[x_col] == x_categories[0])][y_col]
        data2 = df[(df[x_col] == x_categories[1])][y_col]
        if len(data1) > 1 and len(data2) > 1:
            _, p_value = ttest_ind(data1, data2, equal_var=False)
            comparisons.append((data1, data2, p_value))
            print('The P-value for this group is: ',p_value)
    else:
        for i, x_cat in enumerate(x_categories):
            y_max = df[df[x_col] == x_cat][y_col].max()
            bar_height_offset = (df[y_col].max() - df[y_col].min()) * 0.05
            current_y = y_max + bar_height_offset *1.5
            top_annotation_y = current_y
    
            # Get p-values based on the chosen method
            if method == 'anova':
                sub_df = df[df[x_col] == x_cat]
                samples = [sub_df[sub_df[hue_col] == cat][y_col] for cat in hue_order]
                if len([s for s in samples if len(s) > 1]) < 2: continue
                _, p_val_anova = f_oneway(*samples)
                if p_val_anova >= 0.05: continue
                
                tukey_results = pairwise_tukeyhsd(endog=sub_df[y_col], groups=sub_df[hue_col], alpha=0.05)
                p_values_data = tukey_results._results_table.data
                comparisons = [(p[0], p[1], p[-1]) for p in p_values_data[1:]]
            else: # Default to t-test
                comparisons = []
                for hue1, hue2 in combinations(hue_order, 2):
                    data1 = df[(df[x_col] == x_cat) & (df[hue_col] == hue1)][y_col]
                    data2 = df[(df[x_col] == x_cat) & (df[hue_col] == hue2)][y_col]
                    if len(data1) > 1 and len(data2) > 1:
                        _, p_val = ttest_ind(data1, data2, equal_var=False)
                        comparisons.append((hue1, hue2, p_val))

        # Add annotations
        for group1, group2, p_value in comparisons:
            if p_value < 0.05:
                pos1 = hue_positions[group1]
                pos2 = hue_positions[group2]
                x1 = i - dodge_width / 2 + bar_width / 2 + pos1 * bar_width
                x2 = i - dodge_width / 2 + bar_width / 2 + pos2 * bar_width
                _add_significance_bar(ax, p_value, x1, x2, current_y, bar_height_offset)
                current_y += bar_height_offset * 1.5
                top_annotation_y = current_y

        current_bottom, current_top = ax.get_ylim()
        if top_annotation_y > current_top:
            new_top = top_annotation_y + bar_height_offset
            ax.set_ylim(bottom=current_bottom, top=new_top)
            
def annotate_plot_with_cld(ax, analysis_summary, data, x_col, y_col, group_col, hue_order, week_order, **text_kwargs):
    """
    Annotates a Seaborn plot by mathematically calculating annotation positions.

    Args:
        ax (matplotlib.axes.Axes): The axes object of the plot.
        analysis_summary (pd.DataFrame): The summary table with significance letters.
        data (pd.DataFrame): The full DataFrame used for plotting.
        x_col (str): The column name for the x-axis categories.
        y_col (str): The column name for the y-axis values.
        group_col (str): The column name for the hue categories.
        hue_order (list): The order of hue categories.
        week_order (list): The order of x-axis categories.
        **text_kwargs: Additional keyword arguments to pass to ax.text() for styling. Example is
        custom_style = {
    'color': 'red',
    'size': 14,
    'style': 'italic'
}
        annotate_plot_with_cld(
    ax=ax,
    analysis_summary=analysis_summary,
    data=data_tumor,
    x_col='week',
    y_col='Ct.Ar_distal(mm2)',
    group_col='group',
    hue_order=hue_order,
    week_order=week_order,
    **custom_style  # Pass the dictionary here
)
    """
    if analysis_summary.empty:
        print("Analysis summary is empty, skipping annotation.")
        return
        
    # Set default text styling and update with any user-provided arguments
    default_text_style = {'ha': 'center', 'va': 'bottom', 'color': 'black', 'weight': 'bold', 'size': 10}
    if text_kwargs:
        default_text_style.update(text_kwargs)

    y_offset = (data[y_col].max() - data[y_col].min()) * 0.05
    num_hue_groups = len(hue_order)
    dodge_width = 0.8 
    
    for week_idx, week in enumerate(week_order):
        for group_idx, group in enumerate(hue_order):
            letter_row = analysis_summary[(analysis_summary[x_col] == week) & (analysis_summary[group_col] == group)]
            
            if not letter_row.empty:
                letter = letter_row['significance'].iloc[0]
                group_data = data[(data[x_col] == week) & (data[group_col] == group)]
                if group_data.empty: 
                    continue
                
                max_val = group_data[y_col].max()
                y_pos = max_val + y_offset

                offset = (group_idx - (num_hue_groups - 1) / 2) * (dodge_width / num_hue_groups)
                x_pos = week_idx + offset
                
                ax.text(x_pos, y_pos, letter, **default_text_style)
    print(f"--- Annotation Process Finished for {y_col} ---")


def generate_compact_letters(tukey_df, means):
    """
    Generates a compact letter display from Tukey HSD results to show multiple letters
    if a group is not significantly different from multiple other groups.

    Args:
        tukey_df (pd.DataFrame): The DataFrame from the Tukey HSD results.
        means (pd.Series): A Series of group means, used for sorting.

    Returns:
        dict: A dictionary mapping group names to their compact letter string (e.g., 'a', 'b', 'ab').
    """
    # Sort groups by mean in descending order
    sorted_groups = means.sort_values(ascending=False).index.tolist()

    # Create a set of significant pairs for quick lookups
    sig_pairs = set()
    for _, row in tukey_df.iterrows():
        if row['reject']:
            pair = tuple(sorted((row['group1'], row['group2'])))
            sig_pairs.add(pair)

    # Step 1: Identify the founders of each letter group.
    # A new letter group is founded by the first group that is significantly
    # different from all existing founders.
    if not sorted_groups:
        return {}
        
    letter_founders = [sorted_groups[0]]
    for group in sorted_groups[1:]:
        is_sig_from_all_founders = all(
            tuple(sorted((group, founder))) in sig_pairs for founder in letter_founders
        )
        if is_sig_from_all_founders:
            letter_founders.append(group)

    # Step 2: Assign letters to all groups based on the founders.
    # Each group gets the letter of any founder it is NOT significantly different from.
    group_letters = {group: [] for group in sorted_groups}
    letter_map = {founder: chr(ord('A') + i) for i, founder in enumerate(letter_founders)}

    for group in sorted_groups:
        for founder in letter_founders:
            if group == founder or tuple(sorted((group, founder))) not in sig_pairs:
                group_letters[group].append(letter_map[founder])
    
    # Join the lists of letters into sorted strings for the final output
    final_letters = {group: "".join(sorted(letters)) for group, letters in group_letters.items()}
    
    return final_letters


def perform_anova_tukey(df, x_col, y_col, group_col, tukey_on_sig_only=False):
    """
    Performs ANOVA and optionally a Tukey HSD post-hoc test for each week and
    returns all results in a single consolidated DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column name for the weeks or time points or basically the x categorical variable.
        y_col (str): The column name for the dependent variable.
        group_col (str): The column name for the group identifiers.
        tukey_on_sig_only (bool): If True, only runs the Tukey HSD test if the ANOVA p-value is < 0.05.

    Returns:
        pd.DataFrame: A single DataFrame containing all analysis results, with separators for each week.
    """
    all_results_dfs = []
    weeks = df[x_col].unique() #weeks because initially the x variable was time point in weeks. You can have any x values

    for week in weeks:
        # Add a separator for the week's analysis
        separator_text = f"==================== Analysis for {week} ===================="
        all_results_dfs.append(pd.DataFrame([separator_text], columns=['Analysis Section']))
        
        week_df = df[df[x_col] == week].copy()
        model_formula = f'Q("{y_col}") ~ C({group_col})'
        
        try:
            if week_df[group_col].nunique() < 2:
                print(f"\nSkipping {week}: Not enough groups to perform ANOVA.")
                continue

            model = ols(model_formula, data=week_df).fit()
            
            # --- ANOVA Results ---
            anova_table = sm.stats.anova_lm(model, typ=2).reset_index()
            all_results_dfs.append(pd.DataFrame(["--- ANOVA Results ---"], columns=['Analysis Section']))
            all_results_dfs.append(anova_table)

            p_value = anova_table['PR(>F)'].iloc[0]

            # --- Tukey HSD and Summary (conditional) ---
            if not tukey_on_sig_only or p_value < 0.05:
                tukey_hsd = pairwise_tukeyhsd(endog=week_df[y_col], groups=week_df[group_col], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey_hsd._results_table.data[1:], columns=tukey_hsd._results_table.data[0])
                
                group_stats = week_df.groupby(group_col)[y_col].agg(['mean', 'std'])
                letters = generate_compact_letters(tukey_df, group_stats['mean'])
                group_stats['significance'] = group_stats.index.map(letters)

                tukey_df['group1_sig'] = tukey_df['group1'].map(letters)
                tukey_df['group2_sig'] = tukey_df['group2'].map(letters)
                
                cols_order = ['group1', 'group1_sig', 'group2', 'group2_sig', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']
                tukey_df = tukey_df[cols_order]

                # Add Tukey results to the list
                all_results_dfs.append(pd.DataFrame(["--- Tukey HSD Pairwise Comparison ---"], columns=['Analysis Section']))
                all_results_dfs.append(tukey_df)
                
                # Add Group Summary results to the list
                summary_df = group_stats.sort_values('mean', ascending=False).reset_index()
                all_results_dfs.append(pd.DataFrame(["--- Group Significance Summary ---"], columns=['Analysis Section']))
                all_results_dfs.append(summary_df)
            else:
                skip_text = f"ANOVA p-value ({p_value:.4f}) is not significant (< 0.05). Skipping Tukey HSD test."
                all_results_dfs.append(pd.DataFrame([skip_text], columns=['Analysis Section']))

        except Exception as e:
            error_text = f"Could not perform analysis for {week}. Error: {e}"
            all_results_dfs.append(pd.DataFrame([error_text], columns=['Analysis Section']))
        
        # Add a blank row for spacing
        all_results_dfs.append(pd.DataFrame([''], columns=['Analysis Section']))

    # Concatenate all DataFrames into a single one
    consolidated_df = pd.concat(all_results_dfs, ignore_index=True)
    return consolidated_df

def perform_summary_anova_tukey(df, x_col, y_col, group_col, tukey_on_sig_only=False):
    """
    Performs ANOVA and Tukey HSD test and returns a clean summary DataFrame 
    with significance letters, suitable for plotting.
    """
    all_summaries = []
    weeks = sorted(df[x_col].unique())

    for week in weeks:
        week_df = df[df[x_col] == week].copy()
        if week_df[group_col].nunique() < 2:
            continue

        model = ols(f'Q("{y_col}") ~ C({group_col})', data=week_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table['PR(>F)'].iloc[0]

        if not tukey_on_sig_only or p_value < 0.05:
            tukey_hsd = pairwise_tukeyhsd(endog=week_df[y_col], groups=week_df[group_col], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey_hsd._results_table.data[1:], columns=tukey_hsd._results_table.data[0])
            
            group_stats = week_df.groupby(group_col)[y_col].agg(['mean', 'std'])
            letters = generate_compact_letters(tukey_df, group_stats['mean'])
            group_stats['significance'] = group_stats.index.map(letters)
            
            summary_df = group_stats.reset_index()
            summary_df[x_col] = week
            all_summaries.append(summary_df)

    if not all_summaries:
        return pd.DataFrame()
        
    return pd.concat(all_summaries, ignore_index=True)


def analyze_multiple_y_cols(df, x_col, y_col_list, group_col, tukey_on_sig_only=False):
    """
    Wrapper function to run the analysis for a list of y-columns and consolidate results.
    """
    all_results_list = []

    for y_col in y_col_list:
        # Add a main header for the current y-column analysis
        y_col_header = f"******************** Analysis for Y-Column: {y_col} ********************"
        all_results_list.append(pd.DataFrame([y_col_header], columns=['Analysis Section']))
        
        # Get the detailed weekly analysis for this y-column
        weekly_results_df = perform_anova_tukey(
            df=df,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            tukey_on_sig_only=tukey_on_sig_only
        )
        all_results_list.append(weekly_results_df)
        
        # Add a blank row for spacing between y-columns
        all_results_list.append(pd.DataFrame([''], columns=['Analysis Section']))

    # Concatenate everything into the final DataFrame
    final_df = pd.concat(all_results_list, ignore_index=True)
    return final_df
