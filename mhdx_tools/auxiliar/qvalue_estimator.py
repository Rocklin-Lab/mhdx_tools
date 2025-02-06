import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
import argparse
from scipy.stats import zscore

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
import argparse
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os

def plot_score_distributions(target_scores, decoy_scores, prob_threshold_value, output_path=None):
    """
    Plot the score distributions for decoy and target sequences.

    Parameters:
    - target_scores (array-like): Scores for target sequences.
    - decoy_scores (array-like): Scores for decoy sequences.
    - output_path (str): Path to save the plot as PNG. If None, displays the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

    # Calculate the difference in length
    length_diff = len(target_scores) - len(decoy_scores)

    # Append zeros to A if B is longer
    if length_diff > 0:
        decoy_scores = np.append(decoy_scores, [0] * length_diff)


    sns.ecdfplot(target_scores, ax=ax, lw=1, label='Target Sequences')
    sns.ecdfplot(decoy_scores, ax=ax, lw=1, label='Decoy Sequences')

    ax.axvline(prob_threshold_value, color='red', lw=0.8, ls='--')

    n_targets_i, n_targets_f = len(target_scores), len(target_scores[target_scores > prob_threshold_value])

    ax.text(
        0.01, 0.5,
        r"$\mathit{N}$$_{target,initial}$=" + f"{n_targets_i}\n" +
        r"$\mathit{N}$$_{target,final}$=" + f"{n_targets_f}\n" +
        r"prob_threshold=" + f"{prob_threshold_value:0.3f}",
        size=6, ha='left', va='top',
        transform=ax.transAxes
    )

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make spines thinner
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Make ticks thinner and shorter
    ax.tick_params(axis='both', width=0.5, length=3, labelsize=8)

    # Set x-axis ticks from 0 to 1 at intervals of 0.2
    ax.set_xticks(np.arange(0, 1.1, 0.2))  # Include 1.0
    ax.set_xlabel("Probability", size=8)
    ax.set_ylabel("Proportion", size=8)

    # Add grid lines
    ax.grid(alpha=0.5, lw=0.2, zorder=-1, ls='--')

    # Add legend
    ax.legend(frameon=False, fontsize=6, loc=2)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, format='png', dpi=600, bbox_inches='tight')
    else:
        plt.show()


def optimized_amino_acid_composition(sequence):
    """Calculate amino acid composition as fractions and include sequence length."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    amino_acid_count = Counter(sequence)
    total_aa = len(sequence)
    amino_acid_fraction = [amino_acid_count[aa] / total_aa if total_aa else 0.0 for aa in amino_acids]
    return amino_acid_fraction + [total_aa]


def train_aa_rt_model(df, use_all=False):
    """Train a Lasso model to predict RT using amino acid composition."""
    if not use_all:
        df = df[~df['name'].str.contains("decoy")]

    X = np.vstack(df['sequence'].apply(optimized_amino_acid_composition))
    y = df['RT']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(alpha=1e-3)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print(f"RT Model: MSE={mean_squared_error(y, y_pred):.2f}, R2={r2_score(y, y_pred):.3f}")
    return scaler, model


def calculate_rt_residuals(df, scaler, model):
    """Calculate RT residuals using a trained model."""
    X = np.vstack(df['sequence'].apply(optimized_amino_acid_composition))
    X_scaled = scaler.transform(X)
    y = df['RT']
    residuals = (y - model.predict(X_scaled)) ** 2
    return residuals


def preprocess_dataframe(df):
    """Preprocess the input dataframe."""
    # Add composite scores
    df["fxn_zscore_idotp_ppm"] = zscore(df["idotp"]) * df["abs_ppm"]
    df["fxn_zscore_ppm_idotp"] = zscore(df["abs_ppm"]) * df["idotp"]

    # Define the full range of charges you want to include
    charge_range = list(range(3, 13))
    # Ensure the 'charge' column is treated as a categorical variable with the full range
    df['charge'] = pd.Categorical(df['charge'], categories=charge_range)
    # One-hot encode the charge column, ensuring all specified categories are included
    one_hot = pd.get_dummies(df['charge'], prefix='charge')

    df = pd.concat([df, one_hot], axis=1)

    # Aggregate statistics
    grouped_df = df.groupby(['name', 'sequence']).agg(
        mean_n_charges=('n_charges', 'mean'),
        min_abs_ppm=('abs_ppm', 'min'),
        mean_abs_ppm=('abs_ppm', 'mean'),
        max_abs_ppm=('abs_ppm', 'max'),
        min_idotp=('idotp', 'min'),
        mean_idotp=('idotp', 'mean'),
        max_idotp=('idotp', 'max'),
        RT=('RT', 'mean'),
        min_ab_cluster_total=('ab_cluster_total', 'min'),
        mean_ab_cluster_total=('ab_cluster_total', 'mean'),
        max_ab_cluster_total=('ab_cluster_total', 'max'),
        sum_ab_cluster_total=('ab_cluster_total', 'sum'),
        fxn_zscore_idotp_ppm=('fxn_zscore_idotp_ppm', 'min'),  # Aggregate as min
        fxn_zscore_ppm_idotp=('fxn_zscore_ppm_idotp', 'max'),  # Aggregate as max
        # Summing the one-hot encoded values
        **{f'{col}': pd.NamedAgg(column=col, aggfunc='sum') for col in one_hot.columns},
    ).reset_index()

    # Add n_or_more_charge_states features
    def n_or_more_charge_states(row, n):
        charge_columns = [col for col in row.index if col.startswith('charge_')]
        return row[charge_columns].sum() >= n

    for n in range(1, 5):
        grouped_df[f'{n}_or_more_charge_states'] = grouped_df.apply(
            lambda x: n_or_more_charge_states(x, n), axis=1
        ).astype(int)

    # Add nth best idotp and ppm features
    def nth_best_value(group, column, n, ascending):
        sorted_values = group.sort_values(column, ascending=ascending)[column]
        return sorted_values.iloc[n-1] if len(sorted_values) >= n else sorted_values.iloc[-1]

    for n in range(1, 5):
        grouped_df[f'{n}_best_idotp'] = df.groupby(['name', 'sequence']).apply(
            lambda group: nth_best_value(group, 'idotp', n, ascending=False)
        ).reset_index(level=[0, 1], drop=True)
        grouped_df[f'{n}_best_abs_ppm'] = df.groupby(['name', 'sequence']).apply(
            lambda group: nth_best_value(group, 'abs_ppm', n, ascending=True)
        ).reset_index(level=[0, 1], drop=True)

    # Assign TruePositive column
    grouped_df["TruePositive"] = grouped_df["name"].apply(lambda x: 0 if 'decoy' in x else 1)


    return grouped_df.dropna().reset_index(drop=True)

def calculate_q_values(target_dist, decoy_dist, score):
    """Calculate q-value for a given score."""
    n_target = np.sum(target_dist >= score)
    n_decoy = np.sum(decoy_dist >= score)
    return n_decoy / n_target


def prob_threshold(target_dist, decoy_dist, fdr=0.05):
    """Determine the q-value threshold."""
    all_dist = np.concatenate([target_dist, decoy_dist])
    thresholds = np.percentile(all_dist, np.linspace(0, 100, 200))

    for threshold in thresholds:
        n_target = np.sum(target_dist > threshold)
        n_decoy = np.sum(decoy_dist > threshold)
#        print(threshold, n_decoy, n_target, round(n_decoy/n_target, 3))
        if n_target == 0:
            continue
        if n_decoy / n_target <= fdr:
#            print(n_decoy, n_target, fdr)
            return threshold
    return thresholds[-1]


def train_pipeline(df,
                  fdr=0.025,
                  decoy_level=2,
                  return_dataframes=False,
                  model_dir=None):

    """
    Main pipeline to preprocess, train models, and calculate q-values.
    Decoys should initially be provided at 2x the number of true positives for proper FDR computation.
    """
    if decoy_level != 2:
        print("Warning: Run the pipeline with twice as many decoys as targets for proper q-value computation.")

    # Preprocess the dataframe
    processed_df = preprocess_dataframe(df)

    # Train the RT model and calculate residuals
    scaler, rt_model = train_aa_rt_model(processed_df[processed_df["TruePositive"] == 1].reset_index(drop=True))
    processed_df["RT_residual"] = calculate_rt_residuals(processed_df, scaler, rt_model)


    # Split data into true positives and decoys
    df_tp = processed_df[processed_df["TruePositive"] == 1]
    df_decoy = processed_df[processed_df["TruePositive"] == 0].sample(frac=1).reset_index(drop=True)

    # Split decoys into train/test
    half_index = len(df_decoy) // 2
    df_decoy_train = df_decoy.iloc[:half_index]
    df_decoy_test = df_decoy.iloc[half_index:]

    # Combine training data
    df_train = pd.concat([df_tp, df_decoy_train]).reset_index(drop=True)

    features = [
        'mean_n_charges', 'min_abs_ppm', 'mean_abs_ppm', 'max_abs_ppm',
        'min_ab_cluster_total', 'mean_ab_cluster_total', 'max_ab_cluster_total', 'sum_ab_cluster_total',
        'min_idotp', 'mean_idotp', 'max_idotp', 'RT_residual', 'charge_3', 'charge_4', 'charge_5', 'charge_6',
        'charge_7', 'charge_8', 'charge_9', 'charge_10', 'charge_11', 'charge_12',
        '1_or_more_charge_states', '2_or_more_charge_states', '3_or_more_charge_states', '4_or_more_charge_states',
        '1_best_idotp', '2_best_idotp', '3_best_idotp', '4_best_idotp', '1_best_abs_ppm', '2_best_abs_ppm',
        '3_best_abs_ppm', '4_best_abs_ppm', 'fxn_zscore_idotp_ppm', 'fxn_zscore_ppm_idotp'
    ]

    features = [ f for f in features if f in df_train.keys() ]

    # Train logistic regression
    X_train = df_train[features]
    y_train = df_train["TruePositive"]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    log_reg = LogisticRegression(C=0.1)
    log_reg.fit(X_train_scaled, y_train)

    # Test on decoys and targets
    X_decoy_test_scaled = scaler.transform(df_decoy_test[features])
    X_target_scaled = scaler.transform(df_tp[features])
    decoy_scores = log_reg.predict_proba(X_decoy_test_scaled)[:, 1]
    target_scores = log_reg.predict_proba(X_target_scaled)[:, 1]

    # Determine q-value threshold
    prob_threshold_value = prob_threshold(target_scores, decoy_scores, fdr=fdr)


    # Compute probabilities for all entries
    X_all_scaled = scaler.transform(processed_df[features])
    processed_df["logreg_prob"] = log_reg.predict_proba(X_all_scaled)[:, 1]

    # Calculate q-values
    processed_df["q_value"] = processed_df.apply(
        lambda row: calculate_q_values(target_scores, decoy_scores, row["logreg_prob"]),
        # if row["TruePositive"] == 1 else np.nan,
        axis=1,
    )

    # Filter true positives based on q-value threshold
    filtered_tp = processed_df[
        (processed_df["TruePositive"] == 1) & (processed_df["logreg_prob"] >= prob_threshold_value)
    ]


    # Reporting
    total_tp = len(processed_df[processed_df["TruePositive"] == 1])
    remaining_tp = len(filtered_tp)
    dropped_tp = total_tp - remaining_tp
    print(f"Total True Positives: {total_tp}, Remaining: {remaining_tp}, Dropped: {dropped_tp} ({(dropped_tp / total_tp) * 100:.2f}%)")
    print(f"logreg_prob threshold: {prob_threshold_value:.4f}")

    # Save results
    if model_dir is not None:

        # Output scaler and model for future usage
        save_model_and_scaler(scaler, log_reg, model_dir)

        unfiltered_output = os.path.join(model_dir, "qvalue estimator_unfiltered.json")
        filtered_output = os.path.join(model_dir, "qvalue estimator_filtered.json")
        # Output unfiltered qvalue estimator dataframe
        processed_df.to_json(unfiltered_output)
        # Output filtered qvalue estimator dataframe
        filtered_tp.to_json(filtered_output)

        # Output score distributions plot if specified
        score_plot_output = os.path.join(model_dir, "qvalue_estimator.png")
        plot_score_distributions(target_scores, decoy_scores, prob_threshold_value, output_path=score_plot_output)

    # Return dataframes if requested
    if return_dataframes:
        return processed_df, filtered_tp


def save_model_and_scaler(scaler, model, output_dir):
    """
    Save the trained scaler and logistic regression model.

    Parameters:
    - scaler: Trained StandardScaler object.
    - model: Trained LogisticRegression object.
    - output_dir: Directory to save the model and scaler.
    """
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    model_path = os.path.join(output_dir, "log_reg_model.joblib")

    joblib.dump(scaler, scaler_path)
    joblib.dump(model, model_path)

    print(f"Scaler saved to {scaler_path}")
    print(f"Model saved to {model_path}")


def load_model_and_scaler(model_dir):
    """
    Load the saved scaler and logistic regression model.

    Parameters:
    - model_dir: Directory where the model and scaler are saved.

    Returns:
    - scaler: Loaded StandardScaler object.
    - model: Loaded LogisticRegression object.
    """
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    model_path = os.path.join(model_dir, "log_reg_model.joblib")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    print(f"Scaler loaded from {scaler_path}")
    print(f"Model loaded from {model_path}")

    return scaler, model


def apply_model_to_new_data(df,
                            model_dir,
                            output_df_path=None,
                            prob_threshold_value=0.803):
    """
    Apply the trained model and scaler to new data and classify based on the probability threshold.

    Parameters:
    - df: New dataframe to process and predict.
    - model_dir: Directory where the model and scaler are saved.
    - prob_threshold_value: logreg_prob threshold from the original dataset (q-value threshold).

    Returns:
    - processed_df: Processed dataframe with predictions and classifications.
    """
    # Preprocess the dataframe
    processed_df = preprocess_dataframe(df)

    # Train the RT model and calculate residuals
    rt_scaler, rt_model = train_aa_rt_model(processed_df[processed_df["TruePositive"] == 1].reset_index(drop=True))
    processed_df["RT_residual"] = calculate_rt_residuals(processed_df, rt_scaler, rt_model)

    # Load scaler and model for qvalue estimator
    scaler, model = load_model_and_scaler(model_dir)

    # Extract features
    features = [
        'mean_n_charges', 'min_abs_ppm', 'mean_abs_ppm', 'max_abs_ppm',
        'min_ab_cluster_total', 'mean_ab_cluster_total', 'max_ab_cluster_total', 'sum_ab_cluster_total',
        'min_idotp', 'mean_idotp', 'max_idotp', 'RT_residual', 'charge_3', 'charge_4', 'charge_5', 'charge_6',
        'charge_7', 'charge_8', 'charge_9', 'charge_10', 'charge_11', 'charge_12',
        '1_or_more_charge_states', '2_or_more_charge_states', '3_or_more_charge_states', '4_or_more_charge_states',
        '1_best_idotp', '2_best_idotp', '3_best_idotp', '4_best_idotp', '1_best_abs_ppm', '2_best_abs_ppm',
        '3_best_abs_ppm', '4_best_abs_ppm', 'fxn_zscore_idotp_ppm', 'fxn_zscore_ppm_idotp'
    ]

    features = [f for f in features if f in processed_df.columns]

    # Scale features
    X_new_scaled = scaler.transform(processed_df[features])

    # Predict probabilities
    predictions = model.predict_proba(X_new_scaled)[:, 1]

    # Add predictions to the dataframe
    processed_df["logreg_prob"] = predictions

    # Classify based on the threshold
    processed_df["classification"] = processed_df["logreg_prob"] >= prob_threshold_value

    if output_df_path is not None:
        processed_df.to_json(output_df_path)
    else:
        return processed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for RT model and q-value calculation.")
    parser.add_argument("--input", type=str, help="Input JSON file.")
    parser.add_argument("--fdr", type=float, default=0.025, help="FDR threshold")
    parser.add_argument("--model_dir", type=str, default='.', help="Output directory for qvalue estimator files")

    parser.add_argument("--mode", type=str, default="train", help="Output directory for qvalue estimator files")
    parser.add_argument("--prob_threshold_value", type=float, default=0.781, help="Probability threshold value for classification.")
    parser.add_argument("--output_file_path", type=str, default="checked_library_info.json", help="Output for updated library info file.")

    args = parser.parse_args()

    df = pd.read_json(args.input)

    if args.mode == "train":
        train_pipeline(df, fdr=args.fdr, model_dir=args.model_dir )
    elif args.mode == "predict":
        apply_model_to_new_data(df,
                                args.model_dir,
                                prob_threshold_value=args.prob_threshold_value)