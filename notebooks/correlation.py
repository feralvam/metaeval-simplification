from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr, t
from scipy.stats.mstats import mquantiles
from sklearn.metrics import cohen_kappa_score


def _standardise_ratings(df, rater_id_cols, aspect_col):
    return df.groupby(rater_id_cols)[aspect_col].transform(lambda x: zscore(x))


def _simulate_two_annotators(ratings, num_ratings_annotatorA=1):
    ratings_shuffled = np.random.permutation(ratings)
    ratingA = np.mean(ratings_shuffled[:num_ratings_annotatorA])
    ratingB = np.mean(ratings_shuffled[num_ratings_annotatorA:])
    return [ratingA, ratingB]


def compute_inter_annotator_agreement(df_ratings, segment_id_cols, rater_id_cols, aspects,
                                      n_bins=5, use_quantiles=True, n_simulations=1000):
    iaa_per_aspect = {}
    for aspect in aspects:
        if f"{aspect}_zscore" not in df_ratings.columns:
            df_ratings[f"{aspect}_zscore"] = _standardise_ratings(df_ratings, rater_id_cols, aspect)
        df_scores = df_ratings[segment_id_cols + [f'{aspect}_zscore']]
        # Bin the data in n_bins
        if use_quantiles:  # equally-distributed
            _, bins_ranges = pd.qcut(df_scores[f'{aspect}_zscore'], q=n_bins, retbins=True)
        else:  # equally-spaced
            _, bins_ranges = pd.cut(df_scores[f'{aspect}_zscore'], bins=n_bins, retbins=True)
        kappa_values = []
        for _ in tqdm(range(n_simulations)):
            ratings_simulation = df_scores.groupby(segment_id_cols)[f'{aspect}_zscore'].apply(_simulate_two_annotators).to_list()
            raterA, raterB = zip(*ratings_simulation)
            kappa_values.append(cohen_kappa_score(np.digitize(raterA, bins_ranges), np.digitize(raterB, bins_ranges), weights='quadratic'))
        iaa_per_aspect[aspect] = (np.mean(kappa_values), np.std(kappa_values))
    return iaa_per_aspect


def compute_segment_scores(df_ratings, segment_id_cols, rater_id_cols, aspects):
    scores_cols = []
    for aspect in aspects:
        df_ratings[f"{aspect}_zscore"] = _standardise_ratings(df_ratings, rater_id_cols, aspect)
        scores_cols += [aspect, f"{aspect}_zscore"]
    df_segment_scores = df_ratings.groupby(segment_id_cols)[scores_cols].agg([np.mean])
    df_segment_scores.columns = [a for a, _ in df_segment_scores.columns]

    return df_segment_scores


def _select_pairs_in_group(group, min_score_difference=25):
    data = []
    for (system_a, score_a, zscore_a), (system_b, score_b, zscore_b) in combinations(group.values, 2):
        # select the pair if its absolute difference in DA scores is greater than 25
        if abs(score_a - score_b) > min_score_difference:
            data.append([system_a, score_a, zscore_a, system_b, score_b, zscore_b])
    df_selected_pairs = pd.DataFrame(data,
                                     columns=['system_a', "score_a", "zscore_a", "system_b", "score_b", "zscore_b"])
    return df_selected_pairs


def select_segment_pairs(df_human_scores, aspect, sentence_id_cols, system_id_cols):
    df_scores = df_human_scores.reset_index()
    cols_of_interest = system_id_cols + [aspect, f"{aspect}_zscore"]
    selected_pairs = (df_scores.groupby(sentence_id_cols)[cols_of_interest].apply(_select_pairs_in_group)
                                .reset_index(level=1, drop=True)
                                .reset_index())
    return selected_pairs


def compute_relative_ranking_correlations(df_human_scores, df_metrics_scores, aspect,
                                          segment_id_cols, sentence_id_cols, system_id_cols,
                                          use_absolute_values=True, bootstrap_samples=1000):
    df_segment_pairs = select_segment_pairs(df_human_scores, aspect, sentence_id_cols, system_id_cols)

    df_all_scores = pd.merge(left=df_segment_pairs,
                             left_on=sentence_id_cols+['system_a'],
                             right=df_metrics_scores,
                             right_on=segment_id_cols)
    df_all_scores = pd.merge(left=df_all_scores,
                             left_on=sentence_id_cols+['system_b'],
                             right=df_metrics_scores,
                             right_on=segment_id_cols)

    metrics_names = [col for col in df_metrics_scores.columns if col not in segment_id_cols]

    # Compute the correlations
    print("Computing correlations...")
    correlations_data = []
    for metric in metrics_names:
        corr = kendall_tau_wmt(df_all_scores[['zscore_a', 'zscore_b', f"{metric}_x", f"{metric}_y"]])
        if use_absolute_values:
            corr = abs(corr)
        correlations_data.append([metric, corr])
    df_correlations = pd.DataFrame(correlations_data, columns=['metric', 'corr'])

    # Bootstrap sampling
    print("Bootstrap sampling...")
    correlations_bootstrap_data = []
    for _ in range(bootstrap_samples):
        df_scores_sample = df_all_scores.sample(n=len(df_all_scores), replace=True)
        for metric in metrics_names:
            corr_sample = kendall_tau_wmt(df_scores_sample[['zscore_a', 'zscore_b', f"{metric}_x", f"{metric}_y"]])
            if use_absolute_values:
                corr_sample = abs(corr_sample)
            correlations_bootstrap_data.append([metric, corr_sample])
    df_bootstrap_correlations = pd.DataFrame(correlations_bootstrap_data, columns=['metric', 'corr'])

    # Compute 95% confidence intervals for each metric
    print("Computing 95% confidence intervals for each metric...")
    confidence_intervals = []
    for metric in metrics_names:
        metric_corr = df_bootstrap_correlations[df_bootstrap_correlations['metric'] == metric]['corr']
        # Equivalent to using the R function quantile with default type 7
        lower, upper = mquantiles(metric_corr, prob=[0.05, 0.95], alphap=1, betap=1)
        confidence_intervals.append(pd.Interval(left=lower, right=upper, closed='both'))
    df_correlations['conf_interval'] = confidence_intervals
    df_correlations.sort_values(by=['corr'], ascending=False, inplace=True, ignore_index=True)

    # Determine if the difference in performance is significant
    print("Determining if the difference in performance is significant...")
    metrics_names = df_correlations['metric'].to_list()
    significance_matrix = []
    winner_status = []
    for _, row_metric_a in df_correlations.iterrows():
        metric_a = row_metric_a['metric']
        ci_metric_a = row_metric_a['conf_interval']
        is_winner = True
        significance_row = []
        for _, row_metric_b in df_correlations.iterrows():
            metric_b = row_metric_b['metric']
            ci_metric_b = row_metric_b['conf_interval']
            # It's significant if confidence intervals do not overlap
            is_diff_stats_significant = ci_metric_a.left > ci_metric_b.right
            significance_row.append(is_diff_stats_significant)
            # Update winner status (not significantly outperformed by any other metric)
            if metric_b != metric_a:
                is_winner = is_winner and is_diff_stats_significant
        significance_matrix.append(significance_row)
        winner_status.append(is_winner)
    df_correlations['is_winner'] = winner_status
    df_significance = pd.DataFrame(np.array(significance_matrix), columns=metrics_names, index=metrics_names)

    return df_correlations, df_significance


def kendall_tau_wmt(df_scores):
    concordant = 0
    discordant = 0
    for _, (score_a, score_b, metric_a, metric_b) in df_scores.iterrows():
        if score_a < score_b:
            if metric_a < metric_b:
                concordant += 1
            else:
                discordant += 1
        elif score_a > score_b:
            if metric_a <= metric_b:
                discordant += 1
            else:
                concordant += 1

    return (abs(concordant) - abs(discordant)) / (abs(concordant) + abs(discordant))


def compute_direct_assessment_correlations(df_human_scores, df_metrics_scores, aspect, segment_id_cols,
                                           use_absolute_values=True):
    df_da_scores = df_human_scores.reset_index()
    cols_of_interest = segment_id_cols + [aspect, f"{aspect}_zscore"]
    df_da_scores = df_da_scores[cols_of_interest]
    df_all_scores = pd.merge(left=df_metrics_scores, right=df_da_scores, on=segment_id_cols)

    # Compute correlations metrics vs human scores
    print("Computing correlations...")
    metrics_names = [col for col in df_metrics_scores.columns if col not in segment_id_cols]
    correlations_data = []
    for metric in metrics_names:
        corr, p_value = pearsonr(df_all_scores[metric], df_all_scores[f'{aspect}_zscore'])
        if use_absolute_values:
            corr = abs(corr)
        correlations_data.append([metric, corr, p_value])
    df_correlations_metrics_human = pd.DataFrame(correlations_data, columns=['metric', 'corr', 'p_value'])
    df_correlations_metrics_human.sort_values(by=['corr'], ascending=False, inplace=True, ignore_index=True)

    # Compute correlations metrics vs metrics
    metrics_names = df_correlations_metrics_human['metric'].to_list()
    correlations_data = []
    for _, (metric_a, corr_metric_a, _) in df_correlations_metrics_human.iterrows():
        for _, (metric_b, corr_metric_b, _) in df_correlations_metrics_human.iterrows():
            corr_a_b, pvalue_a_b = pearsonr(df_all_scores[metric_a], df_all_scores[metric_b])
            if use_absolute_values:
                corr_a_b = abs(corr_a_b)
            correlations_data.append([metric_a, corr_metric_a,
                                      metric_b, corr_metric_b,
                                      corr_a_b, pvalue_a_b])
    df_correlations_metric_metric = pd.DataFrame(correlations_data,
                                                 columns=['metric_a', 'corr_metric_a',
                                                          'metric_b', 'corr_metric_b',
                                                          'corr_a_b', 'pvalue_a_b'])

    # Determine if the difference in performance is significant
    print("Determining if the difference in performance is significant...")
    significance_matrix = []
    winner_status = []
    for metric_a in metrics_names:
        df_correlations = df_correlations_metric_metric[df_correlations_metric_metric['metric_a'] == metric_a]
        is_winner = True
        significance_row = []
        for _, (_, corr_metric_a, metric_b, corr_metric_b, corr_a_b, _) in df_correlations.iterrows():
            p = np.nan
            if (metric_a != metric_b) and (corr_metric_a > corr_metric_b):
                _, p = williams_test(corr_metric_a, corr_metric_b, corr_a_b, len(df_human_scores))
            is_diff_stats_significant = p < 0.05
            if not is_diff_stats_significant:
                # we do not care about the exact values in cases where it's not significant
                p = np.nan
            significance_row.append(p)
            # Update winner status (not significantly outperformed by any other metric)
            if metric_a != metric_b:
                is_winner = is_winner and is_diff_stats_significant
        significance_matrix.append(significance_row)
        winner_status.append(is_winner)
    df_correlations_metrics_human['is_winner'] = winner_status
    df_significance = pd.DataFrame(np.array(significance_matrix), columns=metrics_names, index=metrics_names)

    return df_correlations_metrics_human, df_significance


# From https://github.com/inmoonlight/nlp-williams/blob/master/williams.py
def williams_test(r12, r13, r23, n):
    """The Williams test (Evan J. Williams. 1959. Regression Analysis, volume 14. Wiley, New York, USA)

    A test of whether the population correlation r12 equals the population correlation r13.
    Significant: p < 0.05

    Arguments:
        r12 (float): correlation between x1, x2
        r13 (float): correlation between x1, x3
        r23 (float): correlation between x2, x3
        n (int): size of the population

    Returns:
        t (float): Williams test result
        p (float): p-value of t-dist
    """
    assert (r12 >= r13), "r12 should be larger than r13"
    assert (n > 3), "n should be larger than 3"

    K = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    denominator = np.sqrt(
        2 * K * (n - 1) / (n - 3) + (((r12 + r13) ** 2) / 4) * ((1 - r23) ** 3)
    )
    numerator = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    p = 1 - t.cdf(numerator / denominator, df=n - 3)  # changed to n-3 on 30/11/14
    return t, p
