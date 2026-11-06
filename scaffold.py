"""
Recommender Systems from Scratch: From Funk SVD to NeuMF and a Feedback Loop scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Recommender systems from scratch.

Story (chapter 22 of Murphy, Probabilistic Machine Learning): a recommender only ever sees the ratings its own past
choices produced. This script builds a planted world shaped like MovieLens-1M at the book's 4.47 percent density, where
the true expected rating of every user-item pair is known, then walks the chapter: baselines and neighbours, the zero-filled
SVD of Figure 22.3, Funk SGD and ALS, AutoRec, implicit feedback with BPR, factorization machines and cold start, NeuMF,
and finally a closed feedback loop where exploration is put to the test.
"""
import numpy as np


def main() -> None:
    rep = recsys_report(seed=0)
    s = rep["stats"]

    print("1. The world against the book's MovieLens-1M (section 22.1.3.3)")
    print("   Book: 6040 users, 3706 movies, 1,000,209 ratings, density 4.47%, about 166 ratings per user.")
    print(f"   Here: 1500 users, 1000 items, {s['n_ratings']:,} ratings, density {100 * s['density']:.2f}%, about {s['n_ratings'] / 1500:.0f} ratings per user.")
    print(f"   Ratings: mean {s['mean']:.2f}, std {s['std']:.2f} (MovieLens-1M: 3.58, 1.12); counts for 1..5 = {s['hist']}.")
    print(f"   Popularity: top 10% of items hold {100 * s['top10pct_share']:.0f}% of ratings, item Gini {s['item_gini']:.2f}.")

    print("\n2. Rating prediction: RMSE on held-out logged ratings (lower is better)")
    names = [("oracle", "true expected rating (no model can beat this on average)"), ("als", "ALS, eq 22.4 with PMF priors"),
             ("autorec_user", "AutoRec, user-based"), ("autorec_item", "AutoRec, item-based"), ("funk_svd", "Funk SVD, eq 22.6-22.9"),
             ("knn_item", "item kNN on residuals"), ("knn_user", "user kNN on residuals"), ("residual_svd", "SVD of residuals from the biases"),
             ("biases", "user and item biases"), ("knn_raw_user", "user kNN on raw ratings, eq 22.1"), ("global_mean", "global mean"),
             ("zero_filled_svd", "zero-filled SVD with K = 50, the book's demo")]
    for key, label in names:
        print(f"   {rep['ladder'][key]:.4f}  {label}")
    print("   Published on MovieLens-1M, for scale (Sedhain et al. 2015; not from this book, which gives no RMSE):")
    print("   global mean 1.12, biases about 0.91, biased MF 0.845, U-AutoRec 0.874, I-AutoRec 0.831.")
    print("   The bottom of the ladder has the same shape. The top differs: this world is bilinear, so ALS beats AutoRec,")
    print("   and user-based AutoRec beats item-based, the reverse of what the book reports for MovieLens and Netflix.")
    sweep = {lf: round(v, 3) for lf, v in rep["lambda_sweep"].items()}
    implied = rep["implied_lambdas"]
    print(f"   ALS validation RMSE by lam_f: {sweep}. The priors imply lam_f from {min(implied['lam_factor']):.1f} to {max(implied['lam_factor']):.1f} (eq 22.10).")
    print(f"   ALS vs planted item factors (cosines of principal angles): {rep['recovery']['als']}")
    print(f"   Funk SVD:                                                  {rep['recovery']['funk_svd']} after {rep['funk_epochs_used']} epochs")

    print("\n3. The price of assuming data is missing at random (section 22.1)")
    print("   beta = how strongly logged pairs favour what a user already likes. Same ALS on each log.")
    print("   beta   logged mean   biases offline   ALS offline   ALS truth over ALL pairs   predicted vs true mean")
    for bt, d in rep["selection"].items():
        print(f"   {bt:4.1f}   {d['obs_mean']:11.2f}   {d['biases_offline']:14.3f}   {d['offline_rmse']:11.3f}   {d['truth_rmse']:24.3f}   {d['pred_mean']:.2f} vs {d['true_mean']:.2f}")
    print("   Offline error holds steady or improves as selection grows, while error against the truth keeps getting worse.")

    print("\n4. Implicit feedback (section 22.2): rank all unseen items, leave-one-out, k = 10")
    print("   model            recall   ndcg     auc      true value of the top 10")
    for name, m in rep["implicit"].items():
        print(f"   {name:15s}  {m['recall']:.3f}    {m['ndcg']:.3f}    {m['auc']:.3f}    {m['true_value']:.2f}")
    print("   Recall rewards predicting what gets logged, including popularity. True value asks if the user would like it.")

    print("\n5. Factorization machines and cold start (sections 22.2.2 and 22.3)")
    c = rep["cold_start"]
    print(f"   Items never rated: global mean {c['cold_global_mean']:.3f}, user bias only {c['cold_user_bias']:.3f}, FM with tags {c['cold_fm_tags']:.3f}, oracle {c['cold_oracle']:.3f}")
    print(f"   Warm items: biases {c['warm_biases']:.3f}, FM with ids only {c['warm_fm_ids']:.3f}, FM with tags {c['warm_fm_tags']:.3f}")

    print("\n6. Neural matrix factorization at one budget (section 22.2.3)")
    for name, m in rep["neumf"].items():
        print(f"   {name:6s} recall {m['recall']:.3f}  ndcg {m['ndcg']:.3f}  auc {m['auc']:.3f}  true value {m['true_value']:.2f}")
    print(f"   numpy BPR at k = 5, for reference: recall {rep['implicit']['bpr']['recall']:.3f}  ndcg {rep['implicit']['bpr']['ndcg']:.3f}")

    print("\n7. The feedback loop (section 22.4): 30 rounds, one item per user per round, retraining every round")
    print("   policy         first 6 rounds  last 6 rounds  mean reward  mean regret  coverage  exposure Gini  truth RMSE")
    for name, m in rep["loop"].items():
        print(f"   {name:13s}  {m['first_fifth']:14.3f}  {m['last_fifth']:13.3f}  {m['mean_reward']:11.3f}  {m['mean_regret']:11.3f}  {m['coverage']:8.3f}  {m['exposure_gini']:13.3f}  {m['truth_rmse']:10.3f}")
    print("   Rewards are true expected ratings. Greedy shows few items, and its model stays wrong about the rest of the catalog.")


if __name__ == "__main__":
    main()

