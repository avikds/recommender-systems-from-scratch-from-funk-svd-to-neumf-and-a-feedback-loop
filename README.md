# Recommender Systems from Scratch: From Funk SVD to NeuMF and a Feedback Loop

A recommender only ever sees the ratings its own past choices produced. This project works through Murphy's Probabilistic Machine Learning on a seeded world shaped like MovieLens-1M: planted low-rank tastes, user and item biases, popularity-skewed and taste-driven logging, 1-to-5 ratings at the book's 4.47 percent density (1500 users, 1000 items, 67,050 ratings, mean about 3.55, standard deviation about 1.08). MovieLens is not used, but this world gives you what MovieLens cannot: the true expected rating of every pair. You write regularized biases, user and item neighbours, the zero-filled SVD, Funk's SGD updates, ALS with PMF priors, item and user AutoRec with hand-derived gradients, BPR and hinge ranking, factorization machines with the O(KD) trick and a cold-start split, NeuMF in tinygrad, and a closed feedback loop with greedy, epsilon-greedy and Thompson-style exploration. Headline numbers, mean of five worlds. Held-out RMSE: zero-filled SVD 2.66, global mean 1.08, biases 0.866, neighbours 0.856, item AutoRec 0.847, user AutoRec 0.838, Funk SVD 0.838, ALS 0.820, noise floor 0.722. The book reports no RMSE for its MovieLens demo; published MovieLens-1M figures are 1.12 for the global mean, about 0.91 for biases and 0.845 for biased MF, and there item AutoRec wins at 0.831 where here user-based wins on all five worlds. As logging selection grows, ALS offline error stays near 0.81 while its error against the truth over all pairs goes from 0.54 to 0.90. BPR reaches recall at 10 of 0.196 against 0.141 for popularity, but its top items have a true rating of 3.78, against 4.43 for a rating model whose recall is only 0.063. Tags take never-rated items from 0.987 to 0.926. NeuMF at recall 0.190 ties GMF at 0.189, and the MLP pathway alone lags at 0.156. In the loop, Thompson-style exploration earns less than greedy over the first six rounds (3.60 against 3.75), then more over the last six (4.21 against 3.98), shows 96 percent of the catalog against 10 percent, and ends with truth RMSE 0.68 against 1.05.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_world
- [x] **2.** sample_ratings
- [x] **3.** split_per_user
- [x] **4.** dataset_stats
- [x] **5.** fit_biases
- [x] **6.** similarity_matrix
- [x] **7.** knn_predict
- [x] **8.** svd_complete
- [x] **9.** funk_sgd_step
- [x] **10.** fit_funk_svd
- [x] **11.** als_biased
- [x] **12.** pmf_objective
- [x] **13.** subspace_cosines
- [x] **14.** offline_vs_truth
- [x] **15.** autorec_loss_grads
- [x] **16.** fit_autorec
- [x] **17.** leave_one_out
- [x] **18.** rank_metrics
- [x] **19.** sample_triples

---

Built on Deep-ML.
