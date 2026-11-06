"""
Recommender Systems from Scratch: From Funk SVD to NeuMF and a Feedback Loop

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_world
import numpy as np

def make_world(n_users, n_items, k, seed, n_tags=24):
    rng = np.random.default_rng(seed)

    # 1. Generate factor strengths and rescale their sum to 0.35
    strength = 0.7 ** np.arange(k)
    strength = strength * (0.35 / strength.sum())

    # 2. Generate user and item latent factor matrices
    U = rng.normal(size=(n_users, k)) * strength ** 0.25
    V = rng.normal(size=(n_items, k)) * strength ** 0.25

    # 3. Generate user and item biases
    b = rng.normal(0.0, 0.42, size=n_users)
    c = rng.normal(0.0, 0.55, size=n_items)

    # 4. Generate item popularity
    log_pop = 1.25 * rng.normal(size=n_items) + c

    # 5. Generate user activity levels
    log_act = 0.9 * rng.normal(size=n_users)

    # 6. Generate item tags using side information
    F = np.hstack([
        c[:, None] / 0.55,
        V / strength ** 0.25
    ])

    G = rng.normal(size=(n_tags, k + 1))

    tag_score = (
        F @ G.T / np.sqrt(k + 1)
        + rng.normal(0.0, 0.5, size=(n_items, n_tags))
    )

    tags = (tag_score > 0.8).astype(float)

    # Return the complete synthetic world
    return {
        "mu": 2.85,
        "b": b,
        "c": c,
        "U": U,
        "V": V,
        "log_pop": log_pop,
        "log_act": log_act,
        "tags": tags,
        "strength": strength
    }


def expected_ratings(world):
    # Calculate the true expected rating for every user-item pair
    return (
        world["mu"]
        + world["b"][:, None]
        + world["c"][None, :]
        + world["U"] @ world["V"].T
    )

# Step 2 - sample_ratings
def exposure_logits(world, beta):
    # Compute the user-item interaction matrix
    A = world["U"] @ world["V"].T

    # Calculate exposure logits using user activity,
    # item popularity, and user-item preferences
    return (
        world["log_act"][:, None]
        + world["log_pop"][None, :]
        + beta * A / A.std()
    )


def sample_ratings(world, n_obs, beta, noise, seed):
    # Create an independent random number generator
    rng = np.random.default_rng([seed, 0])

    n_users = world["U"].shape[0]
    n_items = world["V"].shape[0]

    # Generate Gumbel keys for every user-item pair
    keys = (
        exposure_logits(world, beta)
        + rng.gumbel(size=(n_users, n_items))
    )

    # Select the n_obs pairs with the largest keys
    flat = np.argsort(-keys, axis=None)[:n_obs]

    # Sort selected pairs by their flat indices
    # and convert them back into user and item indices
    users, items = np.unravel_index(
        np.sort(flat),
        keys.shape
    )

    # Generate noisy ratings using the true expected ratings
    y = (
        expected_ratings(world)[users, items]
        + rng.normal(0.0, noise, size=n_obs)
    )

    # Round ratings and clip them to the range [1, 5]
    ratings = np.clip(np.round(y), 1.0, 5.0)

    return users, items, ratings

# Step 3 - split_per_user
def split_per_user(users, items, ratings, test_frac, seed):
    # Create a reproducible random number generator
    rng = np.random.default_rng(seed)

    n = len(users)

    # Generate a random permutation of the input row indices
    perm = rng.permutation(n)

    # Group rows by user while preserving their permutation order
    grouped = perm[
        np.argsort(users[perm], kind="stable")
    ]

    # Identify the boundaries of each user's group
    grouped_users = users[grouped]

    boundaries = np.concatenate((
        [0],
        np.flatnonzero(grouped_users[1:] != grouped_users[:-1]) + 1,
        [n]
    ))

    # Mark the rows assigned to the test set
    test_mask = np.zeros(n, dtype=bool)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Get the permuted rows belonging to the current user
        group = grouped[start:end]

        # Determine how many ratings this user contributes to the test set
        n_test = int(test_frac * len(group))

        # Assign the first n_test rows to the test set
        test_mask[group[:n_test]] = True

    # Recover original input order for both training and testing indices
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    # Return training and testing data, preserving original row order
    return (
        (users[train_idx], items[train_idx], ratings[train_idx]),
        (users[test_idx], items[test_idx], ratings[test_idx])
    )

# Step 4 - dataset_stats
def gini(counts):
    # Sort counts in ascending order
    x = np.sort(np.asarray(counts, dtype=float))

    n = len(x)
    total = np.sum(x)

    # Handle empty or all-zero counts
    if n == 0 or total == 0:
        return 0.0

    # Calculate the Gini coefficient
    indices = np.arange(1, n + 1)

    gini_value = np.sum((2 * indices - n - 1) * x) / (n * total)

    return float(gini_value)


def dataset_stats(users, items, ratings, n_users, n_items):
    # Convert inputs to NumPy arrays
    users = np.asarray(users)
    items = np.asarray(items)
    ratings = np.asarray(ratings)

    # Total number of observed ratings
    n_ratings = len(ratings)

    # Calculate rating density
    density = (
        n_ratings / (n_users * n_items)
        if n_users * n_items > 0 else 0.0
    )

    # Calculate mean and population standard deviation
    mean = float(np.mean(ratings)) if n_ratings > 0 else 0.0
    std = float(np.std(ratings)) if n_ratings > 0 else 0.0

    # Calculate histogram counts for ratings 1 through 5
    hist = [
        int(np.count_nonzero(ratings == rating))
        for rating in range(1, 6)
    ]

    # Calculate the number of ratings received by each item
    item_counts = np.bincount(
        items.astype(int),
        minlength=n_items
    )[:n_items]

    # Calculate the number of ratings submitted by each user
    user_counts = np.bincount(
        users.astype(int),
        minlength=n_users
    )[:n_users]

    # Identify the most-rated max(1, n_items // 10) items
    top_n = max(1, n_items // 10)

    # Calculate the proportion of ratings belonging to those items
    top10pct_share = (
        float(np.sort(item_counts)[-top_n:].sum() / n_ratings)
        if n_ratings > 0 else 0.0
    )

    # Calculate the Gini coefficient of item rating counts
    item_gini = gini(item_counts)

    # Calculate the median number of ratings per user
    user_median = (
        float(np.median(user_counts))
        if n_users > 0 else 0.0
    )

    # Count items that have received no ratings
    items_unrated = int(np.count_nonzero(item_counts == 0))

    # Return all dataset statistics
    return {
        "n_ratings": int(n_ratings),
        "density": round(float(density), 4),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "hist": hist,
        "top10pct_share": round(top10pct_share, 3),
        "item_gini": round(float(item_gini), 3),
        "user_median": user_median,
        "items_unrated": items_unrated
    }

# Step 5 - fit_biases
def rmse(y, pred):
    # Convert inputs to NumPy arrays
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # Clip predictions to the valid rating range
    pred = np.clip(pred, 1.0, 5.0)

    # Calculate and return root mean squared error
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def fit_biases(users, items, ratings, n_users, n_items, lam, sweeps):
    # Convert inputs to NumPy arrays
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    ratings = np.asarray(ratings, dtype=float)

    # Ensure regularization is positive
    if lam <= 0:
        raise ValueError("lam must be positive")

    # Initialize global mean and item biases
    mu = float(ratings.mean())
    b = np.zeros(n_users)
    c = np.zeros(n_items)

    # Precompute the number of ratings per user and item
    user_counts = np.bincount(users, minlength=n_users)
    item_counts = np.bincount(items, minlength=n_items)

    # Alternating optimization using closed-form updates
    for _ in range(sweeps):

        # Update every user bias using the current item biases
        user_residuals = ratings - mu - c[items]

        user_sums = np.bincount(
            users,
            weights=user_residuals,
            minlength=n_users
        )

        b = user_sums / (lam + user_counts)

        # Update every item bias using the newly updated user biases
        item_residuals = ratings - mu - b[users]

        item_sums = np.bincount(
            items,
            weights=item_residuals,
            minlength=n_items
        )

        c = item_sums / (lam + item_counts)

    # Return global mean, user biases, and item biases
    return mu, b, c

# Step 6 - similarity_matrix
def to_matrix(users, items, values, n_users, n_items):
    # Initialize the ratings matrix with zeros
    R = np.zeros((n_users, n_items), dtype=float)

    # Initialize the Boolean observation mask
    mask = np.zeros((n_users, n_items), dtype=bool)

    # Fill observed user-item entries
    # Values can be either an array or a single number
    R[users, items] = values

    # Mark observed entries as True
    mask[users, items] = True

    return R, mask


def similarity_matrix(E, mask, shrink):
    # Convert inputs to NumPy arrays
    E = np.asarray(E, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    # Convert the Boolean mask into a numerical matrix
    M = mask.astype(float)

    # Keep only observed entries in the centered matrix
    X = E * M

    # Calculate the numerator using matrix multiplication
    # Each entry contains the dot product over co-observed columns
    numerator = X @ X.T

    # Calculate the number of shared observed columns
    overlap = M @ M.T

    # Calculate squared norms over the shared observed columns
    left_norm_sq = (E ** 2 * M) @ M.T

    # The corresponding right squared norms are the transpose
    right_norm_sq = left_norm_sq.T

    # Calculate the cosine similarity denominator
    denominator = np.sqrt(
        np.maximum(left_norm_sq * right_norm_sq, 1e-12)
    )

    # Calculate overlap-only cosine similarity
    similarity = numerator / denominator

    # Apply shrinkage based on the number of shared observations
    similarity *= overlap / (overlap + shrink + 1e-12)

    # Prevent a row from being its own neighbour
    np.fill_diagonal(similarity, 0.0)

    return similarity

# Step 7 - knn_predict
def knn_predict(S, E, mask, rows, cols, k, fallback=0.0):
    # Convert inputs to NumPy arrays
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    # Initialize predictions with the fallback value
    predictions = np.full(len(rows), fallback, dtype=float)

    # Process each query (row, column) pair
    for idx, (r, c) in enumerate(zip(rows, cols)):

        # Find candidate neighbours that:
        # 1. Have an observed value in the target column
        # 2. Have positive similarity with the query row
        candidates = np.flatnonzero(
            mask[:, c] & (S[r, :] > 0)
        )

        # Skip if there are no candidate neighbours or k is non-positive
        if len(candidates) == 0 or k <= 0:
            continue

        # Sort candidates by descending similarity.
        # Stable sorting preserves lower row indices when similarities tie.
        order = np.argsort(
            -S[r, candidates],
            kind="stable"
        )

        # Keep only the k most similar neighbours
        neighbours = candidates[order[:k]]

        # Retrieve their similarity weights
        weights = S[r, neighbours]

        # Calculate the similarity-weighted average
        predictions[idx] = np.sum(
            weights * E[neighbours, c]
        ) / np.sum(weights)

    # Return predictions as a floating-point NumPy array
    return predictions

# Step 8 - svd_complete
def svd_complete(R, k):
    # Convert the input matrix to a floating-point NumPy array
    R = np.asarray(R, dtype=float)

    # Compute the thin Singular Value Decomposition
    U, s, Vt = np.linalg.svd(R, full_matrices=False)

    # Retain the k largest singular values and corresponding vectors
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct the rank-k approximation
    R_k = (U_k * s_k) @ Vt_k

    return R_k

# Step 9 - funk_sgd_step
def init_params(ratings, n_users, n_items, k, seed):
    # Create an independent random number generator
    rng = np.random.default_rng([seed, 1])

    # Initialize the global mean rating
    mu = float(ratings.mean())

    # Initialize user and item biases to zero
    b = np.zeros(n_users)
    c = np.zeros(n_items)

    # Initialize user and item latent factor matrices
    U = rng.normal(0.0, 0.1, size=(n_users, k))
    V = rng.normal(0.0, 0.1, size=(n_items, k))

    return {
        "mu": mu,
        "b": b,
        "c": c,
        "U": U,
        "V": V
    }


def mf_predict(params, users, items):
    # Convert user and item indices to NumPy arrays
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)

    # Predict ratings using the biased matrix factorization model
    predictions = (
        params["mu"]
        + params["b"][users]
        + params["c"][items]
        + np.sum(
            params["U"][users] * params["V"][items],
            axis=-1
        )
    )

    return predictions


def funk_sgd_step(params, u, i, y, lr, lam):
    # Calculate prediction error using the current parameters
    e_ui = float(y - mf_predict(params, u, i))

    # Save old bias values
    old_b = params["b"][u]
    old_c = params["c"][i]

    # Save copies of old latent factor vectors
    old_u = params["U"][u].copy()
    old_v = params["V"][i].copy()

    # Update user bias using its old value
    params["b"][u] = (
        old_b + lr * (e_ui - lam * old_b)
    )

    # Update item bias using its old value
    params["c"][i] = (
        old_c + lr * (e_ui - lam * old_c)
    )

    # Update user latent factors using the old item factors
    params["U"][u] = (
        old_u + lr * (e_ui * old_v - lam * old_u)
    )

    # Update item latent factors using the old user factors
    params["V"][i] = (
        old_v + lr * (e_ui * old_u - lam * old_v)
    )

    # Return the prediction error calculated before the updates
    return e_ui

# Step 10 - fit_funk_svd
def fit_funk_svd(users, items, ratings, n_users, n_items, k, lr, lam, epochs, seed, valid=None, patience=12):

    # Initialize model parameters
    params = init_params(ratings, n_users, n_items, k, seed)

    # Create an independent random number generator for epoch shuffling
    rng = np.random.default_rng([seed, 2])

    # Initialize validation history and early stopping variables
    history = []
    best_params = None
    best_rmse = np.inf
    patience_counter = 0

    # Convert training data into NumPy arrays
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    ratings = np.asarray(ratings, dtype=float)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):

        # Randomly shuffle the training observations
        permutation = rng.permutation(len(ratings))

        # Perform one SGD update for every observed rating
        for idx in permutation:
            funk_sgd_step(
                params,
                users[idx],
                items[idx],
                ratings[idx],
                lr,
                lam
            )

        # Perform validation if a validation set is provided
        if valid is not None:
            users_v, items_v, ratings_v = valid

            # Predict ratings on the validation set
            predictions = mf_predict(
                params,
                users_v,
                items_v
            )

            # Calculate validation RMSE
            current_rmse = rmse(ratings_v, predictions)

            # Append the current validation RMSE to history
            history.append(float(current_rmse))

            # Check whether the validation RMSE has improved
            if current_rmse < best_rmse:

                # Save the new best validation RMSE
                best_rmse = current_rmse

                # Save an independent copy of the best parameters
                best_params = {
                    key: value.copy()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in params.items()
                }

                # Reset the early stopping counter
                patience_counter = 0

            else:
                # Increment the counter when validation does not improve
                patience_counter += 1

                # Stop training when patience is exhausted
                if patience_counter >= patience:
                    break

    # If validation was not provided, return the final trained parameters
    if valid is None:
        best_params = params

    return best_params, history

# Step 11 - als_biased
def ridge_side(T, m, X, lam_vec):
    # Convert inputs to floating-point NumPy arrays
    T = np.asarray(T, dtype=float)
    m = np.asarray(m, dtype=float)
    X = np.asarray(X, dtype=float)
    lam_vec = np.asarray(lam_vec, dtype=float)

    # Number of rows and feature dimensions
    n_rows, n_cols = T.shape
    d = X.shape[1]

    # Construct the ridge regression matrices for all rows simultaneously
    # A[r] = X.T @ diag(m[r]) @ X + diag(lam_vec)
    A = np.einsum(
        'rj,jd,je->rde',
        m, X, X
    )

    # Add regularization to the diagonal of every A matrix
    A += np.diag(lam_vec)[None, :, :]

    # Construct the right-hand side for every row
    # rhs[r] = sum_j m[r,j] * T[r,j] * X[j]
    rhs = np.einsum(
        'rj,rj,jd->rd',
        m, T, X
    )

    # Solve all ridge regression systems simultaneously
    theta = np.linalg.solve(
        A,
        rhs[:, :, None]
    )[:, :, 0]

    return theta, A


def als_sweep(T, m, U, V, lam_vec):
    # Convert inputs to NumPy arrays
    T = np.asarray(T, dtype=float)
    m = np.asarray(m, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    lam_vec = np.asarray(lam_vec, dtype=float)

    # --------------------------------------------------
    # 1. Update user parameters while keeping item
    #    parameters fixed
    # --------------------------------------------------

    # Construct item feature vectors with a bias column
    X_users = np.column_stack((
        np.ones(V.shape[0]),
        V[:, 1:]
    ))

    # Remove the current item bias from the targets
    targets_users = T - V[:, 0][None, :] * m

    # Solve the user-side ridge regression problems
    U_new, A_u = ridge_side(
        targets_users,
        m,
        X_users,
        lam_vec
    )

    # --------------------------------------------------
    # 2. Update item parameters using the newly updated
    #    user parameters
    # --------------------------------------------------

    # Construct user feature vectors with a bias column
    X_items = np.column_stack((
        np.ones(U_new.shape[0]),
        U_new[:, 1:]
    ))

    # Remove the newly updated user bias from the targets
    targets_items = (
        T.T - U_new[:, 0][None, :] * m.T
    )

    # Solve the item-side ridge regression problems
    V_new, A_v = ridge_side(
        targets_items,
        m.T,
        X_items,
        lam_vec
    )

    return U_new, V_new, A_u, A_v


def als_biased(users, items, ratings, n_users, n_items,
               k, lam_b, lam_f, sweeps, seed):

    # Initialize model parameters using the existing function
    params = init_params(
        ratings,
        n_users,
        n_items,
        k,
        seed
    )

    # Construct the user-item rating matrix and observation mask
    R, mask = to_matrix(
        users,
        items,
        ratings,
        n_users,
        n_items
    )

    # Convert the observation mask to floating-point form
    m = mask.astype(float)

    # Subtract the global mean from observed ratings
    # Missing entries remain zero
    T = R - params["mu"] * m

    # Combine bias and latent factors into augmented matrices
    # First column represents the corresponding bias
    U = np.column_stack((
        params["b"],
        params["U"]
    ))

    V = np.column_stack((
        params["c"],
        params["V"]
    ))

    # Set regularization:
    # lam_b for biases and lam_f for latent factors
    lam_vec = np.r_[
        lam_b,
        np.full(k, lam_f)
    ]

    # Perform alternating least squares sweeps
    for _ in range(sweeps):
        U, V, A_u, A_v = als_sweep(
            T,
            m,
            U,
            V,
            lam_vec
        )

    # Update the parameter dictionary
    params["b"] = U[:, 0].copy()
    params["U"] = U[:, 1:].copy()

    params["c"] = V[:, 0].copy()
    params["V"] = V[:, 1:].copy()

    # Return parameters in the same format as init_params
    return params

# Step 12 - pmf_objective
def pmf_objective(params, users, items, ratings, lam_b, lam_f):
    # Convert inputs to NumPy arrays
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    ratings = np.asarray(ratings, dtype=float)

    # Predict ratings using the existing matrix factorization function
    predictions = mf_predict(params, users, items)

    # Calculate the sum of squared prediction errors
    squared_error = np.sum((ratings - predictions) ** 2)

    # Calculate the regularization penalty for user and item biases
    bias_penalty = lam_b * (
        np.sum(params["b"] ** 2)
        + np.sum(params["c"] ** 2)
    )

    # Calculate the regularization penalty for latent factors
    factor_penalty = lam_f * (
        np.sum(params["U"] ** 2)
        + np.sum(params["V"] ** 2)
    )

    # Calculate the complete PMF objective
    objective = squared_error + bias_penalty + factor_penalty

    return float(objective)


def implied_lambdas(world, noise):
    # Effective observation noise variance.
    # The additional 1/12 accounts for rounding ratings to integers.
    sigma2 = noise ** 2 + 1.0 / 12.0

    # Calculate the regularization coefficient for user biases
    lam_user_bias = sigma2 / np.var(world["b"])

    # Calculate the regularization coefficient for item biases
    lam_item_bias = sigma2 / np.var(world["c"])

    # Calculate one regularization coefficient per latent factor column
    lam_factor = sigma2 / np.var(world["U"], axis=0)

    # Return the implied regularization coefficients
    return {
        "lam_user_bias": float(lam_user_bias),
        "lam_item_bias": float(lam_item_bias),
        "lam_factor": [float(x) for x in lam_factor]
    }

# Step 13 - subspace_cosines
def subspace_cosines(A, B):
    # Convert inputs to floating-point NumPy arrays
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Center each matrix by subtracting its column means
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)

    # Compute orthonormal bases using reduced QR decomposition
    Qa, _ = np.linalg.qr(A, mode="reduced")
    Qb, _ = np.linalg.qr(B, mode="reduced")

    # Compute the cross-product of the orthonormal bases
    cross_product = Qa.T @ Qb

    # Calculate the singular values in descending order
    # These are the cosines of the principal angles
    cosines = np.linalg.svd(
        cross_product,
        compute_uv=False
    )

    return cosines

# Step 14 - offline_vs_truth
def offline_vs_truth(params, test, world):
    # Unpack the held-out test data
    users, items, ratings = test

    # Convert test data to NumPy arrays
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    ratings = np.asarray(ratings, dtype=float)

    # --------------------------------------------------
    # 1. Calculate offline RMSE on held-out observations
    # --------------------------------------------------

    # Predict ratings for the held-out user-item pairs
    test_predictions = mf_predict(params, users, items)

    # Calculate offline RMSE
    # rmse() automatically clips predictions to [1, 5]
    offline_rmse = rmse(ratings, test_predictions)

    # --------------------------------------------------
    # 2. Generate predictions for all user-item pairs
    # --------------------------------------------------

    # Calculate the full prediction matrix
    P = (
        params["mu"]
        + params["b"][:, None]
        + params["c"][None, :]
        + params["U"] @ params["V"].T
    )

    # Calculate the ground-truth expected rating matrix
    E = expected_ratings(world)

    # Clip both matrices to the valid rating range
    Pc = np.clip(P, 1.0, 5.0)
    Ec = np.clip(E, 1.0, 5.0)

    # --------------------------------------------------
    # 3. Calculate truth RMSE over all user-item pairs
    # --------------------------------------------------

    truth_rmse = float(
        np.sqrt(np.mean((Pc - Ec) ** 2))
    )

    # --------------------------------------------------
    # 4. Calculate prediction and ground-truth means
    # --------------------------------------------------

    pred_mean = float(np.mean(Pc))
    true_mean = float(np.mean(Ec))

    # Return all evaluation metrics as Python floats
    return {
        "offline_rmse": float(offline_rmse),
        "truth_rmse": truth_rmse,
        "pred_mean": pred_mean,
        "true_mean": true_mean
    }

# Step 15 - autorec_loss_grads
def sigmoid(a):
    # Clip values to prevent numerical overflow
    a = np.clip(a, -30.0, 30.0)

    # Apply the sigmoid activation function
    return 1.0 / (1.0 + np.exp(-a))


def autorec_forward(Y, V, mu, W, b):
    # Convert inputs to NumPy arrays
    Y = np.asarray(Y, dtype=float)

    # Compute the hidden layer representation
    # H = sigmoid(Y @ V.T + mu)
    H = sigmoid(Y @ V.T + mu)

    # Reconstruct the input rating vectors
    # output = H @ W + b
    output = H @ W + b

    return output, H


def autorec_loss_grads(Y, mask, V, mu, W, b, lam):
    # Convert inputs to floating-point NumPy arrays
    Y = np.asarray(Y, dtype=float)
    mask = np.asarray(mask, dtype=float)
    V = np.asarray(V, dtype=float)
    mu = np.asarray(mu, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    # --------------------------------------------------
    # 1. Forward pass
    # --------------------------------------------------

    output, H = autorec_forward(Y, V, mu, W, b)

    # Calculate reconstruction error only for observed entries
    err = (output - Y) * mask

    # --------------------------------------------------
    # 2. Calculate the masked reconstruction loss
    # --------------------------------------------------

    loss = (
        np.sum(err ** 2)
        + (lam / 2.0) * (
            np.sum(W ** 2) + np.sum(V ** 2)
        )
    )

    # --------------------------------------------------
    # 3. Calculate gradients with respect to output
    # --------------------------------------------------

    # Gradient of squared reconstruction error
    grad_output = 2.0 * err

    # --------------------------------------------------
    # 4. Calculate gradients for W and b
    # --------------------------------------------------

    # Gradient with respect to output weights
    grad_W = H.T @ grad_output + lam * W

    # Gradient with respect to output bias
    grad_b = np.sum(grad_output, axis=0)

    # --------------------------------------------------
    # 5. Backpropagate through the hidden layer
    # --------------------------------------------------

    # Gradient with respect to hidden activations
    grad_H = grad_output @ W.T

    # Derivative of the sigmoid activation
    grad_Z = grad_H * H * (1.0 - H)

    # --------------------------------------------------
    # 6. Calculate gradients for V and mu
    # --------------------------------------------------

    # Gradient with respect to encoder weights
    grad_V = grad_Z.T @ Y + lam * V

    # Gradient with respect to hidden layer bias
    grad_mu = np.sum(grad_Z, axis=0)

    # --------------------------------------------------
    # 7. Return loss and all parameter gradients
    # --------------------------------------------------

    grads = {
        "V": grad_V,
        "mu": grad_mu,
        "W": grad_W,
        "b": grad_b
    }

    return float(loss), grads

# Step 16 - fit_autorec
def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    # Initialize and increment the optimization step counter
    state["t"] = state.get("t", 0) + 1
    t = state["t"]

    # Initialize first and second moment dictionaries
    if "m" not in state:
        state["m"] = {}

    if "v" not in state:
        state["v"] = {}

    # Update every parameter using Adam
    for key in params:
        # Retrieve the current parameter and its gradient
        param = params[key]
        grad = grads[key]

        # Initialize moment estimates for this parameter
        if key not in state["m"]:
            state["m"][key] = np.zeros_like(param)

        if key not in state["v"]:
            state["v"][key] = np.zeros_like(param)

        # Update first moment estimate
        state["m"][key] = (
            b1 * state["m"][key]
            + (1.0 - b1) * grad
        )

        # Update second moment estimate
        state["v"][key] = (
            b2 * state["v"][key]
            + (1.0 - b2) * (grad ** 2)
        )

        # Apply bias correction
        m_hat = state["m"][key] / (1.0 - b1 ** t)
        v_hat = state["v"][key] / (1.0 - b2 ** t)

        # Update parameters in place
        params[key][...] -= (
            lr * m_hat / (np.sqrt(v_hat) + eps)
        )


def fit_autorec(R, mask, k, lam, lr, steps, seed):
    # Convert inputs to NumPy arrays
    R = np.asarray(R, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    # Create an independent random number generator
    rng = np.random.default_rng([seed, 3])

    # Determine the global mean of observed ratings
    r0 = float(R[mask].mean())

    # Center observed ratings around the global mean
    # Unknown entries are represented as zero
    Y = (R - r0) * mask

    # Convert the observation mask to floating-point form
    m = mask.astype(float)

    # Get the number of rows and input dimensions
    n, d = Y.shape

    # Initialize encoder weights
    V = rng.normal(0.0, 0.02, size=(k, d))

    # Initialize hidden-layer bias
    mu = np.zeros(k)

    # Initialize decoder weights
    W = rng.normal(0.0, 0.02, size=(k, d))

    # Initialize output bias
    b = np.zeros(d)

    # Store all trainable parameters
    params = {
        "V": V,
        "mu": mu,
        "W": W,
        "b": b
    }

    # Initialize Adam optimizer state
    state = {}

    # --------------------------------------------------
    # Train AutoRec using full-batch Adam
    # --------------------------------------------------

    for _ in range(steps):

        # Calculate masked loss and parameter gradients
        loss, grads = autorec_loss_grads(
            Y, m, 
            params["V"],
            params["mu"],
            params["W"],
            params["b"],
            lam
        )

        # Perform one Adam update
        adam_update(
            params,
            grads,
            state,
            lr
        )

    # --------------------------------------------------
    # Generate the completed rating matrix
    # --------------------------------------------------

    # Run the forward pass using the trained parameters
    out, H = autorec_forward(
        Y,
        params["V"],
        params["mu"],
        params["W"],
        params["b"]
    )

    # Restore the global rating mean
    completed_matrix = out + r0

    return completed_matrix, params

