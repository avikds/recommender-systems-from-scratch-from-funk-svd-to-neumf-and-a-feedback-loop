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

