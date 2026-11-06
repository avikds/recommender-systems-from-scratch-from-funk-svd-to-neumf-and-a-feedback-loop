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

