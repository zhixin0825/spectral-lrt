"""Six-community extension of the preregistered benchmark.

Uses the same engine, target, random initialization and method budgets.
Kept in a separate wrapper so running benchmark source hashes are stable.
"""
import benchmark

benchmark.MODELS["balanced6"] = {
    "proportions": [1/6]*6,
    "base_P": [[7 if a==b else 1 for b in range(6)] for a in range(6)],
    "description": "Six balanced assortative communities",
}

if __name__ == "__main__":
    raise SystemExit(benchmark.main())
