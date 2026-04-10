## 2024-05-18 - Optimize Job Cleanup Dictionary Iteration
**Learning:** O(N) dictionary iteration to find actionable (e.g., completed or errored) items in a constantly-growing tracking dictionary is highly inefficient.
**Action:** Always employ a side-channel queue (like `collections.deque`) to push state changes (e.g., job IDs transitioning to "completed") for O(1) popping during cleanup routines instead of full dictionary scans.
