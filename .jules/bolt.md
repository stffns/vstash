## 2024-05-18 - [Optimize Pure Python Vector Math]
**Learning:** Pure python vector math like cosine similarity can be optimized by using `math.hypot(*vec)` for the $L_2$ norm and `sum(map(operator.mul, a, b))` for dot products. This avoids slow list comprehensions and generators in Python.
**Action:** Use these built-in C-accelerated functions whenever dealing with short to medium sized pure python vectors (like embeddings) where numpy overhead or dependency limits restrict external native backends.
