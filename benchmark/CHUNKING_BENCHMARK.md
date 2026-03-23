# Chunking Benchmark: Fixed Window vs Semantic

**Date:** 2026-03-23 19:17
**Config:** chunk_size=1024, overlap=128


## Summary

| File | Tokens | Fixed Chunks | Semantic Chunks | Fixed Avg Tok | Semantic Avg Tok | Headers Preserved (Semantic) |
|------|--------|-------------|----------------|--------------|-----------------|------------------------------|
| climate_change_report.md | 1248 | 2 | 9 | 688.0 | 138.3 | 18/18 |
| fastapi_patterns.md | 1155 | 2 | 5 | 641.0 | 230.0 | 19/19 |
| neural_architecture_search.md | 1652 | 2 | 12 | 890.0 | 137.5 | 18/18 |
| alice_wonderland.txt | 41028 | 46 | 44 | 1017.2 | 931.7 | 0/0 |
| art_of_war.txt | 82207 | 92 | 89 | 1020.2 | 921.3 | 0/0 |
| **Total** | — | **144** | **159** | — | — | **55/55** |

## Performance

| File | Fixed Time | Semantic Time | Overhead |
|------|-----------|--------------|----------|
| climate_change_report.md | 0.29ms | 0.52ms | 79% |
| fastapi_patterns.md | 0.24ms | 0.51ms | 112% |
| neural_architecture_search.md | 0.4ms | 0.75ms | 88% |
| alice_wonderland.txt | 8.8ms | 24.05ms | 173% |
| art_of_war.txt | 16.3ms | 46.42ms | 185% |

## Sample Chunks (first chunk of each file)

### climate_change_report.md

**Fixed (first chunk):**
> # Global Climate Change: Impacts, Mitigation, and Adaptation Strategies  ## Executive Summary  Climate change represents one of the most significant challenges facing humanity in the 21st century. Global average temperatures have risen by approximately 1.1°C above pre-industrial levels, driven prima...

**Semantic (first chunk):**
> # Global Climate Change: Impacts, Mitigation, and Adaptation Strategies  ## Executive Summary  Climate change represents one of the most significant challenges facing humanity in the 21st century. Global average temperatures have risen by approximately 1.1°C above pre-industrial levels, driven prima...

### fastapi_patterns.md

**Fixed (first chunk):**
> # FastAPI Advanced Patterns: Building Production-Ready APIs  ## 1. Authentication and Authorization  ### 1.1 JWT Token Authentication  JSON Web Tokens (JWT) provide a stateless authentication mechanism. The server generates a token containing encoded claims (user ID, roles, expiration time) and sign...

**Semantic (first chunk):**
> # FastAPI Advanced Patterns: Building Production-Ready APIs  ## 1. Authentication and Authorization  ### 1.1 JWT Token Authentication  JSON Web Tokens (JWT) provide a stateless authentication mechanism. The server generates a token containing encoded claims (user ID, roles, expiration time) and sign...

### neural_architecture_search.md

**Fixed (first chunk):**
> # Neural Architecture Search: A Comprehensive Survey  ## Abstract  Neural Architecture Search (NAS) has emerged as a transformative approach to automating the design of deep neural network architectures. Traditional deep learning relies on human experts to manually design network architectures — a p...

**Semantic (first chunk):**
> # Neural Architecture Search: A Comprehensive Survey  ## Abstract  Neural Architecture Search (NAS) has emerged as a transformative approach to automating the design of deep neural network architectures. Traditional deep learning relies on human experts to manually design network architectures — a p...

### alice_wonderland.txt

**Fixed (first chunk):**
> ﻿The Project Gutenberg eBook of Alice's Adventures in Wonderland      This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project G...

**Semantic (first chunk):**
> ﻿The Project Gutenberg eBook of Alice's Adventures in Wonderland      This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project G...

### art_of_war.txt

**Fixed (first chunk):**
> ﻿The Project Gutenberg eBook of The Art of War      This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License i...

**Semantic (first chunk):**
> ﻿The Project Gutenberg eBook of The Art of War      This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License i...


## Verdict

- Semantic chunking preserved **55/55** headers with their body content
- Fixed window: **144** chunks vs Semantic: **159** chunks
- **10% more chunks** with semantic — more granular, but each chunk is more coherent