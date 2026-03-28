"""Adaptive Cold Start Experiment — γ (maturity-gated scoring) vs fixed β.

Compares three scoring strategies on a 120-document synthetic corpus:
  1. baseline:  RRF only (no scoring)
  2. fixed:     Fixed β=0.5 from round 1 (current behavior pre-v0.7)
  3. adaptive:  β scaled by γ = outlier ratio (new behavior)

The corpus is generated locally from 12 topic clusters × 10 docs each.
Each document is a realistic multi-paragraph technical article (~600 words)
that produces 12-15 chunks when ingested through vstash's real chunking pipeline
with 80-token chunks.  Target: 120 docs → ~1,500+ chunks total.
No network access required (except for embedding model download).

Usage:
    python -m experiments.adaptive_cold_start [--rounds 30] [--output results/adaptive_cold_start.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from vstash.config import ScoringConfig
from vstash.embed import embed_query, get_embedding_dim
from vstash.ingest import chunk_text
from vstash.store import VstashStore

MODEL = "BAAI/bge-small-en-v1.5"

# Chunking parameters — use 80-token chunks with 10-token overlap to
# produce 10-25 chunks per multi-paragraph document.  This is smaller
# than the default (1024) to give the experiment a realistic chunk count
# without requiring extremely long documents.  With 120 docs averaging
# ~640 words each, this produces ~1,500-2,500 total chunks.
CHUNK_SIZE = 80
CHUNK_OVERLAP = 10


# ------------------------------------------------------------------ #
# Synthetic corpus: 12 clusters × 10 documents                        #
# Each document is a multi-paragraph technical article (300-600 words) #
# ------------------------------------------------------------------ #

# Per-topic sentence banks.  Each topic has a pool of sentences organized
# into categories: intro, core, detail, example, and conclusion.
# The document generator combines these with transitions and variation
# to produce distinct multi-paragraph articles within each topic.

TOPIC_MATERIAL: dict[str, dict[str, list[str]]] = {
    "transformer_architecture": {
        "intro": [
            "The transformer architecture has fundamentally reshaped deep learning since its introduction in 2017.",
            "Modern language models rely on the transformer as their backbone architecture for sequence processing.",
            "Understanding transformers requires grasping how attention mechanisms replace recurrent computation.",
            "Transformer models have achieved state-of-the-art results across natural language processing, computer vision, and speech recognition.",
            "The shift from recurrent neural networks to transformers represents one of the most significant paradigm changes in deep learning history.",
        ],
        "core": [
            "Self-attention computes pairwise similarities between all tokens in a sequence using query, key, and value projections derived from learned weight matrices. The attention scores are computed as the scaled dot product of queries and keys, normalized by the square root of the key dimension to prevent vanishing gradients during softmax computation.",
            "Multi-head attention partitions the model dimension into multiple heads, each learning different representation subspaces. This allows the model to jointly attend to information from different positions and capture various types of linguistic relationships, such as syntactic dependencies and semantic associations.",
            "Positional encoding injects sequence order information into the model since self-attention is inherently permutation-invariant. The original transformer uses sinusoidal functions of different frequencies, while modern variants employ learned positional embeddings or relative position encodings like RoPE and ALiBi.",
            "The feed-forward network in each transformer layer consists of two linear transformations with a nonlinear activation between them. This component applies the same transformation independently to each position, acting as a per-token nonlinear feature transformation that complements the cross-token mixing performed by attention.",
            "Layer normalization and residual connections are critical for training deep transformer networks. Residual connections allow gradients to flow directly through the network, while layer normalization stabilizes the distribution of activations, enabling training of models with hundreds of layers.",
            "The encoder processes input sequences bidirectionally, allowing each token to attend to all other tokens. In contrast, the decoder uses masked self-attention to prevent positions from attending to subsequent positions, maintaining the autoregressive property necessary for generation tasks.",
            "Cross-attention layers in encoder-decoder architectures allow the decoder to attend to all encoder hidden states. This mechanism enables the decoder to selectively focus on relevant parts of the input when generating each output token, which is essential for tasks like machine translation and summarization.",
        ],
        "detail": [
            "Sparse attention patterns, such as those used in Longformer and BigBird, reduce the quadratic complexity of standard self-attention to linear or near-linear scaling. These approaches use combinations of local sliding window attention, global attention on special tokens, and random attention patterns to approximate full attention.",
            "Flash attention optimizes the memory access patterns of attention computation by tiling the computation to fit within GPU SRAM. This approach avoids materializing the full attention matrix in high-bandwidth memory, achieving significant speedups and memory savings without approximation.",
            "Relative position embeddings encode the distance between tokens rather than their absolute positions. Rotary Position Embedding (RoPE) achieves this by rotating query and key vectors based on their position, naturally encoding relative distances through the dot product operation.",
            "KV-cache optimization stores previously computed key and value tensors during autoregressive generation to avoid redundant computation. This technique is essential for efficient inference in large language models, trading memory for computation and reducing the cost of generating each new token.",
            "Mixture-of-experts layers replace the dense feed-forward network with multiple specialized expert networks and a gating mechanism. Only a subset of experts is activated for each token, enabling models to scale parameters without proportionally increasing computation.",
        ],
        "example": [
            "For instance, in a 12-head attention layer with dimension 768, each head operates on a 64-dimensional subspace, learning distinct attention patterns that capture different aspects of the input.",
            "Consider a translation task where the decoder generates the French word 'chat' — cross-attention allows it to focus strongly on the English word 'cat' in the encoder output while largely ignoring irrelevant context.",
            "As a concrete example, GPT-3 uses 96 transformer layers with 96 attention heads each and a model dimension of 12,288, demonstrating how the basic transformer block scales to 175 billion parameters.",
            "In practice, a transformer processing a 4,096-token sequence with standard attention requires storing a 4096x4096 attention matrix per head, consuming substantial GPU memory that sparse attention methods aim to reduce.",
        ],
        "conclusion": [
            "The continued evolution of transformer architectures, from efficiency improvements to novel attention mechanisms, remains central to advances in artificial intelligence.",
            "As researchers push the boundaries of model scale and efficiency, the fundamental transformer building blocks continue to prove remarkably adaptable and powerful.",
            "Future directions include more efficient attention mechanisms, better position encodings for very long sequences, and hybrid architectures that combine the strengths of transformers with other computational paradigms.",
        ],
    },
    "reinforcement_learning": {
        "intro": [
            "Reinforcement learning addresses the problem of learning optimal behavior through interaction with an environment that provides reward signals.",
            "The field of reinforcement learning bridges control theory, operations research, and machine learning to solve sequential decision-making problems.",
            "Unlike supervised learning, reinforcement learning agents must discover optimal strategies through trial and error without access to labeled training examples.",
            "Reinforcement learning has achieved remarkable successes in game playing, robotics, and resource management, demonstrating its broad applicability.",
            "The challenge of balancing exploration and exploitation lies at the heart of reinforcement learning, requiring agents to gather information while maximizing cumulative reward.",
        ],
        "core": [
            "Q-learning is a model-free algorithm that estimates the optimal action-value function through iterative Bellman equation updates. By maintaining a table or neural network approximation of Q-values for each state-action pair, the agent learns which actions yield the highest expected cumulative reward.",
            "Policy gradient methods directly optimize the policy by estimating the gradient of expected cumulative reward with respect to policy parameters. The REINFORCE algorithm samples trajectories and uses the log-probability trick to compute gradient estimates, though these estimates can have high variance.",
            "Actor-critic architectures combine the strengths of value-based and policy-based methods. The actor learns a parameterized policy, while the critic estimates the value function to provide lower-variance gradient estimates, leading to more stable and efficient learning.",
            "Proximal Policy Optimization constrains policy updates by clipping the probability ratio between new and old policies. This prevents destructively large updates that could collapse performance, providing a simple yet effective trust region approach to policy optimization.",
            "Reward shaping provides dense intermediate reward signals to guide agents through environments where natural rewards are sparse. Potential-based reward shaping preserves the optimal policy while dramatically accelerating learning by reducing the credit assignment problem.",
            "Model-based reinforcement learning builds explicit models of the environment dynamics to enable planning and sample-efficient learning. By learning a transition model and reward function, agents can simulate trajectories without real environment interaction, reducing the number of real samples needed.",
        ],
        "detail": [
            "Multi-agent reinforcement learning extends the single-agent framework to settings where multiple agents simultaneously learn and interact. Challenges include non-stationarity of the environment from each agent's perspective, coordination in cooperative settings, and strategic reasoning in competitive scenarios.",
            "Hierarchical reinforcement learning decomposes complex tasks into subtasks using temporal abstraction. The options framework defines temporally extended actions with initiation sets, policies, and termination conditions, enabling agents to plan at multiple levels of abstraction.",
            "Inverse reinforcement learning infers the underlying reward function from expert demonstrations rather than requiring explicit reward specification. This approach addresses the difficulty of reward engineering by learning what the expert values from observed behavior.",
            "Offline reinforcement learning, also called batch RL, trains policies entirely from fixed datasets of previously collected experience. This is critical for applications where online exploration is expensive or dangerous, such as healthcare and autonomous driving.",
            "Distributional reinforcement learning models the full distribution of returns rather than just the expected value. Algorithms like C51 and QR-DQN learn a distribution over possible outcomes, providing richer information for decision-making and improved performance in practice.",
        ],
        "example": [
            "AlphaGo demonstrated the power of combining Monte Carlo tree search with deep reinforcement learning to defeat the world champion in the game of Go, a milestone previously thought to be decades away.",
            "In robotic manipulation, policy gradient methods have enabled robots to learn dexterous skills like in-hand object rotation directly from raw sensory input, without explicit kinematic models.",
            "OpenAI Five used large-scale distributed PPO to train a team of five agents that cooperated to defeat professional human players in the complex real-time strategy game Dota 2.",
            "In recommendation systems, reinforcement learning agents learn to select items that maximize long-term user engagement rather than optimizing for immediate clicks, leading to better user experiences over time.",
        ],
        "conclusion": [
            "The theoretical foundations and practical algorithms of reinforcement learning continue to expand, driven by both challenging benchmark environments and real-world applications.",
            "As sample efficiency improves and safety guarantees mature, reinforcement learning is poised to tackle increasingly complex real-world decision-making problems.",
            "The intersection of reinforcement learning with large language models and foundation models represents an exciting frontier for developing generally capable AI agents.",
        ],
    },
    "natural_language_processing": {
        "intro": [
            "Natural language processing encompasses computational techniques for understanding, generating, and manipulating human language at scale.",
            "The evolution of NLP from rule-based systems to neural approaches has dramatically improved performance across virtually every language technology task.",
            "Modern NLP leverages pretrained language models that capture rich linguistic knowledge from massive text corpora through self-supervised learning.",
            "Natural language processing sits at the intersection of linguistics, computer science, and artificial intelligence, combining insights from all three fields.",
            "The ability to process and understand natural language is a cornerstone capability for building intelligent systems that interact naturally with humans.",
        ],
        "core": [
            "Named entity recognition identifies and classifies mentions of entities such as persons, organizations, locations, and dates within text. Modern NER systems use sequence labeling approaches, typically framing the task as BIO tagging with neural architectures like BiLSTM-CRF or fine-tuned transformers.",
            "Dependency parsing analyzes the grammatical structure of sentences by identifying directed relationships between words. These syntactic trees reveal which words modify others and how the sentence is structurally organized, providing essential information for downstream semantic analysis.",
            "Sentiment analysis classifies text according to the emotional tone or opinion expressed by the author. Beyond simple positive-negative classification, fine-grained sentiment analysis detects aspect-level opinions, emotional intensity, and the targets of expressed sentiments.",
            "Machine translation converts text from one natural language to another using encoder-decoder neural architectures. Modern systems achieve near-human quality for many language pairs by pretraining on large parallel corpora and using techniques like back-translation for data augmentation.",
            "Text summarization condenses long documents into shorter representations that preserve the most salient information. Extractive methods select key sentences from the source, while abstractive methods generate novel text that paraphrases and reorganizes the content.",
            "Question answering systems extract or generate answers from context passages given a natural language query. Extractive QA identifies answer spans within provided documents, while generative QA synthesizes answers by reasoning over retrieved evidence.",
        ],
        "detail": [
            "Coreference resolution links pronouns, definite descriptions, and other referring expressions to their antecedent entities in discourse. This task is essential for document understanding, as resolving coreference chains reveals who or what is being discussed throughout a text.",
            "Word sense disambiguation determines which meaning of a polysemous word is intended in a given context. For example, the word 'bank' may refer to a financial institution or a river bank, and disambiguation relies on surrounding words to resolve the ambiguity.",
            "Tokenization splits text into subword units that balance vocabulary size and sequence length. Algorithms like BPE, WordPiece, and SentencePiece learn data-driven vocabularies that handle rare words by decomposing them into frequent subword units.",
            "Semantic role labeling identifies the predicate-argument structure of sentences, determining who did what to whom, where, and when. This level of analysis bridges syntax and semantics, enabling downstream tasks to reason about events and their participants.",
            "Information extraction automatically identifies structured facts from unstructured text, including entities, relations, and events. Knowledge graph construction pipelines combine NER, relation extraction, and entity linking to populate structured knowledge bases from raw documents.",
        ],
        "example": [
            "For example, in the sentence 'Apple acquired Beats Electronics for three billion dollars in 2014,' NER should identify Apple and Beats Electronics as organizations, three billion dollars as money, and 2014 as a date.",
            "Consider the challenge of translating idiomatic expressions: 'it's raining cats and dogs' must be rendered as the target language's equivalent idiom rather than translated literally.",
            "A practical application of text summarization is generating concise abstracts from medical research papers, enabling clinicians to quickly assess the relevance of new studies to their practice.",
            "In coreference resolution, the sentence 'The committee met and they decided to postpone' requires linking 'they' back to 'the committee' to correctly understand the passage.",
        ],
        "conclusion": [
            "Natural language processing continues to advance rapidly, with pretrained models achieving increasingly human-like performance across diverse linguistic tasks.",
            "The integration of NLP with knowledge bases, multimodal understanding, and reasoning capabilities points toward systems that truly comprehend language rather than merely pattern-matching.",
            "Challenges remain in handling low-resource languages, reducing bias in language models, and achieving robust understanding of nuanced or ambiguous language.",
        ],
    },
    "computer_vision": {
        "intro": [
            "Computer vision enables machines to interpret and understand visual information from the world, encompassing image classification, object detection, and scene understanding.",
            "The deep learning revolution transformed computer vision from hand-crafted feature engineering to end-to-end learned representations that surpass human performance on many benchmarks.",
            "Visual understanding is fundamental to applications ranging from autonomous vehicles and medical imaging to augmented reality and industrial quality inspection.",
            "Computer vision algorithms must handle enormous variability in viewpoint, lighting, occlusion, and scale, making robust visual recognition a challenging problem.",
            "The convergence of large datasets, powerful GPUs, and novel architectures has driven unprecedented progress in computer vision over the past decade.",
        ],
        "core": [
            "Convolutional neural networks apply learned filters across spatial locations to extract hierarchical visual features from images. Early layers detect edges and textures, while deeper layers recognize increasingly abstract patterns like object parts and complete objects.",
            "Object detection localizes and classifies multiple objects within an image using a combination of region proposals and classification heads. Two-stage detectors like Faster R-CNN first propose candidate regions, then classify each region, while single-stage detectors like YOLO perform both steps simultaneously.",
            "Image segmentation assigns a class label to every pixel, enabling fine-grained scene understanding. Semantic segmentation labels each pixel by class, instance segmentation additionally distinguishes individual object instances, and panoptic segmentation combines both for complete scene parsing.",
            "Data augmentation artificially expands training datasets through geometric transformations, color jittering, random erasing, and mixup strategies. These techniques improve generalization by exposing models to more variation during training without requiring additional labeled data.",
            "Transfer learning leverages features learned from large-scale pretraining on datasets like ImageNet to improve performance on downstream tasks with limited labeled data. Fine-tuning pretrained models has become the standard approach, dramatically reducing the data and computation needed for new visual tasks.",
            "Vision transformers partition images into fixed-size patches and process them as token sequences using self-attention. ViT models have demonstrated that pure transformer architectures can match or exceed the performance of CNNs when trained on sufficient data.",
        ],
        "detail": [
            "Generative adversarial networks synthesize realistic images through adversarial training of a generator and discriminator. The generator learns to produce images that fool the discriminator, while the discriminator learns to distinguish real from generated images, driving both networks to improve.",
            "Feature pyramid networks aggregate multi-scale features from different stages of a backbone network to detect objects across a wide range of sizes. The top-down pathway with lateral connections creates a rich multi-scale feature representation that benefits both small and large object detection.",
            "Optical flow estimation computes dense pixel-level motion vectors between consecutive video frames. Learning-based methods like FlowNet and RAFT use neural networks to predict flow fields, enabling applications in video analysis, action recognition, and autonomous navigation.",
            "Self-supervised visual pretraining methods like MAE, DINO, and contrastive learning extract powerful representations from unlabeled images. These approaches define pretext tasks that provide supervision without manual annotation, enabling pretraining on virtually unlimited image data.",
            "Neural radiance fields represent 3D scenes as continuous volumetric functions that map spatial coordinates and viewing directions to color and density. NeRF enables photorealistic novel view synthesis from a sparse set of input photographs.",
        ],
        "example": [
            "In autonomous driving, object detection must reliably identify pedestrians, vehicles, cyclists, and traffic signs under varying weather and lighting conditions, with latency requirements under 50 milliseconds per frame.",
            "Medical imaging applications use segmentation to delineate tumor boundaries in MRI scans, enabling precise measurement of tumor volume and tracking of disease progression over time.",
            "For example, a ResNet-50 pretrained on ImageNet can be fine-tuned on a dataset of only a few hundred satellite images to accurately classify land use patterns, demonstrating the power of transfer learning.",
            "Style transfer applications demonstrate how GANs and neural optimization can transform photographs to match the visual style of famous paintings while preserving the original content structure.",
        ],
        "conclusion": [
            "Computer vision continues to push boundaries with larger models, more efficient architectures, and the integration of language understanding for multimodal reasoning.",
            "The proliferation of visual AI into safety-critical applications demands continued research into robustness, fairness, and interpretability of computer vision models.",
            "Emerging directions include 3D scene understanding, video foundation models, and embodied visual agents that interact with physical environments.",
        ],
    },
    "database_systems": {
        "intro": [
            "Database systems provide the foundational infrastructure for storing, querying, and managing structured data in virtually every software application.",
            "The design of modern database systems balances competing demands of consistency, performance, durability, and scalability across diverse workloads.",
            "From relational databases to NoSQL stores and NewSQL systems, the landscape of data management continues to evolve in response to changing application requirements.",
            "Understanding database internals is essential for developers and architects who need to design systems that perform well under production workloads.",
            "Database systems embody decades of research in data structures, algorithms, concurrency control, and distributed computing.",
        ],
        "core": [
            "B-tree indexes organize data in balanced tree structures that support efficient range queries and point lookups with logarithmic time complexity. Each internal node contains sorted keys and child pointers, while leaf nodes hold the actual data or pointers to data pages, enabling the database to quickly narrow the search space.",
            "Write-ahead logging ensures durability by persisting all modifications to a sequential log before applying them to data pages. In the event of a crash, the recovery process replays committed transactions from the log and undoes incomplete transactions, guaranteeing that no committed data is lost.",
            "Query optimization transforms declarative SQL queries into efficient execution plans using cost-based analysis. The optimizer estimates the cost of different plan alternatives by considering statistics about data distribution, available indexes, and the computational cost of each operator.",
            "Multi-version concurrency control provides snapshot isolation by maintaining multiple versions of each data row. Readers access a consistent snapshot of the database without blocking writers, and writers create new versions rather than modifying existing data, enabling high concurrency for mixed read-write workloads.",
            "Columnar storage formats organize data by column rather than by row, dramatically improving analytical query performance. By storing each column contiguously, the storage engine can skip irrelevant columns during query execution and achieve much better compression ratios due to similar values being stored together.",
        ],
        "detail": [
            "Distributed consensus protocols like Raft and Paxos ensure data consistency across replicated database nodes by requiring a majority quorum to agree on each state change. These protocols handle leader election, log replication, and membership changes while tolerating minority node failures.",
            "Hash joins build an in-memory hash table on the smaller relation's join key, then probe the hash table with rows from the larger relation. This technique achieves expected linear-time performance for equi-joins, making it the preferred join algorithm when sufficient memory is available.",
            "Connection pooling maintains a cache of reusable database connections to eliminate the overhead of TCP handshakes, SSL negotiation, and authentication for each query. Pool sizing, connection validation, and idle timeout management are critical for production database deployments.",
            "Materialized views precompute and cache the results of expensive queries, providing fast access to aggregated or joined data at the cost of storage and maintenance overhead. Incremental view maintenance techniques update materialized views efficiently as underlying data changes.",
            "Sharding partitions data across multiple database nodes using a shard key to distribute load and enable horizontal scalability. Cross-shard queries and distributed transactions introduce significant complexity, making shard key selection a critical design decision that impacts system performance.",
        ],
        "example": [
            "For instance, a B-tree index with a branching factor of 100 can locate any record among 100 million rows in at most 4 disk reads, demonstrating why B-trees are the default index structure in most relational databases.",
            "Consider an e-commerce application where columnar storage enables a query computing total sales by product category to scan only the category and amount columns, reading perhaps 5% of the total data compared to a row-oriented scan.",
            "A practical example of WAL in action: PostgreSQL writes every modification to its WAL before acknowledging the transaction, and the background writer lazily flushes dirty pages to disk, decoupling transaction latency from disk write speed.",
            "In a sharded social media database, user data might be sharded by user ID, ensuring that all of a user's posts, friends, and messages reside on the same shard for efficient single-shard queries.",
        ],
        "conclusion": [
            "Database systems continue to evolve with innovations in hardware-aware query processing, learned indexes, and serverless architectures that automatically scale to meet demand.",
            "The tension between consistency and scalability drives ongoing research into new transaction protocols, replication strategies, and hybrid storage architectures.",
            "As data volumes grow and latency requirements tighten, the fundamental principles of database design remain essential while implementations adapt to new hardware and workload patterns.",
        ],
    },
    "distributed_systems": {
        "intro": [
            "Distributed systems coordinate multiple networked computers to achieve goals that no single machine could accomplish alone, trading simplicity for scalability and fault tolerance.",
            "Building reliable distributed systems requires understanding the fundamental impossibility results, consistency models, and failure modes that govern their behavior.",
            "The proliferation of cloud computing and microservice architectures has made distributed systems design a core competency for modern software engineers.",
            "Distributed systems face unique challenges including network partitions, partial failures, clock skew, and the difficulty of maintaining consistent state across nodes.",
            "From web-scale search engines to global financial systems, distributed computing underpins the infrastructure that powers the modern internet.",
        ],
        "core": [
            "The CAP theorem establishes that a distributed system cannot simultaneously guarantee consistency, availability, and partition tolerance, and must sacrifice one property during network partitions. In practice, most systems choose between CP (consistent but potentially unavailable) and AP (available but potentially inconsistent) configurations.",
            "Eventual consistency allows replicas to temporarily diverge during normal operation but guarantees that all replicas will converge to the same state once updates stop. This model trades immediate consistency for higher availability and lower latency, making it suitable for applications where brief inconsistency is acceptable.",
            "Leader election algorithms select a single coordinator node to manage writes and maintain ordering in a distributed cluster. Protocols like Raft use term-based elections with randomized timeouts to break symmetry and quickly elect a new leader when the current leader fails.",
            "Consistent hashing distributes keys across nodes in a way that minimizes redistribution when nodes join or leave the cluster. By mapping both nodes and keys to positions on a virtual ring, only a fraction of keys need to be remapped during topology changes.",
            "Two-phase commit coordinates atomic transactions across multiple distributed participants by first collecting prepare votes and then issuing a global commit or abort decision. While this protocol ensures atomicity, it is blocking — a coordinator failure can leave participants in an uncertain state indefinitely.",
        ],
        "detail": [
            "Vector clocks track the causal ordering of events across processes by maintaining a vector of logical timestamps. Each process increments its own counter on local events and merges vectors on message receipt, enabling detection of concurrent events without relying on synchronized physical clocks.",
            "Gossip protocols disseminate information through probabilistic peer-to-peer message exchanges, achieving eventual consistency with high probability. Each node periodically selects random peers and exchanges state updates, providing robust dissemination even in the presence of node failures.",
            "Circuit breakers prevent cascading failures in distributed systems by monitoring error rates and temporarily stopping requests to unhealthy downstream services. When the error rate exceeds a threshold, the circuit breaker opens, returning fast failures instead of waiting for timeouts.",
            "CRDTs (Conflict-free Replicated Data Types) enable conflict-free replication by encoding merge semantics directly into the data structure. Operations on CRDTs are designed to be commutative, associative, and idempotent, guaranteeing that replicas always converge regardless of message ordering.",
            "Service mesh infrastructure provides a dedicated layer for managing service-to-service communication, including load balancing, mutual TLS, traffic shaping, and distributed tracing. Sidecar proxies like Envoy intercept all network traffic, decoupling operational concerns from application code.",
        ],
        "example": [
            "Amazon's Dynamo exemplifies an AP system that uses vector clocks for conflict detection and application-level conflict resolution, prioritizing availability for the shopping cart service where lost writes are more costly than brief inconsistency.",
            "In a Raft-based distributed database, if the leader becomes unreachable, followers observe the election timeout, increment the term, and start a new election — the cluster typically elects a new leader within a few hundred milliseconds.",
            "Consider a global chat application using CRDTs: two users adding messages simultaneously in different data centers can both succeed without coordination, and the message lists will converge to the same order on all replicas.",
            "A real-world circuit breaker example: Netflix's Hystrix library monitors downstream service latency and error rates, opening the circuit to prevent a slow inventory service from causing timeouts across the entire request chain.",
        ],
        "conclusion": [
            "Distributed systems design remains a field where theoretical impossibility results inform practical engineering trade-offs, and understanding these trade-offs is essential for building reliable systems.",
            "The evolution from monolithic architectures to microservices and serverless platforms continues to push the boundaries of distributed systems research and engineering practice.",
            "Future challenges include managing increasingly complex service topologies, handling edge computing scenarios, and building distributed systems that are both correct and operationally manageable.",
        ],
    },
    "cryptography": {
        "intro": [
            "Cryptography provides the mathematical foundations for securing communication, authenticating identities, and protecting data in an increasingly connected world.",
            "Modern cryptographic systems rely on computational hardness assumptions, where the security of algorithms depends on the intractability of specific mathematical problems.",
            "The field of cryptography spans symmetric and asymmetric encryption, hash functions, digital signatures, and advanced protocols for privacy-preserving computation.",
            "Cryptography underpins virtually every aspect of digital security, from HTTPS connections and encrypted messaging to blockchain technology and digital currencies.",
            "As computing power grows and quantum computers approach practical capability, cryptographic systems must evolve to maintain their security guarantees.",
        ],
        "core": [
            "AES (Advanced Encryption Standard) is a symmetric block cipher that processes 128-bit blocks through multiple rounds of substitution, permutation, and key mixing operations. With key sizes of 128, 192, or 256 bits, AES provides a strong security margin and is the most widely deployed symmetric encryption algorithm worldwide.",
            "RSA public key cryptography enables secure communication without shared secrets by relying on the computational difficulty of factoring large semiprime numbers. The public key encrypts messages that only the corresponding private key can decrypt, enabling both confidentiality and digital signatures.",
            "Cryptographic hash functions like SHA-256 produce fixed-size message digests that serve as digital fingerprints of arbitrary-length inputs. These functions must satisfy three properties: preimage resistance, second preimage resistance, and collision resistance, ensuring that finding inputs with matching hashes is computationally infeasible.",
            "Digital signatures verify the authenticity and integrity of messages using asymmetric key pairs. The signer creates a signature using their private key, and any recipient can verify the signature using the corresponding public key, providing non-repudiation — the signer cannot deny having signed the message.",
            "Key exchange protocols like Diffie-Hellman enable two parties to establish a shared secret over an insecure channel without prior shared knowledge. The protocol relies on the difficulty of the discrete logarithm problem, allowing both parties to compute the same shared secret from public information without an eavesdropper being able to derive it.",
        ],
        "detail": [
            "Zero-knowledge proofs demonstrate possession of information without revealing the information itself. A prover can convince a verifier that a statement is true without disclosing any details beyond the validity of the statement, enabling applications in authentication, blockchain privacy, and credential verification.",
            "Homomorphic encryption allows computation directly on encrypted data, producing encrypted results that, when decrypted, match the results of operations performed on the plaintext. Fully homomorphic encryption supports arbitrary computations, enabling cloud computing on sensitive data without exposing it to the service provider.",
            "Certificate authorities issue X.509 digital certificates that bind public keys to verified domain identities, forming the trust infrastructure of the internet. The certificate chain of trust enables browsers to verify that a website's public key genuinely belongs to the claimed domain through a hierarchy of trusted issuers.",
            "Post-quantum cryptography develops algorithms that remain secure against attacks from quantum computers, which threaten current public-key systems. Lattice-based, hash-based, and code-based cryptographic schemes are being standardized through the NIST post-quantum cryptography process.",
            "Secure multi-party computation enables multiple parties to jointly compute a function over their private inputs without revealing those inputs to each other. Applications include private auctions, joint financial analysis, and privacy-preserving machine learning across organizational boundaries.",
        ],
        "example": [
            "When you connect to a website via HTTPS, your browser verifies the server's certificate, performs a Diffie-Hellman key exchange to establish a shared session key, and then uses AES to encrypt all subsequent communication — demonstrating how multiple cryptographic primitives compose into secure protocols.",
            "Bitcoin uses SHA-256 hashing in its proof-of-work consensus mechanism, where miners must find a nonce that produces a hash with a specific number of leading zeros, making block creation computationally expensive while verification remains trivially fast.",
            "Signal messenger implements the Double Ratchet algorithm, which combines Diffie-Hellman key exchange with symmetric key ratcheting to provide forward secrecy — compromising a single message key does not reveal past or future message contents.",
            "A practical zero-knowledge proof application: proving you are over 18 without revealing your exact birthdate, using cryptographic commitments and range proofs to verify the age predicate without disclosing personal information.",
        ],
        "conclusion": [
            "Cryptography remains the cornerstone of digital trust, with ongoing research addressing post-quantum security, privacy-preserving computation, and the tension between strong encryption and legitimate surveillance needs.",
            "The standardization of post-quantum cryptographic algorithms marks a critical transition as the community prepares for the eventual arrival of cryptographically relevant quantum computers.",
            "Advances in zero-knowledge proofs and homomorphic encryption are opening new possibilities for privacy-preserving systems that were previously considered impractical.",
        ],
    },
    "operating_systems": {
        "intro": [
            "Operating systems manage hardware resources and provide abstractions that enable application software to run efficiently and securely on diverse computing platforms.",
            "The design of modern operating systems reflects decades of evolution from simple batch processing monitors to sophisticated concurrent, networked, and virtualized systems.",
            "Understanding operating system internals is essential for developing high-performance applications, debugging system-level issues, and designing distributed infrastructure.",
            "Operating systems sit at the critical boundary between hardware and software, mediating all access to CPU, memory, storage, and network resources.",
            "From smartphones and embedded devices to cloud servers and supercomputers, operating systems adapt common principles to vastly different hardware platforms and workload requirements.",
        ],
        "core": [
            "Process scheduling algorithms allocate CPU time among competing processes to maximize throughput, minimize response time, and ensure fairness. The Completely Fair Scheduler in Linux uses a red-black tree of virtual runtimes to approximate ideal fair queuing, while real-time schedulers use priority-based preemption to meet strict timing deadlines.",
            "Virtual memory maps each process's address space to physical memory pages through page tables maintained by the kernel and translated by the hardware MMU. This abstraction provides process isolation, enables memory overcommitment through demand paging, and allows the operating system to transparently move pages between RAM and swap storage.",
            "File systems organize persistent data into hierarchical directory structures with metadata stored in inodes or equivalent structures. Modern file systems like ext4, XFS, and Btrfs provide journaling for crash consistency, extent-based allocation for large files, and copy-on-write semantics for efficient snapshots.",
            "Interrupt handlers respond to hardware signals by suspending current execution and transferring control to kernel service routines. The interrupt handling subsystem must minimize latency while managing interrupt priorities, shared interrupt lines, and the transition between interrupt context and process context.",
            "Synchronization primitives including mutexes, semaphores, and read-write locks coordinate concurrent threads to prevent race conditions on shared resources. The kernel also provides futexes for fast user-space locking and RCU (Read-Copy-Update) for scalable read-mostly data structures.",
        ],
        "detail": [
            "Page replacement policies determine which memory pages to evict when physical memory is exhausted. The LRU (Least Recently Used) policy approximates the optimal replacement strategy, while modern systems use clock-based approximations and active/inactive page lists to balance accuracy with implementation efficiency.",
            "System calls provide the programmatic interface between user-space applications and kernel-level operations. The syscall mechanism involves a mode switch from user to kernel space, parameter validation, execution of the requested kernel function, and return of results to the calling process.",
            "Device drivers translate generic I/O requests from the kernel's device-independent layer into hardware-specific commands for peripheral devices. The driver model includes probe and initialization routines, interrupt handling, power management callbacks, and compliance with the kernel's locking and memory management rules.",
            "Container technologies use kernel namespaces to isolate process trees, network stacks, and file systems for lightweight virtualization. Combined with cgroups for resource limiting and union file systems for layered storage, containers provide near-native performance with strong isolation between workloads.",
            "Memory-mapped I/O maps device registers and buffers into the processor's address space, allowing software to interact with hardware using standard load and store instructions. This approach simplifies driver implementation and enables direct memory access, where devices transfer data to and from memory without CPU involvement.",
        ],
        "example": [
            "When a user launches a web browser, the operating system creates a new process with its own virtual address space, loads the executable and shared libraries, and begins scheduling the process's threads alongside hundreds of other active processes.",
            "Consider a database server under heavy load: the operating system's page cache transparently keeps frequently accessed data pages in memory, the I/O scheduler reorders disk requests for efficiency, and the process scheduler ensures fair CPU allocation among concurrent queries.",
            "Docker containers demonstrate namespace isolation in practice: each container sees its own PID namespace (where its main process is PID 1), its own network namespace (with separate IP addresses and routing tables), and its own mount namespace (with an independent filesystem view).",
            "A real-time audio application illustrates scheduling priorities: the audio processing thread runs with SCHED_FIFO policy at high priority, ensuring it always preempts lower-priority tasks to meet its strict deadline of delivering audio frames every few milliseconds.",
        ],
        "conclusion": [
            "Operating systems continue to evolve to support new hardware architectures, emerging workloads like machine learning accelerators, and increasingly complex security requirements.",
            "The trend toward hardware heterogeneity, with CPUs, GPUs, and specialized accelerators, challenges operating system designers to provide unified and efficient resource management.",
            "Microkernel and unikernel architectures offer alternative designs that trade flexibility for improved security, reduced attack surface, and more predictable performance characteristics.",
        ],
    },
    "graph_algorithms": {
        "intro": [
            "Graph algorithms provide fundamental techniques for analyzing relationships, networks, and structural patterns in data represented as vertices and edges.",
            "The study of graph algorithms spans theoretical computer science and practical applications in social networks, transportation, biology, and infrastructure planning.",
            "Efficient graph algorithms are essential for processing the massive network datasets generated by modern applications, from web link analysis to protein interaction networks.",
            "Graph problems appear in surprisingly diverse domains, and mastering core graph algorithms provides a powerful toolkit for solving complex computational challenges.",
            "The rich mathematical structure of graphs enables elegant algorithmic solutions, but also gives rise to computationally hard problems that require approximation or heuristic approaches.",
        ],
        "core": [
            "Breadth-first search explores vertices layer by layer starting from a source node, discovering all vertices at distance k before any vertex at distance k+1. This strategy naturally computes shortest paths in unweighted graphs and forms the basis for many higher-level graph algorithms.",
            "Dijkstra's algorithm computes single-source shortest paths in graphs with non-negative edge weights by greedily selecting the unvisited vertex with minimum tentative distance. Using a priority queue implementation, the algorithm runs in O((V + E) log V) time, making it efficient for sparse graphs.",
            "Minimum spanning tree algorithms connect all vertices of a graph with the minimum total edge weight. Kruskal's algorithm sorts edges by weight and greedily adds them if they don't create a cycle, while Prim's algorithm grows the tree from a starting vertex by always adding the cheapest crossing edge.",
            "Topological sorting produces a linear ordering of directed acyclic graph vertices such that for every edge (u, v), vertex u appears before v. This ordering is essential for scheduling tasks with dependencies, resolving build orders, and evaluating dataflow computations.",
            "Maximum flow algorithms determine the greatest throughput from a source to a sink in a capacity-constrained network. The Ford-Fulkerson method repeatedly finds augmenting paths and increases flow along them, while the push-relabel approach works with local operations at individual vertices.",
        ],
        "detail": [
            "Strongly connected components partition a directed graph into maximal subgraphs where every vertex is reachable from every other vertex. Tarjan's algorithm finds all SCCs in a single depth-first search pass using a stack and low-link values, running in linear time.",
            "Graph coloring assigns colors to vertices such that no two adjacent vertices share the same color, using the minimum number of colors possible. While optimal coloring is NP-hard in general, greedy heuristics and specialized algorithms for planar graphs provide practical solutions for applications like register allocation and scheduling.",
            "PageRank computes importance scores for nodes in a directed graph based on the link structure, modeling a random surfer who follows links with probability proportional to the outgoing edges. The algorithm iteratively updates scores until convergence, and a damping factor accounts for random jumps to arbitrary pages.",
            "Community detection algorithms identify densely connected groups of vertices within large networks. Methods like the Louvain algorithm optimize modularity by iteratively merging communities, while spectral clustering uses the eigenvectors of the graph Laplacian to partition vertices into clusters.",
            "A* search combines the actual path cost from the start with a heuristic estimate of the remaining distance to the goal, expanding the most promising nodes first. When the heuristic is admissible (never overestimates), A* is guaranteed to find the optimal path while typically exploring far fewer nodes than uninformed search.",
        ],
        "example": [
            "Navigation applications use Dijkstra's algorithm or A* search with road network graphs to compute driving directions, where edge weights represent travel times that account for distance, speed limits, and real-time traffic conditions.",
            "Social network analysis uses community detection to identify groups of closely connected users, enabling features like friend suggestions, content recommendations, and understanding information diffusion patterns.",
            "In compiler design, topological sorting of the dependency graph determines the order in which source files must be compiled, ensuring that each file is compiled only after all of its dependencies are available.",
            "Google's original search ranking was based on PageRank, which treated the web as a directed graph of hyperlinks and assigned higher importance scores to pages that were linked by many other important pages.",
        ],
        "conclusion": [
            "Graph algorithms remain indispensable tools in computer science, with applications that continue to grow as network data becomes ubiquitous in science, technology, and society.",
            "The challenge of scaling graph algorithms to billion-edge networks drives ongoing research into parallel, distributed, and approximate graph processing frameworks.",
            "Emerging directions include temporal and dynamic graph algorithms, graph neural networks that combine graph structure with deep learning, and quantum algorithms for graph optimization problems.",
        ],
    },
    "machine_learning_fundamentals": {
        "intro": [
            "Machine learning enables computers to learn patterns from data and make predictions without being explicitly programmed with task-specific rules.",
            "The foundations of machine learning draw from statistics, optimization theory, and information theory to develop algorithms that generalize from finite training examples.",
            "Understanding core machine learning concepts is essential for practitioners who need to select appropriate models, diagnose training issues, and deploy reliable systems.",
            "Machine learning encompasses supervised, unsupervised, and self-supervised paradigms, each suited to different types of problems and data availability scenarios.",
            "The practical success of machine learning depends not only on algorithm choice but also on data quality, feature representation, and careful evaluation methodology.",
        ],
        "core": [
            "Gradient descent iteratively adjusts model parameters in the direction that reduces the loss function, with the step size controlled by the learning rate. Stochastic gradient descent processes random minibatches of training examples, introducing beneficial noise that helps escape local minima and improves generalization.",
            "Regularization techniques prevent overfitting by constraining model complexity beyond what the training data alone would dictate. L2 regularization adds a penalty proportional to the squared magnitude of weights, while dropout randomly deactivates neurons during training, forcing the network to learn redundant representations.",
            "Cross-validation partitions the available data into k folds and iteratively trains on k-1 folds while evaluating on the held-out fold. This procedure provides a more reliable estimate of generalization performance than a single train-test split, especially when the total dataset is small.",
            "Feature engineering transforms raw input data into informative representations that improve model predictive power. Effective features capture domain-relevant patterns and can include normalized numerical values, one-hot encoded categories, interaction terms, polynomial features, and domain-specific transformations.",
            "Ensemble methods combine multiple models to produce predictions that are more accurate and robust than any individual model. Random forests aggregate many decision trees trained on bootstrapped subsets with random feature selection, while gradient boosting sequentially builds trees that correct the errors of previous trees.",
        ],
        "detail": [
            "The bias-variance tradeoff characterizes the fundamental tension between underfitting and overfitting. High-bias models are too simple to capture the true data-generating process, while high-variance models are overly sensitive to training set noise, and optimal performance requires balancing these two sources of error.",
            "Batch normalization standardizes layer inputs by normalizing activations to have zero mean and unit variance within each minibatch. This technique accelerates training by reducing internal covariate shift, allows higher learning rates, and acts as a mild regularizer.",
            "Learning rate scheduling adjusts the optimization step size during training to balance exploration and convergence. Common strategies include step decay, cosine annealing, warmup followed by linear decay, and cyclical learning rates that oscillate between bounds.",
            "Loss functions measure the discrepancy between model predictions and ground truth targets, defining the optimization objective. Cross-entropy loss is standard for classification tasks, mean squared error for regression, and specialized losses like focal loss address class imbalance.",
            "Hyperparameter tuning searches the space of configuration settings — including learning rate, batch size, regularization strength, and architecture choices — to find the combination that maximizes validation performance. Methods range from grid search and random search to Bayesian optimization and population-based training.",
        ],
        "example": [
            "Training a spam classifier with logistic regression and L2 regularization demonstrates the interplay of feature engineering (bag-of-words representation), optimization (gradient descent on cross-entropy loss), and regularization (weight penalty to prevent overfitting to rare words).",
            "A random forest trained on tabular data, such as predicting house prices from features like square footage, location, and age, illustrates how ensemble methods reduce variance by averaging many decorrelated trees.",
            "Consider tuning a neural network: learning rate 0.001 with batch size 32 might converge to a loss of 0.15, while learning rate 0.01 with batch size 256 achieves 0.12 — demonstrating why systematic hyperparameter search matters.",
            "In a medical diagnostic model, k-fold cross-validation reveals that accuracy varies from 87% to 94% across folds, indicating high variance that would be masked by a single train-test split.",
        ],
        "conclusion": [
            "The fundamental concepts of gradient optimization, regularization, and evaluation methodology remain relevant regardless of whether the model is a simple linear classifier or a billion-parameter neural network.",
            "As machine learning matures, the emphasis shifts from novel algorithms to robust engineering practices including data management, experiment tracking, and systematic model evaluation.",
            "Understanding these fundamentals enables practitioners to diagnose model failures, make informed design decisions, and build systems that perform reliably in production environments.",
        ],
    },
    "software_engineering": {
        "intro": [
            "Software engineering applies systematic, disciplined approaches to the design, development, testing, and maintenance of software systems at scale.",
            "The practice of software engineering encompasses not only coding but also requirements analysis, architecture design, quality assurance, and team collaboration processes.",
            "Modern software engineering has been transformed by agile methodologies, DevOps practices, and the recognition that software must continuously evolve to remain valuable.",
            "Building reliable software systems requires mastering both technical skills like design patterns and testing, and process skills like code review and continuous integration.",
            "The growing complexity of software systems demands engineering practices that manage technical debt, ensure maintainability, and enable teams to deliver value rapidly and safely.",
        ],
        "core": [
            "Test-driven development reverses the traditional development workflow by writing failing tests before implementing the corresponding functionality. This practice ensures that every piece of code has explicit test coverage, drives cleaner interfaces through testability requirements, and provides a regression safety net for future changes.",
            "Continuous integration automatically builds and runs the full test suite whenever code changes are pushed to the shared repository. CI systems detect integration conflicts and regressions early, when they are cheapest to fix, and provide rapid feedback that keeps the main branch in a deployable state.",
            "Code review examines proposed changes before they are merged, serving multiple purposes: catching bugs that automated tests miss, enforcing coding standards and architectural consistency, spreading knowledge across the team, and mentoring junior developers through constructive feedback.",
            "Dependency injection decouples components by providing their dependencies externally through constructor parameters, method arguments, or framework-managed containers. This pattern improves testability by allowing mock objects to replace real dependencies, and promotes modularity by eliminating hardcoded coupling between components.",
            "Design patterns provide reusable solutions to recurring software design problems. The observer pattern enables loose coupling between event producers and consumers, the strategy pattern encapsulates interchangeable algorithms, and the factory pattern centralizes object creation logic.",
        ],
        "detail": [
            "Refactoring improves the internal structure of code without changing its external behavior to reduce technical debt and improve maintainability. Common refactoring techniques include extracting methods, renaming variables for clarity, breaking large classes into focused components, and replacing conditional logic with polymorphism.",
            "Version control systems like Git track every change to every file over time, enabling collaboration through branching and merging, historical investigation through blame and bisect, and disaster recovery through the complete commit history.",
            "API versioning strategies maintain backward compatibility while allowing service interfaces to evolve. URL-based versioning, header-based versioning, and semantic versioning each offer different trade-offs between simplicity, flexibility, and client migration burden.",
            "Load testing simulates concurrent users and realistic traffic patterns to identify performance bottlenecks before production deployment. Tools like k6 and Gatling generate synthetic load while measuring response times, throughput, and error rates under increasing concurrency levels.",
            "Monitoring and alerting systems collect metrics, logs, and traces from running applications to detect anomalies and enable rapid incident response. The three pillars of observability — metrics, logs, and distributed traces — provide complementary views into system behavior.",
        ],
        "example": [
            "In a microservices architecture, dependency injection allows a payment service to accept any implementation of a payment gateway interface, making it easy to swap between Stripe and PayPal providers or use a mock gateway during testing.",
            "A CI pipeline might include steps for linting, unit tests, integration tests, security scanning, and container image building — catching different classes of issues at each stage and providing feedback within minutes of each push.",
            "Consider a team practicing TDD for a shopping cart feature: they first write a test asserting that adding an item increases the cart total, then implement the minimal code to pass the test, then refactor to improve the design.",
            "When Netflix deploys a new version of a service, canary deployments route a small percentage of traffic to the new version while monitoring error rates and latency, automatically rolling back if the canary performs worse than the stable version.",
        ],
        "conclusion": [
            "Software engineering practices continue to evolve as teams adopt cloud-native architectures, embrace platform engineering, and integrate AI-assisted development tools into their workflows.",
            "The most impactful engineering practices are those that shorten feedback loops, reduce the cost of change, and enable teams to move quickly while maintaining high quality and system reliability.",
            "Investing in engineering fundamentals — testing, code review, monitoring, and documentation — pays compound returns as systems grow in scale and complexity.",
        ],
    },
    "information_retrieval": {
        "intro": [
            "Information retrieval is the science and engineering of finding relevant documents, passages, or data from large collections in response to user information needs.",
            "The field of information retrieval underpins search engines, recommendation systems, question answering, and any application that must match user queries to relevant content.",
            "Modern information retrieval combines classical probabilistic models with neural approaches to achieve both precision and recall across diverse query types and document collections.",
            "The challenge of information retrieval lies in bridging the vocabulary gap between how users express their needs and how relevant documents describe their content.",
            "As document collections grow to billions of items, information retrieval systems must balance retrieval effectiveness with computational efficiency and response latency.",
        ],
        "core": [
            "TF-IDF weights terms by their frequency within a document (term frequency) normalized by how common the term is across the entire corpus (inverse document frequency). This weighting scheme captures the intuition that terms appearing frequently in a document but rarely in others are likely to be important discriminators for that document.",
            "BM25 extends TF-IDF with document length normalization and term frequency saturation parameters. The saturation function ensures that additional occurrences of a term contribute diminishing relevance, while length normalization prevents long documents from being unfairly advantaged by containing more term matches.",
            "Vector space models represent documents and queries as high-dimensional vectors, where each dimension corresponds to a term or learned feature. Relevance is measured by the cosine similarity between the query vector and document vectors, with higher similarity indicating greater relevance.",
            "Inverted indexes map each term in the vocabulary to a posting list containing the document identifiers and positions where that term appears. This data structure enables efficient keyword retrieval by intersecting or unioning posting lists for query terms, avoiding the need to scan every document.",
            "Learning to rank trains machine learning models to combine multiple relevance signals — including BM25 scores, document quality features, click data, and semantic similarity — into a single ranking function that predicts relevance for query-document pairs.",
        ],
        "detail": [
            "Pseudo-relevance feedback automatically expands the original query with terms extracted from the top-ranked documents in an initial retrieval pass. This technique improves recall by adding related terms the user may not have included, though it can introduce topic drift if the initial results are off-target.",
            "Reciprocal rank fusion combines multiple independently generated ranked lists into a single ranking by summing the reciprocal ranks of each candidate across all lists. This simple aggregation method is remarkably effective and requires no training data or parameter tuning beyond the constant k.",
            "Semantic search uses dense vector embeddings from neural language models to find conceptually similar documents beyond exact keyword matching. Bi-encoder models independently embed queries and documents into a shared vector space, enabling approximate nearest neighbor search over precomputed document embeddings.",
            "Evaluation metrics like NDCG (Normalized Discounted Cumulative Gain) and MAP (Mean Average Precision) measure ranking quality with different granularity. NDCG handles graded relevance judgments and accounts for the position of relevant results, while MAP uses binary relevance and averages precision at each relevant result position.",
            "Hybrid retrieval combines sparse keyword matching (via inverted indexes and BM25) with dense semantic search (via vector embeddings and approximate nearest neighbor) to achieve complementary coverage. The sparse component excels at exact term matching, while the dense component captures semantic similarity for paraphrase and concept matching.",
        ],
        "example": [
            "Consider searching for 'how to fix a leaky faucet' — BM25 matches documents containing those exact keywords, while semantic search also retrieves documents about 'plumbing repair techniques' and 'stopping dripping taps' that use different vocabulary to describe the same topic.",
            "A search engine processing millions of queries per day relies on inverted indexes to reduce each query from a full collection scan to looking up a few posting lists and intersecting them, typically completing the retrieval in single-digit milliseconds.",
            "In a legal document retrieval system, learning to rank combines BM25 scores with document recency, citation count, court level, and semantic embeddings to produce rankings that reflect both textual relevance and legal authority.",
            "Reciprocal rank fusion in practice: if a document appears at rank 3 in the BM25 list and rank 7 in the semantic search list, its RRF score is 1/(k+3) + 1/(k+7), naturally favoring documents that appear in both lists.",
        ],
        "conclusion": [
            "Information retrieval continues to evolve with the integration of large language models for query understanding, passage retrieval, and answer generation, blurring the line between retrieval and reasoning.",
            "The combination of sparse and dense retrieval methods with learned reranking represents the current best practice for building effective search systems.",
            "Future directions include retrieval-augmented generation, multi-modal search across text and images, and personalized retrieval that adapts to individual user knowledge and preferences.",
        ],
    },
}

# Topic names in the order they appear (matches EVAL_QUERIES indexing)
TOPIC_NAMES = [
    "transformer_architecture",
    "reinforcement_learning",
    "natural_language_processing",
    "computer_vision",
    "database_systems",
    "distributed_systems",
    "cryptography",
    "operating_systems",
    "graph_algorithms",
    "machine_learning_fundamentals",
    "software_engineering",
    "information_retrieval",
]

# 10 eval queries with expected top results
EVAL_QUERIES = [
    {
        "query": "optimization algorithms for training deep neural networks",
        "relevant_clusters": {"transformer_architecture": 3, "reinforcement_learning": 2, "computer_vision": 1},
    },
    {
        "query": "learning representations from sequential data",
        "relevant_clusters": {"natural_language_processing": 3, "transformer_architecture": 2, "reinforcement_learning": 1},
    },
    {
        "query": "scalable indexing structures for fast similarity search",
        "relevant_clusters": {"database_systems": 3, "information_retrieval": 2, "distributed_systems": 1},
    },
    {
        "query": "security protocols for network communication",
        "relevant_clusters": {"cryptography": 3, "distributed_systems": 2, "operating_systems": 1},
    },
    {
        "query": "resource allocation and scheduling in computing systems",
        "relevant_clusters": {"operating_systems": 3, "distributed_systems": 2, "database_systems": 1},
    },
    {
        "query": "graph neural networks for structured prediction",
        "relevant_clusters": {"graph_algorithms": 3, "transformer_architecture": 2, "natural_language_processing": 1},
    },
    {
        "query": "feature extraction and pattern recognition in images",
        "relevant_clusters": {"computer_vision": 3, "natural_language_processing": 1, "transformer_architecture": 1},
    },
    {
        "query": "ranking models and relevance feedback in search",
        "relevant_clusters": {"information_retrieval": 3, "natural_language_processing": 2, "database_systems": 1},
    },
    {
        "query": "reward signals and policy optimization under uncertainty",
        "relevant_clusters": {"reinforcement_learning": 3, "graph_algorithms": 1, "operating_systems": 1},
    },
    {
        "query": "hash functions and authenticated data structures",
        "relevant_clusters": {"cryptography": 3, "database_systems": 2, "graph_algorithms": 1},
    },
]

# Zipf-weighted query distribution: some topics queried much more.
QUERY_WEIGHTS = [8, 6, 5, 4, 3, 3, 2, 2, 1, 1]


# ------------------------------------------------------------------ #
# Document generation                                                   #
# ------------------------------------------------------------------ #

# Transition phrases used to connect sentences within paragraphs
_TRANSITIONS = [
    "Furthermore, ",
    "In addition, ",
    "Moreover, ",
    "Building on this, ",
    "Importantly, ",
    "It is worth noting that ",
    "From a practical standpoint, ",
    "To elaborate, ",
    "Consequently, ",
    "In this context, ",
    "Extending this idea, ",
    "A key consideration is that ",
    "Notably, ",
    "Along these lines, ",
    "This connects to the broader point that ",
]


def _generate_document(topic_name: str, doc_index: int, rng: random.Random) -> str:
    """Generate a realistic multi-paragraph technical article for one document.

    Each article has 4-6 substantial paragraphs (4-7 sentences each),
    totaling approximately 500-900 words.  With CHUNK_SIZE=128 tokens this
    produces roughly 10-25 chunks per document through vstash's chunking
    pipeline.

    The doc_index (0-9) controls which sentences are used as the primary
    focus, which appear as secondary material, and how paragraphs are
    ordered — ensuring each document within a topic has distinct content
    and embeddings.

    Args:
        topic_name: Key into TOPIC_MATERIAL.
        doc_index: 0-9, determines which sentences and structure to use.
        rng: Seeded random instance for reproducibility.

    Returns:
        A multi-paragraph article as a single string with paragraph breaks.
    """
    material = TOPIC_MATERIAL[topic_name]
    intros = material["intro"]
    cores = material["core"]
    details = material["detail"]
    examples = material["example"]
    conclusions = material["conclusion"]

    human_topic = topic_name.replace("_", " ")
    n_core = len(cores)
    n_detail = len(details)
    n_examples = len(examples)

    # --- Rotate selections so each doc_index gets a unique focus ---
    # Primary core sentences (3-4, rotated by doc_index)
    primary_core_count = 3 + (doc_index % 2)
    primary_cores = [cores[(doc_index + j) % n_core] for j in range(primary_core_count)]

    # Secondary core sentences (remaining ones, used as supplementary)
    secondary_cores = [cores[(doc_index + primary_core_count + j) % n_core]
                       for j in range(min(2, n_core - primary_core_count))]

    # Detail sentences (2-3, rotated)
    detail_count = 2 + (doc_index % 2)
    detail_picks = [details[(doc_index * 2 + j) % n_detail] for j in range(detail_count)]

    # Example sentences (2, rotated)
    example_picks = [examples[(doc_index + j) % n_examples] for j in range(min(2, n_examples))]

    # Intro and conclusion (rotated)
    intro_primary = intros[doc_index % len(intros)]
    intro_secondary = intros[(doc_index + 2) % len(intros)]
    conclusion_primary = conclusions[doc_index % len(conclusions)]
    conclusion_secondary = conclusions[(doc_index + 1) % len(conclusions)]

    # --- Framing and connecting templates (varied by doc_index) ---
    framing_pool = [
        f"This article examines key aspects of {human_topic} that are particularly relevant to modern practice and ongoing research.",
        f"The following discussion covers several important dimensions of {human_topic} and explores their practical implications for practitioners.",
        f"In what follows, we explore fundamental concepts in {human_topic} that practitioners and researchers should understand deeply.",
        f"This overview highlights the most significant developments and techniques within {human_topic}, with attention to both theory and application.",
        f"We present a focused exploration of {human_topic}, covering both theoretical foundations and practical applications in real-world systems.",
        f"The material presented here provides a comprehensive introduction to {human_topic} suitable for both newcomers and experienced practitioners.",
        f"Understanding {human_topic} requires familiarity with several interconnected concepts that we examine in detail below.",
        f"This discussion synthesizes key ideas from {human_topic}, drawing connections between foundational theory and modern implementations.",
        f"The landscape of {human_topic} has evolved significantly in recent years, and this article captures the current state of knowledge.",
        f"We survey the essential building blocks of {human_topic}, explaining how they fit together to enable practical systems and applications.",
    ]

    connecting_pool = [
        f"These technical details illustrate the depth of engineering involved in {human_topic} and the careful design choices that practitioners must make.",
        f"Such considerations are critical when implementing {human_topic} solutions in production environments where reliability and performance matter.",
        f"The interplay between these factors shapes how {human_topic} systems behave under real-world conditions and varying workloads.",
        f"Practitioners working with {human_topic} must account for these nuances to achieve optimal results in their specific use cases.",
        f"This level of technical depth is necessary for making informed decisions when designing and deploying {human_topic} systems.",
        f"Understanding these mechanisms at a deeper level enables engineers to diagnose issues and optimize {human_topic} implementations effectively.",
        f"The relationship between these concepts reveals important design trade-offs that are central to {human_topic} in practice.",
        f"Mastering these aspects of {human_topic} provides the foundation needed to tackle more advanced problems and cutting-edge research.",
    ]

    elaboration_pool = [
        f"This example highlights how theoretical concepts in {human_topic} translate into practical outcomes that affect real users and systems.",
        f"Real-world applications like this demonstrate why {human_topic} remains an active area of research and development across the industry.",
        f"Such practical scenarios underscore the importance of understanding {human_topic} fundamentals thoroughly before attempting to build production systems.",
        f"These applications show the broad impact of advances in {human_topic} across different domains and industries.",
        f"Concrete examples like these make the abstract principles of {human_topic} more tangible and help practitioners apply them to new problems.",
        f"The practical significance of this work extends beyond academic interest, as {human_topic} directly impacts the systems that millions of people rely on daily.",
    ]

    bridge_pool = [
        f"Building on these foundational ideas, we can now examine more specialized aspects of {human_topic} that arise in practice.",
        f"With this background established, the following section delves deeper into the technical mechanisms that make {human_topic} effective.",
        f"These core principles set the stage for understanding the more nuanced challenges and solutions in {human_topic}.",
        f"Having covered the essential concepts, we turn to specific techniques and their implementation details in {human_topic}.",
        f"The concepts above provide the necessary context for exploring the advanced topics in {human_topic} discussed below.",
    ]

    # --- Assemble paragraphs ---
    paragraphs: list[str] = []

    # ---- Paragraph 1: Introduction (3-4 sentences) ----
    p1 = [intro_primary, intro_secondary, framing_pool[doc_index % len(framing_pool)]]
    if doc_index % 3 == 0:
        p1.append(framing_pool[(doc_index + 5) % len(framing_pool)])
    paragraphs.append(" ".join(p1))

    # ---- Paragraph 2: Primary core concepts (4-5 sentences) ----
    p2 = [primary_cores[0]]
    t = rng.choice(_TRANSITIONS)
    p2.append(f"{t}{primary_cores[1][0].lower()}{primary_cores[1][1:]}")
    if len(primary_cores) > 2:
        p2.append(primary_cores[2])
    p2.append(bridge_pool[doc_index % len(bridge_pool)])
    paragraphs.append(" ".join(p2))

    # ---- Paragraph 3: More core + detail (5-6 sentences) ----
    p3 = []
    if len(primary_cores) > 3:
        p3.append(primary_cores[3])
    if secondary_cores:
        t2 = rng.choice(_TRANSITIONS)
        p3.append(f"{t2}{secondary_cores[0][0].lower()}{secondary_cores[0][1:]}")
    p3.append(detail_picks[0])
    if len(secondary_cores) > 1:
        p3.append(secondary_cores[1])
    p3.append(connecting_pool[doc_index % len(connecting_pool)])
    paragraphs.append(" ".join(p3))

    # ---- Paragraph 4: Detail depth (4-5 sentences) ----
    p4 = [detail_picks[1 % len(detail_picks)]]
    if len(detail_picks) > 2:
        t3 = rng.choice(_TRANSITIONS)
        p4.append(f"{t3}{detail_picks[2][0].lower()}{detail_picks[2][1:]}")
    else:
        p4.append(connecting_pool[(doc_index + 3) % len(connecting_pool)])
    p4.append(connecting_pool[(doc_index + 5) % len(connecting_pool)])
    # Add a topic-specific synthesis sentence
    synthesis_pool = [
        f"Taken together, these techniques form a coherent framework that addresses the central challenges of {human_topic} from multiple angles.",
        f"The convergence of these approaches has significantly advanced the state of the art in {human_topic} over the past several years.",
        f"Each of these mechanisms contributes a distinct capability, and their combination enables the sophisticated systems that define modern {human_topic}.",
        f"The effectiveness of {human_topic} in practice depends on carefully balancing these different components and understanding their interactions.",
        f"Progress in {human_topic} has been driven by innovations across all of these dimensions, with each advance enabling further improvements in the others.",
    ]
    p4.append(synthesis_pool[doc_index % len(synthesis_pool)])
    paragraphs.append(" ".join(p4))

    # ---- Paragraph 5: Examples and applications (4-5 sentences) ----
    p5 = [example_picks[0]]
    p5.append(elaboration_pool[doc_index % len(elaboration_pool)])
    if len(example_picks) > 1:
        p5.append(f"Similarly, {example_picks[1][0].lower()}{example_picks[1][1:]}")
    p5.append(elaboration_pool[(doc_index + 3) % len(elaboration_pool)])
    paragraphs.append(" ".join(p5))

    # ---- Paragraph 6: Conclusion (3-4 sentences) ----
    p6 = [conclusion_primary, conclusion_secondary]
    closing_pool = [
        f"Continued research and practical experience will further refine our understanding and application of {human_topic} in the years ahead.",
        f"As the field advances, the principles discussed here will remain foundational to progress in {human_topic} and related disciplines.",
        f"The ongoing evolution of {human_topic} promises new capabilities and challenges for researchers and practitioners alike.",
        f"Looking ahead, {human_topic} will continue to be shaped by both theoretical insights and engineering innovation at the frontier of computing.",
        f"The future of {human_topic} lies at the intersection of fundamental research, engineering practice, and the evolving needs of real-world applications.",
    ]
    p6.append(closing_pool[doc_index % len(closing_pool)])
    if doc_index % 2 == 0:
        p6.append(closing_pool[(doc_index + 3) % len(closing_pool)])
    paragraphs.append(" ".join(p6))

    return "\n\n".join(paragraphs)


# ------------------------------------------------------------------ #
# Corpus generation                                                    #
# ------------------------------------------------------------------ #


def build_corpus(store: VstashStore) -> int:
    """Generate and ingest 120 synthetic documents (12 clusters x 10).

    Each document is a multi-paragraph technical article that is chunked
    through vstash's real chunking pipeline (chunk_text) before embedding
    and ingestion.

    Returns:
        Total number of chunks ingested.
    """
    total_chunks = 0
    rng = random.Random(42)

    for topic_name in TOPIC_NAMES:
        for doc_index in range(10):
            # Generate a multi-paragraph article
            doc_text = _generate_document(topic_name, doc_index, rng)

            # Use vstash's real chunking pipeline
            chunks = chunk_text(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunks:
                # Fallback: if chunking produces nothing, use the whole text
                chunks = [doc_text]

            # Embed each chunk
            embeddings = [embed_query(c, MODEL) for c in chunks]

            path = f"/corpus/{topic_name}/doc_{doc_index:02d}.md"
            title = f"{topic_name}_{doc_index:02d}"

            store.add_document(
                path=path,
                title=title,
                chunks=chunks,
                embeddings=embeddings,
                source_type="markdown",
            )
            total_chunks += len(chunks)

    return total_chunks


# ------------------------------------------------------------------ #
# NDCG computation                                                     #
# ------------------------------------------------------------------ #


def ndcg_at_k(
    result_clusters: list[str],
    relevant_clusters: dict[str, int],
    k: int = 5,
    docs_per_cluster: int = 10,
) -> float:
    """Compute NDCG@k using graded relevance from cluster relevance map.

    Args:
        result_clusters: Cluster names for each result position.
        relevant_clusters: Dict mapping cluster name → relevance grade (0-3).
        k: Cutoff position.
        docs_per_cluster: Number of documents per cluster (for ideal DCG).
    """
    dcg = 0.0
    for i, cluster in enumerate(result_clusters[:k]):
        rel = float(relevant_clusters.get(cluster, 0))
        dcg += rel / math.log2(i + 2)

    # Ideal: each cluster can contribute up to docs_per_cluster results,
    # all at the same relevance grade.  Sort all available grades descending.
    ideal_rels: list[float] = []
    for cluster, grade in relevant_clusters.items():
        ideal_rels.extend([float(grade)] * docs_per_cluster)
    ideal_rels.sort(reverse=True)
    ideal_rels = ideal_rels[:k]
    while len(ideal_rels) < k:
        ideal_rels.append(0.0)
    ideal_dcg = sum(ideal_rels[i] / math.log2(i + 2) for i in range(k))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def get_cluster_for_path(path: str) -> str:
    """Extract cluster name from document path."""
    # /corpus/transformer_architecture/doc_00.md -> transformer_architecture
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


# ------------------------------------------------------------------ #
# Usage simulation                                                     #
# ------------------------------------------------------------------ #


def simulate_usage_round(
    store: VstashStore,
    query_embeddings: dict[str, list[float]],
    round_num: int,
    top_k: int = 10,
) -> int:
    """Simulate one round of non-uniform user queries.  Returns access count."""
    rng = random.Random(42 + round_num)

    # Build weighted query list
    weighted_queries = []
    for eq, weight in zip(EVAL_QUERIES, QUERY_WEIGHTS):
        weighted_queries.extend([eq] * weight)

    n_queries = min(5 + round_num * 2, len(weighted_queries))
    selected = rng.sample(weighted_queries, k=min(n_queries, len(weighted_queries)))

    total_tracked = 0
    for eq in selected:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q],
            query_text=q,
            top_k=top_k,
            scoring=None,  # Usage queries build history, don't use scoring
        )

        chunk_ids = []
        for r in results[:top_k]:
            row = store._conn.execute(
                "SELECT id FROM chunks WHERE text = ? LIMIT 1", [r.text]
            ).fetchone()
            if row:
                chunk_ids.append(row["id"])
        if chunk_ids:
            store.track_access(chunk_ids)
            total_tracked += len(chunk_ids)

    return total_tracked


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #


@dataclass
class RoundResult:
    round_num: int
    baseline_ndcg: float
    fixed_ndcg: float
    adaptive_ndcg: float
    gamma: float
    effective_beta: float
    total_accesses: int
    max_access: int
    mean_access: float


@dataclass
class ExperimentResult:
    rounds: list[RoundResult]
    total_rounds: int
    corpus_docs: int
    corpus_chunks: int
    n_queries: int
    fixed_crossover: int | None
    adaptive_crossover: int | None
    fixed_degradation_rounds: int
    adaptive_degradation_rounds: int


# ------------------------------------------------------------------ #
# Experiment                                                           #
# ------------------------------------------------------------------ #


def run_experiment(
    store: VstashStore,
    n_rounds: int = 30,
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    decay_lambda: float = 0.10,
) -> ExperimentResult:
    """Run the adaptive vs fixed scoring experiment."""

    fixed_scoring = ScoringConfig(
        enabled=True, alpha=alpha, beta=beta,
        decay_lambda=decay_lambda, over_fetch=50, track_access=False,
    )
    # Baseline uses same over_fetch pool but gamma=0 (pure RRF + dedup)
    baseline_scoring = ScoringConfig(
        enabled=True, alpha=alpha, beta=beta,
        decay_lambda=decay_lambda, over_fetch=50, track_access=False,
    )

    # Pre-compute query embeddings
    query_embeddings = {}
    for eq in EVAL_QUERIES:
        query_embeddings[eq["query"]] = embed_query(eq["query"], MODEL)

    # Baseline: RRF only with same candidate pool (gamma forced to 0)
    baseline_ndcgs = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        results = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=baseline_scoring,
            _gamma_override=0.0,  # force pure RRF
        )
        clusters = [get_cluster_for_path(r.path) for r in results]
        baseline_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
    avg_baseline = sum(baseline_ndcgs) / len(baseline_ndcgs)

    # Reset access history
    store._conn.execute("UPDATE chunks SET access_count = 0, last_accessed_at = NULL")
    store._conn.commit()

    rounds: list[RoundResult] = []
    cumulative_accesses = 0

    for round_num in range(n_rounds):
        # Phase A: simulate usage
        usage = simulate_usage_round(store, query_embeddings, round_num, top_k)
        cumulative_accesses += usage

        # Read gamma from the store (adaptive)
        gamma = store.scoring_maturity()
        effective_beta = beta * gamma

        # Access stats
        row = store._conn.execute(
            "SELECT AVG(access_count) as mean, MAX(access_count) as mx "
            "FROM chunks WHERE access_count > 0"
        ).fetchone()
        mean_acc = float(row["mean"]) if row and row["mean"] else 0.0
        max_acc = int(row["mx"]) if row and row["mx"] else 0

        # Phase B: evaluate with FIXED scoring (gamma=1.0 always — old behavior)
        fixed_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                _gamma_override=1.0,  # force full beta — simulates pre-v0.7
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            fixed_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        avg_fixed = sum(fixed_ndcgs) / len(fixed_ndcgs)

        # Phase C: evaluate with ADAPTIVE scoring (gamma computed from data)
        adaptive_ndcgs = []
        for eq in EVAL_QUERIES:
            q = eq["query"]
            results = store.search(
                query_embedding=query_embeddings[q], query_text=q,
                top_k=top_k, scoring=fixed_scoring,
                # _gamma_override not set -> uses real scoring_maturity()
            )
            clusters = [get_cluster_for_path(r.path) for r in results]
            adaptive_ndcgs.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        avg_adaptive = sum(adaptive_ndcgs) / len(adaptive_ndcgs)

        rounds.append(RoundResult(
            round_num=round_num + 1,
            baseline_ndcg=avg_baseline,
            fixed_ndcg=avg_fixed,
            adaptive_ndcg=avg_adaptive,
            gamma=gamma,
            effective_beta=effective_beta,
            total_accesses=cumulative_accesses,
            max_access=max_acc,
            mean_access=round(mean_acc, 2),
        ))

    # Phase D: inject a strong outlier to simulate a "power user" pattern
    # (one topic queried 50x more than others — like a researcher's focus area)
    print("\n  --- Injecting power-user pattern (50x boost on transformers) ---\n")
    transformer_chunks = store._conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
        "WHERE d.path LIKE '%transformer%'"
    ).fetchall()
    for row in transformer_chunks:
        store._conn.execute(
            "UPDATE chunks SET access_count = access_count + 50 WHERE id = ?",
            [row["id"]],
        )
    store._conn.commit()

    # Re-evaluate after power-user boost
    gamma_post = store.scoring_maturity()
    row = store._conn.execute(
        "SELECT AVG(access_count) as mean, MAX(access_count) as mx "
        "FROM chunks WHERE access_count > 0"
    ).fetchone()
    mean_post = float(row["mean"]) if row["mean"] else 0
    max_post = int(row["mx"]) if row["mx"] else 0

    fixed_post = []
    adaptive_post = []
    for eq in EVAL_QUERIES:
        q = eq["query"]
        # Fixed
        r_fixed = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=fixed_scoring, _gamma_override=1.0,
        )
        clusters = [get_cluster_for_path(r.path) for r in r_fixed]
        fixed_post.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))
        # Adaptive
        r_adapt = store.search(
            query_embedding=query_embeddings[q], query_text=q,
            top_k=top_k, scoring=fixed_scoring,
        )
        clusters = [get_cluster_for_path(r.path) for r in r_adapt]
        adaptive_post.append(ndcg_at_k(clusters, eq["relevant_clusters"], k=top_k))

    avg_fixed_post = sum(fixed_post) / len(fixed_post)
    avg_adaptive_post = sum(adaptive_post) / len(adaptive_post)
    eff_beta_post = beta * gamma_post

    print(f"  After power-user injection:")
    print(f"    gamma = {gamma_post:.3f}, effective beta = {eff_beta_post:.3f}")
    print(f"    max/mean = {max_post}/{mean_post:.1f} (ratio = {max_post/mean_post:.1f}x)")
    print(f"    Baseline:  {avg_baseline:.4f}")
    print(f"    Fixed beta:   {avg_fixed_post:.4f} ({(avg_fixed_post-avg_baseline)/avg_baseline*100:+.1f}%)")
    print(f"    Adaptive:  {avg_adaptive_post:.4f} ({(avg_adaptive_post-avg_baseline)/avg_baseline*100:+.1f}%)")

    # Analyze crossover and degradation
    def find_crossover(values: list[float], baseline: float) -> int | None:
        consec = 0
        for i, v in enumerate(values):
            if v > baseline:
                consec += 1
                if consec >= 3:
                    return i - 1  # first of 3 consecutive
            else:
                consec = 0
        return None

    fixed_crossover = find_crossover([r.fixed_ndcg for r in rounds], avg_baseline)
    adaptive_crossover = find_crossover([r.adaptive_ndcg for r in rounds], avg_baseline)

    fixed_degrade = sum(1 for r in rounds if r.fixed_ndcg < avg_baseline)
    adaptive_degrade = sum(1 for r in rounds if r.adaptive_ndcg < avg_baseline)

    return ExperimentResult(
        rounds=rounds,
        total_rounds=n_rounds,
        corpus_docs=len(TOPIC_NAMES) * 10,
        corpus_chunks=store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        n_queries=len(EVAL_QUERIES),
        fixed_crossover=fixed_crossover,
        adaptive_crossover=adaptive_crossover,
        fixed_degradation_rounds=fixed_degrade,
        adaptive_degradation_rounds=adaptive_degrade,
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Cold Start Experiment")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--output", type=str, default="experiments/results/adaptive_cold_start.json")
    args = parser.parse_args()

    dim = get_embedding_dim(MODEL)
    tmp_dir = tempfile.mkdtemp(prefix="vstash_adaptive_")
    db_path = str(Path(tmp_dir) / "adaptive.db")
    store = VstashStore(db_path, embedding_dim=dim)

    print(f"DB: {db_path}")
    print("Generating 120-document corpus (12 clusters x 10 docs)...")
    print("Each doc is a multi-paragraph article chunked via vstash pipeline...")
    t0 = time.time()
    n_chunks = build_corpus(store)
    t1 = time.time()
    print(f"Ingested: 120 docs, {n_chunks} chunks in {t1 - t0:.1f}s\n")

    sep = "=" * 90
    print(f"{sep}")
    print("  ADAPTIVE COLD START: fixed beta vs adaptive beta (gamma-gated)")
    print(f"{sep}\n")

    t0 = time.time()
    result = run_experiment(store, n_rounds=args.rounds)
    elapsed = time.time() - t0
    print(f"  Experiment completed in {elapsed:.1f}s\n")

    baseline = result.rounds[0].baseline_ndcg
    print(f"  Baseline NDCG@5 (RRF only): {baseline:.4f}")
    print(f"  Corpus: {result.corpus_docs} docs, {result.corpus_chunks} chunks")
    print(f"  Queries: {result.n_queries}\n")

    print(f"  {'Rnd':>4} {'Baseline':>9} {'Fixed':>8} {'d%':>7} {'Adaptive':>9} {'d%':>7} "
          f"{'gamma':>5} {'eff_b':>6} {'max/mean':>10} {'Accesses':>9}")
    print(f"  {'-' * 85}")

    for r in result.rounds:
        fixed_d = ((r.fixed_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        adapt_d = ((r.adaptive_ndcg - baseline) / baseline * 100) if baseline > 0 else 0
        ratio_str = f"{r.max_access}/{r.mean_access:.1f}" if r.mean_access > 0 else "--"
        print(
            f"  {r.round_num:>4} {baseline:>9.4f} {r.fixed_ndcg:>8.4f} {fixed_d:>+6.1f}% "
            f"{r.adaptive_ndcg:>9.4f} {adapt_d:>+6.1f}% {r.gamma:>5.2f} {r.effective_beta:>6.3f} "
            f"{ratio_str:>10} {r.total_accesses:>9}"
        )

    print(f"\n{sep}")
    print(f"  Fixed beta:    crossover={result.fixed_crossover or 'never'}, "
          f"degradation={result.fixed_degradation_rounds}/{result.total_rounds} rounds")
    print(f"  Adaptive gamma: crossover={result.adaptive_crossover or 'never'}, "
          f"degradation={result.adaptive_degradation_rounds}/{result.total_rounds} rounds")

    if result.adaptive_degradation_rounds < result.fixed_degradation_rounds:
        improvement = result.fixed_degradation_rounds - result.adaptive_degradation_rounds
        print(f"\n  * Adaptive scoring eliminates {improvement} rounds of degradation")
    print(f"{sep}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"  Results saved to: {output_path}")

    store.close()


if __name__ == "__main__":
    main()
