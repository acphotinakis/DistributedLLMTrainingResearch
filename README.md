# Distributed Training of Large Language Models: Architectures, Parallelism, and Communication Challenges

A literature-based research report on why training large language models (LLMs) requires distributed computing, and how training work, model state, and communication are organized across GPUs, nodes, and data centers.

This is a synthesis and critique of published work from a distributed-systems perspective. It does not report original experiments.

## Research question

**How do distributed LLM training architectures combine parallelism and communication techniques to scale training, and what performance, memory, synchronization, and reliability tradeoffs result?**

Supporting questions:

1. Why can an LLM no longer be trained efficiently on a single accelerator or machine?
2. What is partitioned in data, tensor, pipeline, and hybrid parallelism?
3. What communication operations does each strategy require?
4. Where do latency, limited bandwidth, synchronization, pipeline bubbles, and stragglers reduce performance?
5. How do sharding, checkpointing, recomputation, and offloading interact with parallelism?
6. How do cluster topology and cross-data-center links affect the best training strategy?
7. Which assumptions and limitations keep the proposed systems from generalizing to other models, hardware, or networks?

## Scope

**In scope**

- Training across multiple GPUs, nodes, or data centers
- Data, tensor, pipeline, expert, and hybrid parallelism
- Model-state and optimizer-state sharding
- Gradient and activation communication
- Collectives: All-Reduce, All-Gather, Reduce-Scatter
- Communication and computation overlap
- Pipeline scheduling, bubbles, load imbalance, and device placement
- Memory capacity and memory-performance tradeoffs
- Scalability, utilization, reliability, and fault tolerance

**Out of scope, except where it clarifies a training issue:** inference, fine-tuning, federated learning, recommendation systems, multimodal models, privacy, and model quality.

## Repository layout

```
.
├── README.md
└── artifacts/
    ├── README.md        # Project background and working notes
    ├── overview/        # Term paper draft/overview (PDF)
    ├── notes/           # Per-paper structured reading notes
    └── papers/          # Source papers (PDF)
```

## Sources

### Primary research

- **Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization** (EuroSys 2025). It co-optimizes data, tensor, and pipeline parallelism with checkpointing, redundancy elimination, and offloading.
- **Parallel Gradient Computation and Synchronization: Enhancing the Efficiency of Distributed Training for LLMs** (IEEE Transactions on Network Science and Engineering). It covers gradient synchronization for expert models trained across multiple data centers under limited bandwidth.

### Background surveys

- **Distributed Training of Large Language Models: A Survey**
- **Efficient Training of Large Language Models on Distributed Infrastructures: A Survey**

The surveys provide terminology, taxonomy, and pointers to primary work. They are not treated as experimental evidence.

## Comparison framework

Each primary paper is analyzed along the same dimensions:

| Category | Question |
| --- | --- |
| Problem | Which distributed-training bottleneck is addressed? |
| Contribution | What architecture, algorithm, scheduler, or optimization is introduced? |
| Parallelism | Which forms are used or combined? |
| Distribution | What is distributed across devices, nodes, or data centers? |
| Communication | What is exchanged, with which operations and network assumptions? |
| Memory | What state or activations are stored, sharded, recomputed, or offloaded? |
| Evaluation | Which models, hardware, networks, baselines, and metrics are used? |
| Results | What improvements are reported, and under what conditions? |
| Strengths | What is distinctive or convincingly demonstrated? |
| Limitations | Which assumptions or evaluation gaps weaken the claims? |
| Applicability | When is the method preferable to alternatives? |

Results from different environments (model size, GPU count, topology, bandwidth, datasets, baselines, metrics) are not compared directly without qualification.

## Status

- Source collection and structured reading notes are in progress.
- A third primary systems paper, such as *PipeDream* or *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, is planned to broaden the comparison.
