## 1. Basic identification

### What is the complete title?

### Who are the authors?

### Where and when was it published?

### Is it a peer-reviewed conference paper, journal article, survey, or preprint?

### Is the venue reputable for distributed systems, networking, HPC, or machine learning systems?

### Is this primary research or a summary of other research?

## 2. Problem and motivation

### What specific problem does the paper address?

### Why is this problem important for distributed LLM training?

### What limitation in previous systems motivates the work?

### Is the main constraint computation, GPU memory, communication, synchronization, scheduling, reliability, or some combination?

### Under what training environment does the problem occur?

### What would happen if the problem were left unresolved?

Try to summarize the problem in one sentence:

> The paper addresses ________, which causes ________ when training ________ across ________.

## 3. Main contribution

These questions are explicitly emphasized by the assignment:

### What is the paper’s main technical contribution?

### Does it introduce an architecture, algorithm, scheduler, optimization, analytical model, or framework?

### What are its secondary contributions?

### What is technically new about the contribution?

### Is the contribution unique compared with previous work?

### Which parts are new, and which parts combine existing techniques?

### What assumptions make the proposed approach possible?

You should be able to explain the contribution without copying the abstract.

## 4. Distributed architecture

### What are the participating components?

### How many GPUs, nodes, clusters, or data centers are involved?

### What is placed on each device or node?

### Is there a centralized controller, parameter server, scheduler, or coordinator?

### How is the model divided?

### How is the training data divided?

### How are tasks assigned to workers?

### Does the architecture assume homogeneous or heterogeneous hardware?

### Does it operate within one server, across a cluster, or across data centers?

### What network topology does it assume?

Draw a simple architecture diagram for yourself. If you cannot draw the system, you probably do not yet understand it sufficiently.

## 5. Parallelism strategy

### Which forms of parallelism are used?

  ### Data parallelism
  ### Tensor parallelism
  ### Pipeline parallelism
  ### Expert parallelism
  ### Sequence parallelism
  ### Hybrid parallelism

### What exactly is partitioned in each strategy?

### Why did the authors choose this combination?

### How are microbatches scheduled?

### Does the approach create pipeline bubbles?

### Does it introduce stale parameters or gradients?

### How does it handle load imbalance?

### Can the parallelism configuration change automatically?

## 6. Communication behavior

This is essential for your topic.

### What information moves between devices or nodes?

### When does communication occur during a training iteration?

### Which collective operations are used?

  ### All-Reduce
  ### All-Gather
  ### Reduce-Scatter
  ### All-to-All
  ### Point-to-point transfers

### How frequently are gradients synchronized?

### Does the system use synchronous or asynchronous communication?

### Can communication overlap with computation?

### What bandwidth and latency assumptions are made?

### Is communication occurring within a server, between servers, or across data centers?

### What becomes the communication bottleneck?

### How does communication volume change as the number of workers grows?

### What technique does the paper use to reduce communication overhead?

You should be able to trace one complete training iteration and identify every major communication point.

## 7. Memory management

### What must each GPU store?

  ### Parameters
  ### Gradients
  ### Optimizer states
  ### Activations

### Which objects are replicated?

### Which objects are sharded?

### Does the system use activation checkpointing or recomputation?

### Does it offload anything to CPU memory or storage?

### Does reducing memory use increase communication or computation?

### What is the largest model that the evaluated hardware can support?

### What memory tradeoff does the proposed method introduce?

## 8. Algorithm or system operation

### What are the major steps of the proposed method?

### What inputs does the method require?

### What decisions does it make?

### What objective does it optimize?

### Does it use a heuristic, search procedure, mathematical optimization, or learned policy?

### Is there pseudocode that you can explain?

### What is the computational cost of the algorithm itself?

### Does it operate before training, during training, or both?

### Can it adapt to runtime changes?

Try explaining the method as five to eight ordered steps.

## 9. Experimental methodology

### What models are evaluated?

### What model sizes are used?

### Is the evaluation based on complete training, a limited number of iterations, simulation, or analytical prediction?

### What datasets are used?

### What hardware is used?

### How many GPUs and nodes are used?

### What interconnect or network is used?

### What software frameworks are used?

### What baselines are selected?

### Are those baselines appropriate and competitive?

### What metrics are reported?

### Are experiments repeated?

### Are error bars or variability reported?

### Is the implementation or artifact publicly available?

### Are there enough details to reproduce the experiments?

## 10. Results

### What are the three most important results?

### Which table or figure supports each result?

### Against which baseline is each improvement measured?

### Is the result an average, maximum, or result from one configuration?

### Under what model, GPU count, and network conditions was it obtained?

### Does the method improve throughput, memory capacity, utilization, or training time?

### Does it preserve convergence and model quality?

### Does it scale as more devices are added?

### At what point does performance stop scaling efficiently?

### Do the results support all the authors’ claims?

### Are any results weaker than the paper’s wording suggests?

Record results with their conditions:

> The method improves ________ by ________ relative to ________ when evaluated on ________ using ________.

## 11. Strengths

The assignment requires you to identify strengths and weaknesses.

### What does the paper do particularly well?

### Is the problem clearly defined?

### Is the contribution technically meaningful?

### Is the design well justified?

### Is the evaluation broad and convincing?

### Are strong baselines included?

### Does the system solve a realistic distributed-systems problem?

### Are the assumptions clearly stated?

### Is the work reproducible?

### Does the paper provide useful insights beyond its reported speedup?

## 12. Weaknesses and justified criticism

### Which assumptions may be unrealistic?

### Is the evaluation environment too small or specialized?

### Are important models, hardware types, or network conditions missing?

### Are the comparisons fair?

### Are any important baselines omitted?

### Does the approach shift cost from memory to communication or computation?

### Does it require expensive hardware or high-bandwidth networking?

### Does it increase implementation or operational complexity?

### How would failures, stragglers, or network congestion affect it?

### Would it work on heterogeneous GPUs?

### Are simulation results presented as if they were physical measurements?

### Are convergence and model quality adequately evaluated?

### Do the citations actually support the claims associated with them?

### Can its conclusions be generalized beyond the tested environment?

Every criticism should have a reason:

> The evaluation does not establish cross-data-center scalability because all experiments use ________.

## 13. Scope and applicability

### Under what conditions should this approach be used?

### For which models and cluster sizes is it appropriate?

### What hardware or network does it require?

### When would another approach be preferable?

### Does it address pretraining, fine-tuning, or both?

### Does it apply to dense models, mixture-of-experts models, or both?

### Would it work across data centers?

### What happens when the model or cluster becomes much larger?

## 14. Relationship to the other papers

### What problem do the papers have in common?

### How do their system assumptions differ?

### Do they operate at the same layer of the training stack?

### Does one paper build upon ideas introduced by another?

### Do they optimize the same metric?

### Can their reported performance values be compared fairly?

### Are their techniques complementary or mutually exclusive?

### Could their techniques be combined?

### Which paper offers the strongest evidence?

### Which paper has the narrowest applicability?

### Which approach would you select for a specific cluster, and why?

## 15. Open problems

### Which challenges remain after applying the proposed method?

### What future work do the authors identify?

### What future work do you identify independently?

### Does the system address fault tolerance?

### Can it adapt to changing network conditions?

### Can it accommodate heterogeneous hardware?

### Does it consider energy or financial cost?

### What prevents it from scaling to substantially larger models or clusters?

### What question should a subsequent research paper investigate?

## 16. Presentation and writing

The assignment also asks you to examine how published technical papers are written.

### How is the paper organized?

### How do the authors state the problem?

### How are the contributions presented?

### Which diagrams make the architecture understandable?

### Are figures and tables clearly explained?

### How do the authors connect claims to experimental results?

### How is related work distinguished from the new contribution?

### Does the conclusion accurately reflect the evidence?

### What writing or presentation techniques could you apply in your own words?

## Minimum readiness test

Before drafting, you should be able to give a five-minute explanation of each primary paper covering:

01. The problem  
02. The main contribution  
03. The architecture  
04. The parallelism method  
05. The communication pattern  
06. The evaluation environment  
07. The major results  
08. Two strengths  
09. Two limitations  
10. Its relationship to the other papers  

If you cannot answer one of those without reopening the abstract, reread the relevant section before beginning the term paper.
