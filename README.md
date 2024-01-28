# Accelerating Distributed ML Training via Selective Synchronization

**Code for SelSync presented at IEEE International Conference on Cluster Computing (CLUSTER), 2023, Santa Fe, New Mexico, USA.**

_In distributed training, deep neural networks (DNNs) are launched over multiple workers concurrently and aggregate their local updates on each step in bulk-synchronous parallel (BSP) training. 
However, BSP does not linearly scale-out due to high communication cost of aggregation. 
To mitigate this overhead, alternatives like Federated Averaging (FedAvg) and Stale-Synchronous Parallel (SSP) either reduce synchronization frequency or eliminate it altogether, usually at the cost of lower final accuracy. 
In this work, we present SelSync, a practical, low-overhead method for DNN training that dynamically chooses to incur or avoid communication at each step either by calling the aggregation op or applying local updates based on their significance. 
We propose various optimizations as part of SelSync to improve convergence in the context of semi-synchronous training. 
Our system converges to the same or better accuracy than BSP while reducing training time by up to 14×._

**ACCESS LINKS**
- [Link1](https://ieeexplore.ieee.org/document/10319965)
- [Link2](https://sahiltyagi.academicwebsite.com/publications/23152-accelerating-distributed-ml-training-via-selective-synchronization)

**RUNNING**

- Execution scripts available in ```scripts``` directory.
- Data-injection currently implemented on CIFAR/100; ImageNet to be added in the future.

**CITATION**
- **_Bibtex_**: @article{Tyagi2023AcceleratingDM,
  title={Accelerating Distributed ML Training via Selective Synchronization},
  author={Sahil Tyagi and Martin Swany},
  journal={2023 IEEE International Conference on Cluster Computing (CLUSTER)},
  year={2023},
  pages={1-12}}
  
- **_MLA_**: Tyagi, Sahil and Martin Swany. “Accelerating Distributed ML Training via Selective Synchronization.” 2023 IEEE International Conference on Cluster Computing (CLUSTER) (2023): 1-12.

- **_APA_**: Tyagi, S., & Swany, M. (2023). Accelerating Distributed ML Training via Selective Synchronization. 2023 IEEE International Conference on Cluster Computing (CLUSTER), 1-12.

- **_Chicago_**: Tyagi, Sahil and Martin Swany. “Accelerating Distributed ML Training via Selective Synchronization.” 2023 IEEE International Conference on Cluster Computing (CLUSTER) (2023): 1-12.