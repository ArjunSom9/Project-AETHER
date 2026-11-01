# NVIDIA H100 Tensor Core GPU Architecture

**Exceptional Performance, Scalability, and Security for the Data Center**  
*Includes final GPU / memory clocks and final TFLOPS performance specs.*  
**Version:** 1.04  

---

## Table of Contents

- [Introduction](#introduction) (Page 6)
- [NVIDIA H100 Tensor Core GPU Overview](#nvidia-h100-tensor-core-gpu-overview) (Page 8)
- [NVIDIA H100 GPU Key Feature Summary](#nvidia-h100-gpu-key-feature-summary) (Page 11)
- [NVIDIA GPU-Accelerated Data Centers](#nvidia-gpu-accelerated-data-centers) (Page 14)
  - [H100 SXM5 GPU](#h100-sxm5-gpu) (Page 15)
  - [H100 PCIe Gen 5 GPU](#h100-pcie-gen-5-gpu) (Page 15)
  - [DGX H100 and DGX SuperPOD](#dgx-h100-and-dgx-superpod) (Page 15)
  - [HGX H100](#hgx-h100) (Page 16)
  - [H100 CNX Converged Accelerator](#h100-cnx-converged-accelerator) (Page 16)
- [NVIDIA H100 GPU Architecture In-Depth](#nvidia-h100-gpu-architecture-in-depth) (Page 17)
  - [H100 SM Architecture](#h100-sm-architecture) (Page 19)
  - [H100 SM Key Feature Summary](#h100-sm-key-feature-summary) (Page 22)
  - [H100 Tensor Core Architecture](#h100-tensor-core-architecture) (Page 22)
  - [Hopper FP8 Data Format](#hopper-fp8-data-format) (Page 23)
  - [New DPX Instructions for Accelerated Dynamic Programming](#new-dpx-instructions-for-accelerated-dynamic-programming) (Page 27)
  - [Combined L1 Data Cache and Shared Memory](#combined-l1-data-cache-and-shared-memory) (Page 27)
  - [H100 Compute Performance Summary](#h100-compute-performance-summary) (Page 28)
- [H100 GPU Hierarchy and Asynchrony Improvements](#h100-gpu-hierarchy-and-asynchrony-improvements) (Page 29)
  - [Thread Block Clusters](#thread-block-clusters) (Page 29)
  - [Distributed Shared Memory](#distributed-shared-memory) (Page 30)
  - [Asynchronous Execution](#asynchronous-execution) (Page 31)
  - [Tensor Memory Accelerator (TMA)](#tensor-memory-accelerator-tma) (Page 32)
  - [Asynchronous Transaction Barrier](#asynchronous-transaction-barrier) (Page 34)
- [H100 HBM and L2 Cache Memory Architectures](#h100-hbm-and-l2-cache-memory-architectures) (Page 36)
  - [H100 HBM3 and HBM2e DRAM Subsystems](#h100-hbm3-and-hbm2e-dram-subsystems) (Page 36)
  - [H100 L2 Cache](#h100-l2-cache) (Page 37)
  - [Memory Subsystem RAS Features](#memory-subsystem-ras-features) (Page 37)
    - [ECC Memory Resiliency](#ecc-memory-resiliency) (Page 38)
    - [Memory Row Remapping](#memory-row-remapping) (Page 38)
- [Compute Capability](#compute-capability) (Page 41)
- [Second-Generation Secure MIG](#second-generation-secure-mig) (Page 42)
  - [MIG Technology Review](#mig-technology-review) (Page 42)
  - [H100 MIG Enhancements](#h100-mig-enhancements) (Page 43)
- [Transformer Engine](#transformer-engine) (Page 44)
- [Fourth-Generation NVLink and NVLink Network](#fourth-generation-nvlink-and-nvlink-network) (Page 47)
  - [Third-Generation NVSwitch](#third-generation-nvswitch) (Page 47)
  - [New NVLink Switch System](#new-nvlink-switch-system) (Page 48)
  - [PCIe Gen 5](#pcie-gen-5) (Page 49)
- [Security Enhancements and Confidential Computing](#security-enhancements-and-confidential-computing) (Page 51)
  - [NVIDIA Confidential Computing](#nvidia-confidential-computing) (Page 51)
  - [Measure of Success](#measure-of-success) (Page 55)
  - [NVIDIA Confidential Computing Implementation Overview](#nvidia-confidential-computing-implementation-overview) (Page 56)
- [H100 Video / IO Features](#h100-video--io-features) (Page 58)
  - [NVDEC for DL](#nvdec-for-dl) (Page 58)
  - [NVJPG (JPEG) Decode](#nvjpg-jpeg-decode) (Page 59)
- [Appendix A - NVIDIA DGX - The Foundational Building Blocks of Data Center AI](#appendix-a---nvidia-dgx---the-foundational-building-blocks-of-data-center-al) (Page 60)
  - [NVIDIA DGX H100 - The World's Most Complete AI Platform](#nvidia-dgx-h100---the-worlds-most-complete-al-platform) (Page 60)
  - [DGX H100 Overview](#dgx-h100-overview) (Page 60)
  - [Unmatched Data Center Scalability](#unmatched-data-center-scalability) (Page 61)
  - [NVIDIA DGX H100 System Specifications](#nvidia-dgx-h100-system-specifications) (Page 62)
- [Appendix B - NVIDIA CUDA Platform Update](#appendix-b---nvidia-cuda-platform-update) (Page 63)
  - [High-Performance Libraries and Frameworks](#high-performance-libraries-and-frameworks) (Page 63)
  - [System Software](#system-software) (Page 64)
  - [Documentation and Training](#documentation-and-training) (Page 64)
  - [Languages and Compilers](#languages-and-compilers) (Page 65)
- [Appendix C - Genomics Acceleration using DPX Instructions](#appendix-c---genomics-acceleration-using-dpx-instructions) (Page 68)

---

## List of Figures

- **Figure 1.** Modern cloud datacenter workloads require NVIDIA GPU acceleration (Page 7)
- **Figure 2.** NVIDIA H100 GPU on new SXM5 Module (Page 8)
- **Figure 3.** H100 Enables Next-Generation AI and HPC Breakthroughs (Page 9)
- **Figure 4.** Grace Hopper Superchip (Page 10)
- **Figure 5.** New Technologies in Hopper H100 (Page 13)
- **Figure 6.** GH100 Full GPU with 144 SMs (Page 19)
- **Figure 7.** GH100 Streaming Multiprocessor (SM) (Page 21)
- **Figure 8.** H100 FP16 Tensor Core has 3x throughput compared to A100 FP16 Tensor Core (Page 23)
- **Figure 9.** New Hopper FP8 Precisions - 2x throughput and half the footprint of FP16/BF16 (Page 24)
- **Figure 10.** H100 FP8 Tensor Core 6x throughput compared to A100 FP16 Tensor Core (Page 24)
- **Figure 11.** H100 TF32, FP64, and INT8 Tensor Cores all have 3x throughput versus A100 (Page 25)
- **Figure 12.** DPX Instructions Accelerate Dynamic Programming (Page 27)
- **Figure 13.** H100 Compute Improvement Summary (Page 28)
- **Figure 14.** Thread Block Clusters and Grids with Clusters (Page 30)
- **Figure 15.** Thread Block to Thread Block data exchange (A100 vs H100 with Clusters) (Page 30)
- **Figure 16.** Cluster versus non-Cluster Performance Comparisons (Page 31)
- **Figure 17.** Asynchronous Execution Concurrency and Enhancements in Hopper (Page 32)
- **Figure 18.** TMA Address Generation via Copy Descriptor (Page 33)
- **Figure 19.** Asynchronous Memory Copy with TMA on H100 vs LDGSTS on A100 (Page 34)
- **Figure 20.** Asynchronous Barrier in A100 vs Asynchronous Transaction Barrier in H100 (Page 35)
- **Figure 21.** World's First HBM3 GPU Memory Architecture, 2x Delivered Bandwidth (Page 37)
- **Figure 22.** Example CSP MIG Configuration (Page 42)
- **Figure 23.** Secure MIG Example in Multi-Tenant Single GPU Configuration (Page 44)
- **Figure 24.** Transformers Model Sizes Increasing Exponentially with Different Use Cases (Page 45)
- **Figure 25.** Transformer Engine Conceptual Operation (Page 46)
- **Figure 26.** DGX A100 vs DGX H100 32-node, 256 GPU NVIDIA SuperPOD Comparison (Page 49)
- **Figure 27.** Confidential Computing Protects Multiple ISV Scenarios (Page 53)
- **Figure 28.** Confidential Computing for Different Use Cases (Page 54)
- **Figure 29.** Confidential Federated Learning (Page 55)
- **Figure 30.** NVIDIA CC Off vs CC On VM Isolation (Page 56)
- **Figure 31.** The NVIDIA CUDA platform and its ecosystem (Page 64)
- **Figure 32.** High-level language Frontends (Page 65)
- **Figure 33.** NVCC split compilation model and NVC++ unified compilation model (Page 66)
- **Figure 34.** Unified toolchain supports execution-space inference (Page 67)
- **Figure 35.** NVIDIA CLARA Parabricks Accelerated Framework (Page 69)
- **Figure 36.** Smith-Waterman Algorithm for Genome sequencing (Page 70)

---

## List of Tables

- **Table 1.** NVIDIA H100 Tensor Core GPU Performance Specs (Page 20)
- **Table 2.** H100 speedup over A100 (H100 Performance, TC = Tensor Core) (Page 26)
- **Table 3.** Comparison of NVIDIA A100 and H100 Data Center GPUs (Page 39)
- **Table 4.** Compute Capability: V100 vs A100 vs H100 (Page 41)
- **Table 5.** Comparison of A100 to H100 video decode (number of streams) (Page 58)
- **Table 6.** H100 Hardware Decode Support (Page 58)
- **Table 7.** NVJPG Decode Performance (Page 59)
- **Table 8.** NVIDIA DGX H100 System Specifications (Page 62)

---

## Introduction

NVIDIA® accelerated computing technologies tackle computational challenges far beyond the capabilities of ordinary computers. Accelerated computing requires more than just powerful GPUs. The combination of NVIDIA® CUDA® general purpose programmable GPUs and numerous GPU-accelerated SDKs, APIs, and algorithms provides full-stack computing solutions to deliver incredible application speed-ups across multiple domains. Distributed GPU computing systems and software scale processing across an entire data center. Cloud data centers worldwide are increasingly scaling up and scaling out with NVIDIA GPU-accelerated systems and architectures, running a diverse set of AI, HPC, and data analytics applications.

Over 15 years ago, NVIDIA introduced the CUDA parallel computing platform with the G80 GPU. Since that time, CUDA tools and libraries have been downloaded over 30 million times and used by nearly 3 million developers. The CUDA platform has been continuously improved, optimized, and expanded with more powerful CUDA-enabled GPUs, new and diverse sets of GPU-accelerated libraries, workstations, servers, and applications to broaden the reach of NVIDIA accelerated computing.

NVIDIA now has full stack solutions for different industries, fields of science, and applications. Over 450 NVIDIA SDKs, toolkits, libraries, and models serve industries and applications from gaming and design, to life and earth sciences, robotics, self-driving cars, quantum computing, supply-chain logistics, cybersecurity, 5G, climate science, digital biology, and more. Over 25,000 companies use NVIDIA AI technologies today.

The ease of programming and richness of NVIDIA's CUDA platform allows designers, researchers, and engineers to innovate quickly. And with continued platform software optimizations, it's common for users to experience speed-ups of multiple X-factors through the life of the NVIDIA products.

NVIDIA GPUs are used in many of the world's largest data centers, providing tremendous speed-ups for AI, HPC, and data analytics systems and applications. Cloud data centers are rapidly scaling-up AI training and scaling-out inference applications with NVIDIA GPUs. Many different types of AI models are now matured and industrialized for broad enterprise use, and have been trained and continuously improved by using NVIDIA GPUs. Examples of matured AI models include: computer vision models, speech recognition, recommender systems, graphs and trees, time series models, generative models, variable encoders, and large language models. In fact, customizing large language models for new languages and domains will likely be one of the largest supercomputing applications ever.

NVIDIA's new Omniverse™ platform will power numerous metaverse environments and require massive GPU computational abilities. In addition to the NVIDIA RTX GPUs that will power the real-time rendering and simulation in many Omniverse-enabled metaverses, we expect H100-enabled systems to add additional AI and simulation horsepower for complex digital twin challenges. One of the largest supercomputing endeavors will be NVIDIA's own Earth-2 Supercomputer project which will continuously stream enormous amounts of data into a digital twin of the Earth running physical simulations in Omniverse to predict future weather patterns worldwide.

```
+---------------------------------------------------------------------------------------+
|                       DIVERSE WORKLOADS IN MODERN CLOUD COMPUTING                     |
|                                                                                       |
|  [Graphics]           [Scientific Computing]  [Data Analytics]     [AI Deep Learning] |
|                                                                    [    Training    ] |
|                                                                                       |
|  [Edge AI Video]      [Cloud Gaming]          [Genomics]           [Classical ML]     |
|  [   Analytics ]                                                                      |
|                                                                                       |
|  [AI Deep Learning]   [5G Private Networks]                                           |
|  [   Inference    ]                                                                   |
+---------------------------------------------------------------------------------------+
```
*Figure 1. Modern cloud datacenter workloads require NVIDIA GPU acceleration*

In this whitepaper we introduce the new NVIDIA H100 Tensor Core GPU, our next-generation, highest-performing data center GPU. Based on the NVIDIA Hopper GPU architecture, H100 will accelerate AI training and inference, HPC, and data analytics applications in cloud data centers, servers, systems at the edge, and workstations.

A high-level overview of H100, new H100-based DGX, DGX SuperPOD, and HGX systems, and a new H100-based Converged Accelerator is followed by a deep dive into H100 hardware architecture, efficiency improvements, and new programming features.

---

## NVIDIA H100 Tensor Core GPU Overview

The complexity of artificial intelligence (AI), high-performance computing (HPC), and data analytics is increasing exponentially, requiring scientists and engineers to use the most advanced computing platforms. NVIDIA Hopper GPU architecture securely delivers the highest performance computing with low latency, and integrates a full stack of capabilities for computing at data center scale.

The NVIDIA® H100 Tensor Core GPU powered by the NVIDIA Hopper GPU architecture delivers the next massive leap in accelerated computing performance for NVIDIA's data center platforms. H100 securely accelerates diverse workloads from small enterprise workloads, to exascale HPC, to trillion parameter AI models.

Implemented using TSMC's 4N process customized for NVIDIA with 80 billion transistors, and including numerous architectural advances, H100 is the world's most advanced chip ever built.

*Figure 2. NVIDIA H100 GPU on new SXM5 Module*

H100 is NVIDIA's 9th-generation data center GPU designed to deliver an order-of-magnitude performance leap for large-scale AI and HPC over our prior generation NVIDIA A100 Tensor Core GPU. H100 carries over the major design focus of A100 to improve strong scaling for AI and HPC workloads, with substantial improvements in architectural efficiency.

For today's mainstream AI and HPC models, H100 with InfiniBand interconnect delivers up to 30 times the performance of A100 (see Figure 3).

With the new NVLink Switch System interconnect targeted at some of the largest and most challenging computing workloads that require model parallelism across multiple GPU-accelerated nodes to fit, these workloads receive yet another generational performance leap, in some cases tripling performance yet again over H100 with InfiniBand.

*Figure 3. H100 Enables Next-Generation AI and HPC Breakthroughs*  
*(Performance comparisons for HPC, AI Inference, and AI Training across A100, H100, and H100 + NVLink Network)*

> **Note:** All performance numbers are preliminary based on current expectations and subject to change in shipping products. A100 cluster: HDR IB network. H100 cluster: NDR IB network with NVLink Switch System where indicated. Note that NVLink Switch System technology is not available with H100 systems now, but systems and availability will be announced in the future.  
> **# GPUs:** Climate Modeling 1K, LQCD 1K, Genomics 8, 3D-FFT 256, MT-NLG 32 (batch sizes: 4 for A100, 60 for H100 at 1 sec, 8 for A100 and 64 for H100 at 1.5 and 2sec), MRCNN 8 (batch 32), GPT-3 16B 512 (batch 256), DLRM 128 (batch 64K), GPT-3 16K (batch 512), MoE 8K (batch 512, one expert per GPU).

At GTC Spring 2022, the new NVIDIA Grace Hopper Superchip product was announced. The Hopper H100 Tensor Core GPU will power the NVIDIA Grace Hopper Superchip CPU+GPU architecture, purpose-built for terabyte-scale accelerated computing and providing $10\times$ higher performance on large-model AI and HPC.

The NVIDIA Grace CPU leverages the flexibility of the Arm® architecture to create a CPU and server architecture designed from the ground up for accelerated computing. H100 is paired to Grace with NVIDIA's ultra-fast chip-to-chip interconnect, delivering $900\text{ GB/s}$ of bandwidth, $7\times$ faster than PCIe Gen5. This innovative design will deliver up to $30\times$ higher aggregate bandwidth compared to today's fastest servers and up to $10\times$ higher performance for applications running terabytes of data.

*Figure 4. Grace Hopper Superchip*

---

## NVIDIA H100 GPU Key Feature Summary

- **New Streaming Multiprocessor (SM)** has many performance and efficiency improvements. Key new features include:
  - **New fourth-generation Tensor Cores** are up to $6\times$ faster chip-to-chip compared to A100, including per-SM speedup, additional SM count, and higher clocks of H100. On a per SM basis, the Tensor Cores deliver $2\times$ the MMA (Matrix Multiply-Accumulate) computational rates of the A100 SM on equivalent data types, and $4\times$ the rate of A100 using the new FP8 data type, compared to previous generation 16-bit floating point options. The Sparsity feature exploits fine-grained structured sparsity in deep learning networks, doubling the performance of standard Tensor Core operations.
  - **New DPX Instructions** accelerate Dynamic Programming algorithms by up to $7\times$ over the A100 GPU. Two examples include the Smith-Waterman algorithm for genomics processing, and the Floyd-Warshall algorithm used to find optimal routes for a fleet of robots through a dynamic warehouse environment.
  - **$3\times$ faster IEEE FP64 and FP32 processing rates** chip-to-chip compared to A100, due to $2\times$ faster clock-for-clock performance per SM, plus additional SM counts and higher clocks of H100.
  - **New Thread Block Cluster feature** allows programmatic control of locality at a granularity larger than a single Thread Block on a single SM. This extends the CUDA programming model by adding another level to the programming hierarchy to now include Threads, Thread Blocks, Thread Block Clusters, and Grids. Clusters enable multiple Thread Blocks running concurrently across multiple SMs to synchronize and collaboratively fetch and exchange data.
  - **New Asynchronous Execution features** include a new Tensor Memory Accelerator (TMA) unit that can transfer large blocks of data very efficiently between global memory and shared memory. TMA also supports asynchronous copies between Thread Blocks in a Cluster. There is also a new Asynchronous Transaction Barrier for doing atomic data movement and synchronization.
- **New Transformer Engine** uses a combination of software and custom Hopper Tensor Core technology designed specifically to accelerate Transformer model training and inference. The Transformer Engine intelligently manages and dynamically chooses between FP8 and 16-bit calculations, automatically handling re-casting and scaling between FP8 and 16-bit in each layer to deliver up to $9\times$ faster AI training and up to $30\times$ faster AI inference speedups on large language models compared to the prior generation A100.
- **HBM3 memory subsystem** provides nearly a $2\times$ bandwidth increase over the previous generation. The H100 SXM5 GPU is the world's first GPU with HBM3 memory delivering a class-leading $3\text{ TB/sec}$ of memory bandwidth.
- **$50\text{ MB}$ L2 cache architecture** caches large portions of models and datasets for repeated access, reducing trips to HBM3.
- **Second-generation Multi-Instance GPU (MIG) technology** provides approximately $3\times$ more compute capacity and nearly $2\times$ more memory bandwidth per GPU Instance compared to A100. Confidential Computing capability with MIG-level Trusted Execution Environments (TEE) is now provided for the first time. Up to seven individual GPU Instances are supported, each with dedicated NVDEC and NVJPG units. Each Instance now includes its own set of performance monitors that work with NVIDIA developer tools.
- **New Confidential Computing support** protects user data, defends against hardware and software attacks, and better isolates and protects VMs from each other in virtualized and MIG environments. H100 implements the world's first native Confidential Computing GPU and extends the Trusted Execution Environment with CPUs at full PCIe line rate.
- **Fourth-generation NVIDIA NVLink®** provides a $3\times$ bandwidth increase on all-reduce operations and a 50% general bandwidth increase over the prior generation NVLink with $900\text{ GB/sec}$ total bandwidth for multi-GPU IO operating at $7\times$ the bandwidth of PCIe Gen 5.
- **Third-generation NVSwitch technology** includes switches residing both inside and outside of nodes to connect multiple GPUs in servers, clusters, and data center environments. Each NVSwitch inside a node provides 64 ports of fourth-generation NVLink links to accelerate multi-GPU connectivity. Total switch throughput increases to $13.6\text{ Tbits/sec}$ from $7.2\text{ Tbits/sec}$ in the prior generation. New third-generation NVSwitch technology also provides hardware acceleration for collective operations with multicast and NVIDIA SHARP in-network reductions.
- **New NVLink Switch System interconnect technology** and new second-level NVLink Switches based on third-gen NVSwitch technology introduce address space isolation and protection, enabling up to 32 nodes or 256 GPUs to be connected over NVLink in a 2:1 tapered, fat tree topology. These connected nodes are capable of delivering $57.6\text{ TB/sec}$ of all-to-all bandwidth and can supply an incredible one exaFLOP of FP8 sparse AI compute.
- **PCIe Gen 5** provides $128\text{ GB/sec}$ total bandwidth ($64\text{ GB/sec}$ in each direction) compared to $64\text{ GB/sec}$ total bandwidth ($32\text{ GB/sec}$ in each direction) in Gen 4 PCIe. PCIe Gen 5 enables H100 to interface with the highest performing x86 CPUs and SmartNICs / DPUs (Data Processing Units).

*Figure 5. New Technologies in Hopper H100: World's Most Advanced Chip (80B Transistors), Transformer Engine ($6\times$ Transformer Performance), 2nd Generation Multi-Instance GPU ($7\times$ Secure Tenants), Confidential Computing (Secure Data and AI Models In-Use), 4th Generation NVLink ($7\times$ PCIe Gen5), DPX Instructions ($7\times$ Dynamic Prog. Performance)*

---

## NVIDIA GPU-Accelerated Data Centers

From AI and data analytics to high-performance computing (HPC), data centers are key to solving some of the most important challenges. End-to-end NVIDIA accelerated computing platforms, integrated across hardware and software, gives enterprises the blueprint to a robust, secure infrastructure that supports develop-to-deploy implementations across all modern workloads.

Deep learning datasets are becoming larger and more complex, with workloads like conversational AI, recommender systems, and computer vision becoming increasingly prevalent across industries. NVIDIA data center platforms, including hardware and software, significantly accelerates AI training, resulting in highly productive data science teams, significant cost savings, and faster time to ROI.

Accelerating inference workloads in the data center requires an agile, elastic infrastructure that can scale out and utilize every bit of compute resources available. With new technologies like Multi-Instance GPU (MIG), NVIDIA solutions are uniquely positioned to accelerate inference workloads like image recognition, recommender systems, and natural language processing, providing the highest throughput and real-time responsiveness needed to bring AI to applications.

HPC is one of the most essential tools fueling the advancement of science in the data center. NVIDIA GPUs are the engine of the modern HPC data center. By delivering breakthrough performance with fewer servers resulting in faster insights and dramatically lower costs, NVIDIA data center platforms pave the way to scientific discovery.

Businesses are generating and collecting unprecedented amounts of data. The more data available to analyze, the more that can be learned. With NVIDIA data center platforms and analytics solutions, businesses can derive actionable insights from their data faster than ever before.

NVIDIA GPU acceleration for data centers is available through a broad range of servers from NVIDIA's vast ecosystem of partner server makers. H100 GPUs are available in different configurations to support the different requirements of server designs.

### H100 SXM5 GPU
The H100 SXM5 configuration using NVIDIA's custom-built SXM5 board that houses the H100 GPU and HBM3 memory stacks, and also provides fourth-generation NVLink and PCIe Gen 5 connectivity, provides the highest application performance. This configuration is ideal for customers with applications scaling to multiple GPUs in a server, and across servers. It's available through HGX H100 server boards with 4-GPU and 8-GPU configurations. While the 4-GPU configuration includes point-to-point NVLink connections between GPUs and provides a higher CPU-to-GPU ratio in the server, the 8-GPU configuration includes NVSwitch to provide SHARP in-network reductions and full NVLink bandwidth of $900\text{ GB/s}$ between any pair of GPUs. The H100 SXM5 GPU is also used in the powerful new DGX H100 servers and DGX SuperPOD systems.

### H100 PCIe Gen 5 GPU
The H100 PCIe Gen 5 configuration provides all the capabilities of H100 SXM5 GPUs in just 350 Watts of Thermal Design Power (TDP). This configuration can optionally use the NVLink bridge for connecting up to two GPUs at $600\text{ GB/s}$ of bandwidth, nearly five times PCIe Gen5. Well suited for mainstream accelerated servers that go into standard racks offering lower power per server, H100 PCIe provides great performance for applications that scale to 1 or 2 GPUs at a time, including AI Inference and some HPC applications. On a basket of 10 top data analytics, AI and HPC applications, a single H100 PCIe GPU efficiently provides 65% delivered performance of the H100 SXM5 GPU while consuming 50% of the power.

### DGX H100 and DGX SuperPOD
NVIDIA DGX H100 is a universal high-performance AI system for training, inference, and analytics. DGX H100 is equipped with BlueField-3, NDR InfiniBand, and second-generation MIG technology. A single DGX H100 system delivers an unmatched 16 petaFLOPS of FP16 sparse AI compute performance. This performance can be easily scaled up by connecting multiple DGX H100 systems into clusters known as DGX PODs or even DGX SuperPODs. A DGX SuperPOD starts with 32 DGX H100 systems, referred to as a "scalable unit", which integrates 256 H100 GPUs connected via the new second-level NVLink switches based on third-generation NVSwitch technology, delivering an unprecedented one exaFLOP of FP8 sparse AI compute performance. DGX H100 SuperPOD will support both InfiniBand and NVLINK Switch network options.

*(See Appendix A for more details.)*

### HGX H100
As workloads explode in complexity there's a need for multiple GPUs to work together with extremely fast communication between them. NVIDIA HGX H100™ combines multiple H100 GPUs with the high-speed interconnect powered by NVLink and NVSwitch to enable the creation of the world's most powerful scale-up servers.

HGX H100 is available as a server building block in the form of integrated baseboards in four or eight H100 GPUs configurations. Four GPU HGX H100 offers fully interconnected point to point NVLink connections between GPUs, while the eight GPU configuration offers full GPU-to-GPU bandwidth through NVSwitch. Leveraging the power of H100 multi-precision Tensor Cores, an 8-way HGX H100 provides over 32 petaFLOPS of deep learning compute performance using sparse FP8 operations. HGX H100 enables standardized high-performance servers that provide predictable performance on various application workloads, while also enabling faster time to market for NVIDIA's ecosystem of partner server makers.

### H100 CNX Converged Accelerator
NVIDIA H100 CNX combines the power of the NVIDIA H100 GPU with the advanced networking capabilities of the NVIDIA® ConnectX-7 SmartNIC, which delivers up to $400\text{ Gb/s}$ of bandwidth and includes innovative features such as NVIDIA ASAP2 (Accelerated Switching and Packet Processing), and in-line hardware acceleration for TLS/IPsec/MACsec encryption/decryption. This unique architecture delivers unprecedented performance for GPU-powered I/O intensive workloads, such as distributed AI training in the enterprise data center, or 5G signal processing at the edge.

---

## NVIDIA H100 GPU Architecture In-Depth

The NVIDIA H100 GPU based on the new Hopper GPU architecture features multiple innovations:
- **New fourth-generation Tensor Cores** perform faster matrix computations than ever before on an even broader array of AI and HPC tasks.
- **A new Transformer Engine** enables H100 to deliver up to $9\times$ faster AI training and up to $30\times$ faster AI inference speedups on large language models compared to the prior generation A100.
- **The new NVLink Network interconnect** enables GPU-to-GPU communication among up to 256 GPUs across multiple compute nodes.
- **Secure MIG** partitions the GPU into isolated, right-size instances to maximize QoS (quality of service) for smaller workloads.
- NVIDIA's H100 is the first truly **asynchronous GPU**. H100 extends A100's global-to-shared asynchronous transfers across all address spaces, and adds support for tensor memory access patterns. It enables applications to build end-to-end asynchronous pipelines that move data into and off the chip, completely overlapping and hiding data movement with computation.
- Only a small number of CUDA threads are now required to manage the full memory bandwidth of H100 using the new **Tensor Memory Accelerator**, while most other CUDA threads can be focused on general-purpose computations, such as pre-processing and post-processing data for the new generation of Tensor Cores.
- H100 grows the CUDA thread group hierarchy with a new level called the **Thread Block Cluster**. A Cluster is a group of Thread Blocks that are guaranteed to be concurrently scheduled, and enable efficient cooperation and data sharing for threads across multiple SMs. A Cluster also cooperatively drives asynchronous units like the Tensor Memory Accelerator and the Tensor Cores more efficiently.
- Orchestrating the growing number of on-chip accelerators and diverse groups of general-purpose threads requires synchronization. NVIDIA's **Asynchronous Transaction Barrier** enables general-purpose CUDA threads and on-chip accelerators within a Cluster to synchronize efficiently, even if they reside on separate SMs.

The full GH100 GPU that powers the H100 GPU is fabricated using TSMC's 4N process customized for NVIDIA, with 80 billion transistors, a die size of $814\text{ mm}^2$, and higher frequency design.

The NVIDIA GH100 GPU is composed of multiple GPU Processing Clusters (GPCs), Texture Processing Clusters (TPCs), Streaming Multiprocessors (SMs), L2 cache, and HBM3 memory controllers.

The full implementation of the **GH100 GPU** includes:
- 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU
- 128 FP32 CUDA Cores per SM, 18432 FP32 CUDA Cores per full GPU
- 4 Fourth-Generation Tensor Cores per SM, 576 per full GPU
- 6 HBM3 or HBM2e stacks, 12 512-bit Memory Controllers
- 60 MB L2 Cache
- Fourth-Generation NVLink and PCIe Gen 5

The **NVIDIA H100 GPU with SXM5** board form-factor includes:
- 8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs per GPU
- 128 FP32 CUDA Cores per SM, 16896 FP32 CUDA Cores per GPU
- 4 Fourth-generation Tensor Cores per SM, 528 per GPU
- 80 GB HBM3, 5 HBM3 stacks, 10 512-bit Memory Controllers
- 50 MB L2 Cache
- Fourth-Generation NVLink and PCIe Gen 5

The **NVIDIA H100 GPU with PCIe Gen 5** board form-factor includes:
- 7 or 8 GPCs, 57 TPCs, 2 SMs/TPC, 114 SMs per GPU
- 128 FP32 CUDA Cores/SM, 14592 FP32 CUDA Cores per GPU
- 4 Fourth-generation Tensor Cores per SM, 456 per GPU
- 80 GB HBM2e, 5 HBM2e stacks, 10 512-bit Memory Controllers
- 50 MB L2 Cache
- Fourth-Generation NVLink and PCIe Gen 5

*Figure 6. GH100 Full GPU with 144 SMs*

---

### H100 SM Architecture

Building upon the NVIDIA A100 Tensor Core GPU SM architecture, the H100 SM quadruples A100's peak per-SM floating point computational power, due to the introduction of FP8, and doubles A100's raw SM computational power on all previous Tensor Core and FP32 / FP64 data types, clock-for-clock.

The new Transformer Engine, combined with Hopper's FP8 Tensor Cores, delivers up to $9\times$ faster AI training and $30\times$ faster AI inference speedups on large language models compared to the prior generation A100. Hopper's new DPX instructions enable up to $7\times$ faster Smith-Waterman algorithm processing for genomics and protein sequencing.

Hopper's new fourth-generation Tensor Core, Tensor Memory Accelerator, and many other new SM and general H100 architecture improvements together deliver up to $3\times$ faster HPC and AI performance in many other cases.

#### Table 1. NVIDIA H100 Tensor Core GPU Performance Specs

| Metric | NVIDIA H100 SXM5 | NVIDIA H100 PCIe |
| :--- | :--- | :--- |
| **Peak FP64** | 33.5 TFLOPS | 25.6 TFLOPS |
| **Peak FP64 Tensor Core** | 66.9 TFLOPS | 51.2 TFLOPS |
| **Peak FP32** | 66.9 TFLOPS | 51.2 TFLOPS |
| **Peak FP16** | 133.8 TFLOPS | 102.4 TFLOPS |
| **Peak BF16** | 133.8 TFLOPS | 102.4 TFLOPS |
| **Peak TF32 Tensor Core** | 494.7 TFLOPS \| 989.4 TFLOPS¹ | 378 TFLOPS \| 756 TFLOPS¹ |
| **Peak FP16 Tensor Core** | 989.4 TFLOPS \| 1978.9 TFLOPS¹ | 756 TFLOPS \| 1513 TFLOPS¹ |
| **Peak BF16 Tensor Core** | 989.4 TFLOPS \| 1978.9 TFLOPS¹ | 756 TFLOPS \| 1513 TFLOPS¹ |
| **Peak FP8 Tensor Core** | 1978.9 TFLOPS \| 3957.8 TFLOPS¹ | 1513 TFLOPS \| 3026 TFLOPS¹ |
| **Peak INT8 Tensor Core** | 1978.9 TOPS \| 3957.8 TOPS¹ | 1513 TOPS \| 3026 TOPS¹ |

*1. Effective TFLOPS / TOPS using the Sparsity feature.*

*Figure 7. GH100 Streaming Multiprocessor (SM) architectural diagram showing 4 processing partitions with L0 Instruction Cache, Warp Scheduler (32 thread/clk), Dispatch Unit (32 thread/clk), Register File ($16,384 \times 32\text{-bit}$), INT32/FP32/FP64 Execution Units, 4th Generation Tensor Cores, LD/ST, SFU, unified 256 KB L1 Data Cache / Shared Memory, and Tensor Memory Accelerator.*

---

### H100 SM Key Feature Summary

- **Fourth-generation Tensor Cores:**
  - Up to $6\times$ faster chip-to-chip compared to A100, including per-SM speedup, additional SM count, and higher clocks of H100.
  - On a per SM basis, the Tensor Cores deliver $2\times$ the MMA (Matrix Multiply-Accumulate) computational rates of the A100 SM on equivalent data types, and $4\times$ the rate of A100 using the new FP8 data type, compared to previous generation 16-bit floating point options.
  - Sparsity feature exploits fine-grained structured sparsity in deep learning networks, doubling the performance of standard Tensor Core operations.
- **New DPX Instructions** accelerate Dynamic Programming algorithms by up to $7\times$ over the A100 GPU.
- **$3\times$ faster IEEE FP64 and FP32 processing rates** chip-to-chip compared to A100.
- **$256\text{ KB}$ of combined shared memory and L1 data cache**, $1.33\times$ larger than A100.
- **New Asynchronous Execution features** including Tensor Memory Accelerator (TMA) and Asynchronous Transaction Barrier.
- **New Thread Block Cluster feature** exposes control of locality across multiple SMs.
- **Distributed Shared Memory** allows direct SM-to-SM communications for loads, stores, and atomics across multiple SM shared memory blocks.

---

### H100 Tensor Core Architecture

Tensor Cores are specialized high-performance compute cores for matrix multiply and accumulate (MMA) math operations that provide groundbreaking performance for AI and HPC applications. Tensor Cores operating in parallel across SMs in one NVIDIA GPU deliver massive increases in throughput and efficiency compared to standard Floating-Point (FP), Integer (INT), and FMA (Fused Multiply-Accumulate) operations. Tensor Cores were first introduced in the NVIDIA Tesla® V100 GPU, and further enhanced in each new NVIDIA GPU architecture generation.

The new fourth-generation Tensor Core architecture in H100 delivers double the raw dense and sparse matrix math throughput per SM, clock-for-clock, compared to A100, and even more when considering the higher GPU Boost clock of H100 over A100. FP8, FP16, BF16, TF32, FP64, and INT8 MMA data types are supported. The new Tensor Cores also have more efficient data management, saving up to 30% operand delivery power.

*Figure 8. H100 FP16 Tensor Core has 3x throughput compared to A100 FP16 Tensor Core*

---

### Hopper FP8 Data Format

The H100 GPU adds FP8 Tensor Cores to accelerate both AI training and inference. FP8 Tensor Cores support FP32 and FP16 accumulators, and two new FP8 input types:
- **E4M3:** 1 sign bit, 4 exponent bits, and 3 mantissa bits.
- **E5M2:** 1 sign bit, 5 exponent bits, and 2 mantissa bits.

E4M3 supports computations requiring less dynamic range with more precision, while E5M2 provides a wider dynamic range and less precision. FP8 halves data storage requirements and doubles throughput compared to FP16 or BF16.

```
Precision bit formats:
FP32:  [s] [  e8  ] [                 m23                 ]
FP16:  [s] [  e5  ] [    m10    ]
BF16:  [s] [  e8  ] [    m7   ]
FP8 (E5M2): [s] [  e5  ] [ m2 ]
FP8 (E4M3): [s] [  e4  ] [ m3 ]
```
*Figure 9. New Hopper FP8 Precisions - 2x throughput and half the footprint of FP16/BF16*

*Figure 10. H100 FP8 Tensor Core 6x throughput compared to A100 FP16 Tensor Core*

*Figure 11. H100 TF32, FP64, and INT8 Tensor Cores all have 3x throughput versus A100*

#### Table 2. H100 Speedup over A100 (H100 Performance, TC = Tensor Core)

| Data Type | A100 (Dense) | A100 Sparse | H100 SXM5 (Dense) | H100 SXM5 Sparse | H100 SXM5 Speedup vs A100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP8 Tensor Core** | NA | NA | 1978.9 TFLOPS | 3957.8 TFLOPS | $6.3\times$ vs A100 FP16 TC |
| **FP16** | 78 TFLOPS | NA | 133.8 TFLOPS | NA | $1.7\times$ |
| **FP16 Tensor Core** | 312 TFLOPS | 624 TFLOPS | 989.4 TFLOPS | 1978.9 TFLOPS | $3.2\times$ |
| **BF16 Tensor Core** | 312 TFLOPS | 624 TFLOPS | 989.4 TFLOPS | 1978.9 TFLOPS | $3.2\times$ |
| **FP32** | 19.5 TFLOPS | NA | 66.9 TFLOPS | NA | $3.4\times$ |
| **TF32 Tensor Core** | 156 TFLOPS | 312 TFLOPS | 494.7 TFLOPS | 989.4 TFLOPS | $3.2\times$ |
| **FP64** | 9.7 TFLOPS | NA | 33.5 TFLOPS | NA | $3.5\times$ |
| **FP64 Tensor Core** | 19.5 TFLOPS | NA | 66.9 TFLOPS | NA | $3.4\times$ |
| **INT8 Tensor Core** | 624 TOPS | 1248 TOPS | 1978.9 TOPS | 3957.8 TOPS | $3.2\times$ |

---

### New DPX Instructions for Accelerated Dynamic Programming

Many "brute force" optimization algorithms have the property that a sub-problem solution is reused many times when solving the larger problem. Dynamic Programming is an algorithmic technique for solving a complex recursive problem by breaking it down into simpler sub-problems. By storing the results of sub-problems, without the need to recompute them when needed later, Dynamic Programming algorithms reduce the computational complexity of exponential problem sets to a linear scale.

Dynamic programming is commonly used in a broad range of optimization, data processing, and genomics algorithms:
- In genome sequencing: **Smith-Waterman** dynamic programming algorithm.
- In robotics and routing: **Floyd-Warshall** algorithm to find optimal routes for a fleet of robots through a dynamic warehouse environment in real time.

H100 introduces DPX instructions to accelerate the performance of Dynamic Programming algorithms by up to $7\times$ compared to Ampere GPUs. These new instructions provide support for advanced fused operands for the inner loop of many DP algorithms.

*Figure 12. DPX Instructions Accelerate Dynamic Programming (Up to 40x speedup over CPU)*

---

### Combined L1 Data Cache and Shared Memory

Combining data cache and shared memory functionality into a single memory block provides the best overall performance for both types of memory accesses. The combined capacity of the L1 data cache and shared memory is $256\text{ KB/SM}$ in H100 versus $192\text{ KB/SM}$ in A100. SM shared memory size itself is configurable up to $228\text{ KB}$ in H100.

---

### H100 Compute Performance Summary

Overall, H100 provides approximately $6\times$ compute performance improvement over A100:
- **$1.2\times$:** 132 SMs ($22\%$ increase over A100's 108 SMs)
- **$2.0\times$:** New 4th Generation Tensor Core (doubling MMA rates clock-for-clock)
- **$2.0\times$:** FP8 format and associated Transformer Engine
- **$1.3\times$:** Increased GPU core clock frequency

Combined, these factors yield an approximate $6\times$ peak compute throughput advantage over A100.

*Figure 13. H100 Compute Improvement Summary*

---

## H100 GPU Hierarchy and Asynchrony Improvements

### Thread Block Clusters
The CUDA programming model has long relied on a GPU compute architecture that uses Grids containing multiple Thread Blocks to leverage locality. H100 introduces a new **Thread Block Cluster** architecture that exposes control of locality at a granularity larger than a single Thread Block on a single SM.

Thread Block Clusters extend the CUDA programming model to include four levels:
$$\text{Threads} \longrightarrow \text{Thread Blocks} \longrightarrow \text{Thread Block Clusters} \longrightarrow \text{Grids}$$

Clusters in H100 run concurrently across SMs within a single GPU Processing Cluster (GPC). A dedicated SM-to-SM network provides fast, low-latency data sharing between threads in a Cluster.

*Figure 14. Thread Block Clusters and Grids with Clusters*

### Distributed Shared Memory
With Clusters, all threads can directly access other SMs' shared memory with load, store, and atomic operations. This is called **Distributed Shared Memory (DSMEM)**.
- Data no longer needs to be written to and read from global memory to pass between SMs.
- Accelerates data exchange between Thread Blocks by approximately $7\times$ compared to global memory.

*Figure 15. Thread Block to Thread Block data exchange (A100 vs H100 with Clusters)*  
*Figure 16. Cluster versus non-Cluster Performance Comparisons (64K FFT: 2x; Longstaff Schwartz Pricing: 2.7x; Histogram Collection: 1.7x)*

### Asynchronous Execution
Hopper adds new features that improve asynchronous execution and allow further overlap of memory copies with computation:

| CUDA Programming Model Exposure | A100 | New for H100 |
| :--- | :--- | :--- |
| `Barrier.arrive()`, `Barrier.wait()` | Asynchronous Barrier; Waiter spins in SMEM | **Asynchronous Transaction Barrier**; Waiter sleeps until all threads arrive |
| `Memcpy_async()` | Direct copy to SMEM (using `LDGSTS`) | **Asynchronous mem copy unit (TMA)** |

*Figure 17. Asynchronous Execution Concurrency and Enhancements in Hopper*

### Tensor Memory Accelerator (TMA)
TMA improves data fetching efficiency by transferring large blocks of multi-dimensional tensors between global and shared memory using a **copy descriptor** (specifying tensor dimensions, strides, and coordinates) instead of per-element addressing.
- Supports 1D to 5D tensors, memory access modes, reductions, and boundary padding.
- Single-threaded issue: a single thread issues `cuda::memcpy_async` and threads wait on a `cuda::barrier`.
- Eliminates thread-based address generation loops required by A100's `LDGSTS`.

*Figure 18. TMA Address Generation via Copy Descriptor*  
*Figure 19. Asynchronous Memory Copy with TMA on H100 vs LDGSTS on A100*

### Asynchronous Transaction Barrier
Hopper extends the split arrive/wait Asynchronous Barrier from Ampere by tracking not only thread arrivals but also **transaction byte counts**.
- A producer thread writes to shared memory with a transaction count.
- Waiting threads sleep (saving power and execution resources) until all threads have arrived and the expected transaction byte count has been reached.

*Figure 20. Asynchronous Barrier in A100 vs Asynchronous Transaction Barrier in H100*

---

## H100 HBM and L2 Cache Memory Architectures

### H100 HBM3 and HBM2e DRAM Subsystems
- **H100 SXM5:** Supports $80\text{ GB}$ (5 stacks) of HBM3 memory, delivering over $3\text{ TB/sec}$ of memory bandwidth (nearly $2\times$ A100).
- **H100 PCIe:** Supports $80\text{ GB}$ of fast HBM2e with over $2\text{ TB/sec}$ of memory bandwidth.

*Figure 21. World's First HBM3 GPU Memory Architecture, 2x Delivered Bandwidth*

### H100 L2 Cache
- **Capacity:** $50\text{ MB}$, a $1.25\times$ increase over A100's $40\text{ MB}$.
- Partitioned crossbar structure localizes and caches data for memory accesses from directly connected GPCs.
- Includes cache residency controls and hardware data compression/decompression.

### Memory Subsystem RAS Features
1. **ECC Memory Resiliency:** Sideband SECDED ECC protects HBM3/HBM2e without carving out main memory space. L2, L1, and SM register files are also SECDED ECC protected.
2. **Memory Row Remapping:** Replaces rows that generated ECC errors with spare known-good rows at boot time.

---

### Table 3. Comparison of NVIDIA A100 and H100 Data Center GPUs

| GPU Features | NVIDIA A100 | NVIDIA H100 SXM5 | NVIDIA H100 PCIe |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | NVIDIA Ampere | NVIDIA Hopper | NVIDIA Hopper |
| **GPU Board Form Factor** | SXM4 | SXM5 | PCIe Gen 5 |
| **SMs** | 108 | 132 | 114 |
| **TPCs** | 54 | 66 | 57 |
| **FP32 Cores / SM** | 64 | 128 | 128 |
| **FP32 Cores / GPU** | 6912 | 16896 | 14592 |
| **FP64 Cores / SM (excl. Tensor)** | 32 | 64 | 64 |
| **FP64 Cores / GPU (excl. Tensor)** | 3456 | 8448 | 7296 |
| **INT32 Cores / SM** | 64 | 64 | 64 |
| **INT32 Cores / GPU** | 6912 | 8448 | 7296 |
| **Tensor Cores / SM** | 4 | 4 | 4 |
| **Tensor Cores / GPU** | 432 | 528 | 456 |
| **GPU Boost Clock (FP8, FP16, BF16, TF32 TC)** | 1410 MHz | 1830 MHz | 1620 MHz |
| **GPU Boost Clock (FP64 TC, FP32/FP64 non-TC)** | 1410 MHz | 1980 MHz | 1755 MHz |
| **Peak FP8 Tensor TFLOPS (FP16 Accumulate)** | NA | 1978.9 / 3957.8¹ | 1513 / 3026¹ |
| **Peak FP8 Tensor TFLOPS (FP32 Accumulate)** | NA | 1978.9 / 3957.8¹ | 1513 / 3026¹ |
| **Peak FP16 Tensor TFLOPS (FP16 Accumulate)** | 312 / 624¹ | 989.4 / 1978.9¹ | 756 / 1513¹ |
| **Peak FP16 Tensor TFLOPS (FP32 Accumulate)** | 312 / 624¹ | 989.4 / 1978.9¹ | 756 / 1513¹ |
| **Peak BF16 Tensor TFLOPS (FP32 Accumulate)** | 312 / 624¹ | 989.4 / 1978.9¹ | 756 / 1513¹ |
| **Peak TF32 Tensor TFLOPS** | 156 / 312¹ | 494.7 / 989.4¹ | 378 / 756¹ |
| **Peak FP64 Tensor TFLOPS** | 19.5 | 66.9 | 51.2 |
| **Peak INT8 Tensor TOPS** | 624 / 1248¹ | 1978.9 / 3957.8¹ | 1513 / 3026¹ |
| **Peak FP16 TFLOPS (non-Tensor)** | 78 | 133.8 | 102.4 |
| **Peak BF16 TFLOPS (non-Tensor)** | 39 | 133.8 | 102.4 |
| **Peak FP32 TFLOPS (non-Tensor)** | 19.5 | 66.9 | 51.2 |
| **Peak FP64 TFLOPS (non-Tensor)** | 9.7 | 33.5 | 25.6 |
| **Peak INT32 TOPS** | 19.5 | 33.5 | 25.6 |
| **Texture Units** | 432 | 528 | 456 |
| **Memory Interface** | 5120-bit HBM2 | 5120-bit HBM3 | 5120-bit HBM2e |
| **Memory Size** | 40 GB | 80 GB | 80 GB |
| **Memory Data Rate** | 1215 MHz DDR | 2619 MHz DDR | 1593 MHz DDR |
| **Memory Bandwidth** | 1555 GB/sec | 3352 GB/sec | 2039 GB/sec |
| **L2 Cache Size** | 40 MB | 50 MB | 50 MB |
| **Shared Memory Size / SM** | Configurable up to 164 KB | Configurable up to 228 KB | Configurable up to 228 KB |
| **Register File Size / SM** | 256 KB | 256 KB | 256 KB |
| **Register File Size / GPU** | 27648 KB | 33792 KB | 29184 KB |
| **TDP** | 400 Watts | 700 Watts | 350 Watts |
| **Transistors** | 54.2 billion | 80 billion | 80 billion |
| **GPU Die Size** | $826\text{ mm}^2$ | $814\text{ mm}^2$ | $814\text{ mm}^2$ |
| **TSMC Manufacturing Process** | 7 nm N7 | 4N customized for NVIDIA | 4N customized for NVIDIA |

*1. Effective TOPS / TFLOPS using the Sparsity feature.*  
*2. GPU Peak Clock and GPU Boost Clock are synonymous for NVIDIA Data Center GPUs.*  
*Note: H100 and A100 do not include display connectors, NVIDIA RT Cores, or NVENC encoders.*

---

## Compute Capability

The H100 GPU supports **Compute Capability 9.0**.

#### Table 4. Compute Capability: V100 vs A100 vs H100

| Feature | NVIDIA Tesla V100 | NVIDIA A100 | NVIDIA H100 |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | NVIDIA Volta | NVIDIA Ampere | NVIDIA Hopper |
| **Compute Capability** | 7.0 | 8.0 | 9.0 |
| **Threads / Warp** | 32 | 32 | 32 |
| **Max Warps / SM** | 64 | 64 | 64 |
| **Max Threads / SM** | 2048 | 2048 | 2048 |
| **Max Thread Blocks (CTAs) / SM** | 32 | 32 | 32 |
| **Max Thread Blocks / Thread Block Clusters** | NA | NA | 16 |
| **Max 32-bit Registers / SM** | 65536 | 65536 | 65536 |
| **Max Registers / Thread Block (CTA)** | 65536 | 65536 | 65536 |
| **Max Registers / Thread** | 255 | 255 | 255 |
| **Max Thread Block Size (# of threads)** | 1024 | 1024 | 1024 |
| **FP32 Cores / SM** | 64 | 64 | 128 |
| **Ratio of SM Registers to FP32 Cores** | 1024 | 1024 | 512 |
| **Shared Memory Size / SM** | Configurable up to 96 KB | Configurable up to 164 KB | Configurable up to 228 KB |

---

## Second-Generation Secure MIG

### MIG Technology Review
Multi-Instance GPU (MIG) partitions a GPU into up to seven independent, fully isolated GPU Instances with defined QoS, dedicated memory paths, L2 cache banks, crossbars, and memory controllers.

*Figure 22. Example CSP MIG Configuration*

### H100 MIG Enhancements
- Delivers $\sim 3\times$ compute capacity and $\sim 2\times$ memory bandwidth per instance compared to A100.
- Introduces MIG-level **Trusted Execution Environments (TEE)** for cloud-native multi-tenant security.
- Hardware firewalls enforce strict memory isolation.
- Each instance gets dedicated NVDEC and NVJPG engines and independent profiling counters.

*Figure 23. Secure MIG Example in Multi-Tenant Single GPU Configuration*

---

## Transformer Engine

Transformer models have grown by $275\times$ every two years, far outpacing typical compute trends. H100 introduces the **Transformer Engine**, combining custom Hopper Tensor Core hardware with intelligent FP8/FP16 selection software.

*Figure 24. Transformers Model Sizes Increasing Exponentially with Different Use Cases*

```
   [ Transformer Layer ]
             |
             v
   +------------------------------------+
   |         Transformer Engine         |
   |                                    |
   |  [Range Analysis] ---> [Scaling]   |
   |         ^                  |       |
   |         |                  v       |
   |   (Statistics)    [Format Conversion]|
   |         |                  |       |
   +---------|------------------|-------+
             |                  v
   [  H100 Tensor Core (FP8 / FP16 MMA) ]
```
*Figure 25. Transformer Engine Conceptual Operation*

- Dynamically tracks tensor statistics at each layer.
- Applies per-layer scaling factors to keep values within FP8 dynamic range without underflow or overflow.
- Converts to FP8 for computation while maintaining full model accuracy.

---

## Fourth-Generation NVLink and NVLink Network

- **Bandwidth:** $900\text{ GB/sec}$ total bidirectional bandwidth ($7\times$ PCIe Gen 5).
- **Physical implementation:** 18 links per GPU, using 2 differential pairs per direction at $25\text{ GB/sec}$ per link.
- **NVLink Network:** Introduces a scalable Network Address Space with hardware address translation, expanding direct NVLink clustering up to 256 GPUs (32 nodes).

### Third-Generation NVSwitch
- 64 ports of 4th-gen NVLink per switch chip ($13.6\text{ Tbits/sec}$ throughput).
- In-network hardware acceleration for collective operations: multicast, all-gather, reduce-scatter, and SHARP reductions.

### New NVLink Switch System
- Connects up to 256 GPUs across 32 DGX nodes in a 2:1 tapered fat-tree topology.
- Delivers $57.6\text{ TB/sec}$ all-to-all bisection bandwidth and 1 ExaFLOP of FP8 sparse compute.
- Employs LinkX cables up to 20 meters with Quad-Port OSFP optical transceivers.

#### Comparison: DGX A100 vs DGX H100 256-GPU SuperPOD

| Metric | A100 SuperPOD (IB HDR) | H100 SuperPOD (NVLink Switch) | Speedup |
| :--- | :--- | :--- | :--- |
| **1 DGX / 8 GPUs Dense Compute** | 2.5 PFLOP/s | 16 PFLOP/s | $6.4\times$ |
| **1 DGX / 8 GPUs Bisection BW** | $2,400\text{ GB/s}$ | $3,600\text{ GB/s}$ | $1.5\times$ |
| **1 DGX / 8 GPUs Reduce BW** | $150\text{ GB/s}$ | $450\text{ GB/s}$ | $3.0\times$ |
| **32 DGXs / 256 GPUs Dense Compute** | 80 PFLOP/s | 512 PFLOP/s | $6.4\times$ |
| **32 DGXs / 256 GPUs Bisection BW** | $6,400\text{ GB/s}$ | $57,600\text{ GB/s}$ | $9.0\times$ |
| **32 DGXs / 256 GPUs Reduce BW** | $100\text{ GB/s}$ | $450\text{ GB/s}$ | $4.5\times$ |

*Figure 26. DGX A100 vs DGX H100 32-node, 256 GPU NVIDIA SuperPOD Comparison*

### PCIe Gen 5
- Total bidirectional bandwidth of $128\text{ GB/sec}$ ($64\text{ GB/sec}$ per direction).
- Native PCIe atomic operations (CAS, exchange, fetch-add for 32/64-bit).
- Support for Single Root I/O Virtualization (SR-IOV) and NVLink cross-access from SR-IOV virtual functions.

---

## Security Enhancements and Confidential Computing

H100 is the world's first GPU to support hardware-based **Confidential Computing**.
- Extends the CPU Trusted Execution Environment (TEE) across PCIe at line rate using hardware **AES-256-GCM** encryption.
- **On-Die Root of Trust (RoT):** Authenticates GPU firmware and prevents tampering.
- **Measured Boot & Attestation:** Cryptographically verifies device identity, state, and firmware integrity against NVIDIA endorsement certificates (compliant with FIPS 140-3 Level 2 targets).

*Figure 27. Confidential Computing Protects Multiple ISV Scenarios*  
*Figure 28. Confidential Computing for Different Use Cases (Single Tenant Single/Multi-GPU, Multi-Tenant Single GPU)*  
*Figure 29. Confidential Federated Learning*  
*Figure 30. NVIDIA CC Off vs CC On VM Isolation*

---

## H100 Video / IO Features

### NVDEC for DL
H100 features seven (7) NVDEC units (compared to 5 on A100), enabling high-throughput decoding for deep learning pipelines.

#### Table 5. Comparison of A100 to H100 Video Decode (Number of 1080p30 Streams)

| GPU | HEVC Decode | H.264 Decode | VP9 Decode |
| :--- | :--- | :--- | :--- |
| **H100** | 340 | 170 | 260 |
| **A100** | 157 | 75 | 108 |

#### Table 6. H100 Hardware Decode Support

| Codec | Bit Depth | Chroma Format |
| :--- | :--- | :--- |
| **H.264** | 8-bit | 4:2:0 |
| **HEVC** | 8 / 10 / 12-bit | 4:2:0 / 4:4:4 |
| **VP9** | 8 / 10 / 12-bit | 4:2:0 |

### NVJPG (JPEG) Decode
H100 features seven single-core NVJPG hardware engines (replacing the single 5-core engine on A100), avoiding batch bottlenecks and accelerating heterogeneous images.

#### Table 7. NVJPG Decode Performance (Images/sec @ 1080p)

| GPU | JPEG 4:4:4 Decode | JPEG 4:2:0 Decode |
| :--- | :--- | :--- |
| **H100** | 3310 | 6350 |
| **A100** | 1490 | 2950 |

---

## Appendix A - NVIDIA DGX - The Foundational Building Blocks of Data Center AI

### NVIDIA DGX H100 - The World's Most Complete AI Platform
- 8x NVIDIA H100 Tensor Core GPUs (32 PFLOPS FP8 compute).
- 4x Third-generation NVSwitches delivering $7.2\text{ TB/sec}$ bidirectional NVLink bandwidth.
- 8x ConnectX-7 400 Gb/s InfiniBand / Ethernet adapters.
- 2x NVIDIA BlueField-3 DPUs for network, storage, and security offload.

#### Table 8. NVIDIA DGX H100 System Specifications

| Specification | DGX A100 | DGX H100 |
| :--- | :--- | :--- |
| **GPUs** | 8x NVIDIA A100 GPUs | 8x NVIDIA H100 GPUs |
| **Peak Compute** | 5 GPU Tensor PFLOP | 32 GPU Tensor PFLOP |
| **GPU Memory** | 80 GB per GPU / 640 GB total | 80 GB per GPU / 640 GB total |
| **System Memory** | 1 TB (up to 2 TB) 3200 MHz DDR4 | 2 TB |
| **Storage (Data Cache)** | 15 TB NVMe SSD (up to 30 TB) | 30 TB NVMe SSD (8x 3.84 TB) |
| **Storage (OS Drives)** | 2x 1.92 TB NVMe SSDs | 2x 1.92 TB NVMe SSDs |
| **Network Interfaces** | 8x ConnectX-6 HDR (200Gb/s) IB, 2x dual-port 100/200GbE | 4x OSFP ports (8x ConnectX-7 400Gb/s IB/Ethernet), 2x dual-port BlueField-3 DPUs |
| **Cooling** | Air | Air |

---

## Appendix B - NVIDIA CUDA Platform Update

The CUDA platform integrates compilers, high-performance libraries, frameworks, system software, and developer tools.

*Figure 31. The NVIDIA CUDA platform and its ecosystem*

### Compilation Architecture
- **NVVM IR:** LLVM 7-based compiler Intermediate Representation targeted by high-level languages (Python, Julia, Rust, Fortran, C++).
- **libNVVM:** Translates NVVM IR to PTX.
- **NVRTC:** Compiles CUDA C++ at runtime.
- **Unified Compilation (NVC++):** Automatically infers execution space (`__host__` vs `__device__`) without explicit qualifiers.

*Figure 32. High-level language Frontends*  
*Figure 33. NVCC split compilation model and NVC++ unified compilation model*  
*Figure 34. Unified toolchain supports execution-space inference*

```cpp
// NVCC requires explicit execution space annotations:
__host__ __device__ int square(int x) { return x * x; }
__global__ void square_elements(int* x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = square(x[i]);
}

// NVC++ automatically infers execution space:
int square(int x) { return x * x; }
__global__ void square_elements(int* x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = square(x[i]);
}
```

---

## Appendix C - Genomics Acceleration using DPX Instructions

The Smith-Waterman dynamic programming algorithm aligns DNA read sequences against reference genomes. Each matrix cell computation evaluates substitution scores and gap penalties:

$$\text{Substitution Matrix: } S(a_i, b_j) = \begin{cases} +3, & a_i = b_j \\ -3, & a_i \neq b_j \end{cases}$$

$$\text{Gap penalty: } W_k = k \cdot W_1, \quad W_1 = 2$$

Each cell update requires five basic operations:
1. Add score $x$ ($x = 3$) from diagonal element on a match.
2. Subtract score $x$ from diagonal element on a mismatch.
3. Subtract penalty $y$ ($y = 2$) on a vertical element insertion/deletion.
4. Subtract penalty $z$ ($z = 2$) on a horizontal element insertion/deletion.
5. Compute the maximum of the above four values (clamped to zero if negative).

H100 DPX instructions fuse these inner-loop operations, providing up to a $7\times$ speedup over A100 and up to $40\times$ over CPUs.

*Figure 35. NVIDIA CLARA Parabricks Accelerated Framework*  
*Figure 36. Smith-Waterman Algorithm for Genome sequencing*

---

## Notice & Trademarks

Information in this document is provided "as is" without warranties. NVIDIA reserves the right to make changes to specifications and product descriptions at any time without notice.

**Trademarks:** NVIDIA, the NVIDIA logo, CUDA, NVLink, NVSwitch, DGX, HGX, BlueField, TensorRT, and Omniverse are trademarks and/or registered trademarks of NVIDIA Corporation.

*Copyright © 2023 NVIDIA Corporation. All rights reserved.*