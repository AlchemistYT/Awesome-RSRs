# A Survey on Robust Sequential Recommendation: Fundamentals, Challenges, Taxonomy, and Future Directions

Survey Link: [Link to the paper on Arxiv](https://arxiv.org/abs/)

In the era of information overload, sequential recommender systems (SRSs) have become indispensable tools for modeling users' dynamic preferences, assisting personalized decision-making and information filtering, and thus attracting significant research and industrial attention. 
Conventional SRSs operate on a critical assumption that every input interaction sequence is reliably matched with the target subsequent interaction. However, this assumption is frequently violated in practice: real-world user behaviors are often driven by extrinsic motivations—such as behavioral randomness, contextual influences, and malicious attacks—which introduce perturbations into interaction sequences. These perturbations result in mismatched input-target pairs, termed as \textit{unreliable instances}, which corrupt sequential patterns, mislead model training and inference, and ultimately degrade recommendation accuracy. To mitigate these issues, the study of Robust Sequential Recommenders (RSRs) has thus emerged as a focal point.
This survey provides the first systematic review of advances in RSR research. We begin with a thorough analysis of unreliable instances, detailing their causes, manifestations, and adverse impacts. We then delineate the unique challenges of RSRs, which are absent in non-sequential settings and general denoising tasks. Subsequently, we present a holistic taxonomy of RSR methodologies and a systematic comparative analysis based on eight key properties, critically evaluating the strengths and limitations of existing approaches. We also summarize standard evaluation metrics and benchmarks. Finally, we identify open issues and discuss promising future research directions.

# Table of Contents
- [Background and Fundamentals](#Background-and-Fundamentals)
- [Unreliable Instances in Sequential Recommendation](#Unreliable-Instances-in-Sequential-Recommendation)
- [Taxonomy of RSRs](#Taxonomy-of-RSRs)
- [Evaluation Framework of RSRs](#Evaluation-Framework-of-RSRs)
  - [Architecture-centric RSRs](#Architecture-centric-RSRs)
  - [Data-centric RSRs](#Data-centric-RSRs)
  - [Learning-centric RSRs](#Learning-centric-RSRs)
  - [Inference-centric RSRs](#Inference-centric-RSRs )





# Background and Fundamentals
![Compared Image](instance.png)

An example showing how an interaction sequence is split into data instances by a sliding window.
In each step of sliding, the last item in the sliding window is treated as the target of an instance, while the
preceding items in the window serve as the input.

![Compared Image](robustness.png)

- **Training-phase Robustness**: During training, the RSR must precisely identify items within
the input sequence that are genuinely relevant to the target (i.e., driven by the same intrinsic
motivations). By focusing on these items, the model avoids learning erroneous patterns from
perturbations.
- **Inference-phase Robustness**: During inference, the target is unobservable. The RSR must
infer the underlying motivations from the input and ensure the recommendation list provides
complete coverage for these motivations, without being skewed by perturbations [26, 27].


# Unreliable Instances in Sequential Recommendation
![Compared Image](causes.png)

# Taxonomy of RSRs

![Compared Image](taxonomy.png)

- **Architecture-centric RSRs** embed robustness directly into the model architecture through perturbation-resistant designs (e.g., gating mechanisms or diffusion models), ensuring stable internal representations despite perturbed sequences.
- **Data-centric RSRs** operate at the Instance Construction stage, focusing on cleansing training data before or during model training. They proactively identify and rectify mismatched input-target pairs (via selection, reweighting, or correction), thereby eliminating erroneous sequential patterns from the training process.
- **Learning-centric RSRs** introduce robustness during model training. Rather than modifying the data or core architecture, they leverage specialized training strategies (e.g., adversarial training, robust loss functions) to guide the model to learn genuine user preferences while diminishing the influence of unreliable instances.
- **Inference-centric RSRs** address robustness at the final model inference stage. Acknowledging that real-time input sequences may contain perturbations, these methods generate comprehensive and balanced recommendation lists that fully capture users’ underlying motivations and avoid being skewed by perturbations.

# Evaluation Framework of RSRs

- **Multi-cause Robustness**: Ability to address diverse extrinsic motivations (behavioral randomness, contextual influences, malicious manipulations) that induce unreliable instances.
- **Dual-manifestation Robustness**: Capacity to handle both complete mismatch (perturbed targets) and partial mismatch (perturbed inputs).
- **Dual-phase Robustness**: Capability to satisfy robustness requirements (Section~\ref{sec:rsr-definition}) in both the training phase and the inference phase.
- **Motivation Transformation Awareness**: Ability to model transformations between intrinsic and extrinsic motivations over time.
- **Generality**: Compatibility with existing SRSs without extensive architectural modifications.
- **Data Accessibility**: Independence on side information (e.g., item attributes, user demographics) beyond raw user-item interaction data.
- **Scalability**: Efficiency in large-scale real-world scenarios.
- **Theoretical Grounding**: Existence of formal theoretical guarantees (e.g., robustness bounds, convergence proofs) for the method’s efficacy.

## Architecture-centric RSRs

| Category | Subcategory | Method | Venue-Year | P1: Multi-cause Robustness | P2: Dual-manifestation Robustness | P3: Dual-phase Robustness | P4: Motivation Transformation Awareness | P5: Generality | P6: Data Accessibility | P7: Scalability | P8: Theoretical Grounding |
|----------|-------------|--------------------------------------------------------|------------|----------------------|-----------------------------|----------------------|-----------------------------------|-----------|-------------------|-------------|---------------------|
| **Attention Mechanism** | **Basic Attention** | Neural Attentive Session-based Recommendation ([NARM](https://dl.acm.org/doi/10.1145/3132847.3132926)) | CIKM'17 | △ | △ | △ | × | × | ○ | △ | × |
| | | STAMP | KDD'18 | △ | △ | △ | × | × | ○ | ○ | × |
| | **Self Attention** | SASRec | ICDM'18 | △ | △ | △ | × | × | ○ | △ | × |
| | | BERT4Rec | CIKM'19 | △ | △ | △ | × | × | ○ | △ | × |
| | | DT4Rec | CIKM'21 | △ | △ | △ | × | × | ○ | △ | × |
| | | STOSA | WWW'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | ADT | KDD'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | AC-TSR | CIKM'23 | △ | △ | △ | × | × | ○ | △ | × |
| | **Sparse Attention** | DSAN | AAAI'21 | △ | △ | △ | × | × | ○ | △ | × |
| | | Locker | CIKM'21 | △ | △ | △ | × | × | ○ | △ | × |
| | | RecDenoiser | RecSys'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | RETR | WWW'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | DPDM | ESWA'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | AutoDisenSeq | TOIS'25 | △ | △ | △ | × | × | ○ | △ | × |
| | **Knowledge-enhanced Attention** | IARN | CIKM'17 | △ | △ | △ | × | × | × | × | × |
| | | NOVA | AAAI'21 | △ | △ | △ | × | × | × | × | × |
| | | KGDPL | TKDE'24 | △ | △ | △ | × | × | × | × | × |
| **Memory Networks** | **Key-value MN** | RUM | WSDM'18 | △ | △ | △ | × | × | ○ | × | × |
| | | MAGNN | AAAI'20 | △ | △ | △ | × | × | ○ | × | × |
| | | DMAN | AAAI'21 | △ | △ | △ | × | × | ○ | × | × |
| | | MASR | CIKM'22 | △ | △ | △ | × | × | ○ | △ | × |
| | **Knowledge-enhanced MN** | KSR | SIGIR'18 | △ | △ | △ | × | × | × | × | × |
| | | CmnRec | TKDE'23 | △ | △ | △ | × | × | × | △ | × |
| | | LMN | WWW'25 | △ | △ | △ | × | × | × | △ | × |
| **Gating Networks** | **Basic Gating Networks** | HGN | KDD'19 | △ | △ | △ | × | × | ○ | △ | × |
| | | M3R | WWW'19 | △ | △ | △ | × | × | ○ | △ | × |
| | | ASPPA | IJCAI'20 | △ | △ | △ | × | × | ○ | △ | × |
| | | STAR-Rec | SIGIR'25 | △ | △ | △ | × | × | ○ | △ | × |
| | **Consistency-aware Gating Networks** | π-Net | SIGIR'19 | △ | △ | △ | × | × | ○ | △ | × |
| | | CAR | WSDM'20 | △ | △ | △ | × | × | ○ | △ | × |
| | | MAN | IJCAI'20 | △ | △ | △ | × | × | ○ | △ | × |
| | | S2PNM | TKDE'22 | △ | △ | △ | × | × | ○ | △ | × |
| **Graph Neural Networks** | **Basic GNNs** | GCSAN | IJCAI'19 | △ | △ | △ | × | × | ○ | △ | × |
| | | SRGNN | AAAI'19 | △ | △ | △ | × | × | ○ | △ | × |
| | **Knowledge-enhanced GNNs** | IMFOU | WWW'20 | △ | △ | △ | × | × | × | × | × |
| | | FAPAT | NeurIPS'23 | △ | △ | △ | × | × | × | × | × |
| | | I-DIDA | TOIS'25 | △ | △ | △ | △ | × | × | × | × |
| | | TGODE | KDD'25 | △ | △ | △ | △ | × | × | × | × |
| | **Sparse GNNs** | SLED | TOIS'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | MAERec | SIGIR'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | GDRN | TKDE'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | RAIN | NN'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | MA-GCL4SR | TKDD'25 | △ | △ | △ | × | × | ○ | △ | × |
| **Time-frequency Analysis** | **Fourier Transform** | FMLP | WWW'22 | △ | △ | △ | × | × | ○ | ○ | × |
| | | SLIME4Rec | ICDE'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | BSARec | AAAI'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | FDCLRec | CIKM'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | LFDFSR | SMC'24 | △ | △ | △ | × | × | × | △ | × |
| | | END4Rec | WWW'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | DPCPL | CoRR'24 | △ | △ | △ | × | × | ○ | △ | × |
| | | SSR | IJCAI'25 | △ | △ | △ | × | × | ○ | ○ | × |
| | | Oracle4Rec | WSDM'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | DIFF | SIGIR'25 | △ | △ | △ | × | × | × | △ | × |
| | **Wavelet Transform** | Wavelet | CISM'25 | △ | △ | △ | × | × | ○ | ○ | × |
| | | WaveRec | ICTIR'25 | △ | △ | △ | × | × | ○ | ○ | × |
| | **Hybrid Frequency Attention** | FEARec | SIGIR'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | MUFFIN | CIKM'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | FICLRec | IPM'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | MIRRN | KDD'25 | △ | △ | △ | × | × | ○ | × | × |
| **Diffusion Models** | **Basic Diffusion** | DiffuRec | TOIS'23 | △ | △ | △ | × | × | ○ | × | × |
| | | CF-Diff | SIGIR'24 | △ | △ | △ | × | × | ○ | × | × |
| | **Conditional Diffusion** | CDDRec | PAKDD'24 | △ | △ | △ | × | × | ○ | × | × |
| | | SeeDRec | IJCAI'24 | △ | △ | △ | × | × | × | × | × |
| | | M3BSR | SIGIR'25 | △ | △ | △ | × | × | × | × | × |
| | | TDM | SIGIR'25 | △ | △ | △ | × | × | ○ | × | × |
| | **Advanced Diffusion** | PDRec | AAAI'24 | △ | △ | △ | × | × | ○ | × | × |
| | | FMRec | IJCAI'25 | △ | △ | △ | × | × | ○ | × | × |
| | | ADRec | KDD'25 | △ | △ | △ | × | × | ○ | × | × |
| | | DiffDiv | SIGIR'25 | △ | △ | △ | × | × | ○ | × | × |
| | | DiQDiff | WWW'25 | △ | △ | △ | × | × | ○ | × | × |

## Data-centric RSRs

| Category | Subcategory | Method | Venue-Year | Multi-cause Robustness | Dual-manifestation Robustness | Dual-phase Robustness | Motivation Transformation Awareness | Generality | Data Accessibility | Scalability | Theoretical Grounding |
|----------|-------------|--------|------------|----------------------|-----------------------------|----------------------|-----------------------------------|-----------|-------------------|-------------|---------------------|
| **Instance Selection** | **Loss-uncertainty Modeling** | BERD | IJCAI'21 | △ | △ | △ | × | ○ | ○ | △ | × |
| | | BERD+ | TOIS'23 | △ | △ | △ | × | ○ | × | △ | × |
| | | PLD | WWW'25 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | **Semantic Modeling** | LoRec | SIGIR'24 | △ | △ | △ | × | ○ | × | × | × |
| | | ConsRec | CoRR'25 | △ | △ | △ | × | × | × | × | × |
| **Instance Correction** | **Data-driven Correction** | STEAM | WWW'23 | △ | ○ | △ | × | ○ | ○ | × | × |
| | | BirDRec | NeurIPS'23 | △ | ○ | △ | × | ○ | ○ | △ | ○ |
| | | DR4SR | KDD'24 | △ | ○ | △ | × | ○ | ○ | △ | × |
| | **LLM-guided Correction** | LLM4DSR | TOIS'25 | △ | △ | △ | × | ○ | × | × | × |
| | | LLM4RSR | AAAI'25 | △ | ○ | △ | × | ○ | × | △ | × |
| | | IADSR | CIKM'25 | △ | △ | △ | × | ○ | △ | × | × |
| **Data Augmentation** | **Rule-based Augmentation** | PERIS | CIKM'22 | △ | △ | △ | × | ○ | △ | ○ | × |
| | | MRFI | RecSys'23 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | ASSR | CAIT'24 | △ | △ | △ | × | ○ | ○ | △ | × |
| | **Model-base Augmentation** | Diff4Rec | MM'23 | △ | △ | △ | × | ○ | ○ | × | × |
| | | DiffuASR | CIKM'23 | △ | △ | △ | × | ○ | ○ | × | × |
| | | SSDRec | ICDE'24 | △ | △ | △ | × | ○ | ○ | × | × |
| | | CeDRec | KBS'25 | △ | △ | △ | × | × | ○ | × | × |
| | | TTA | SIGIR'25 | △ | △ | △ | × | ○ | ○ | ○ | × |

# Learning-centric RSRs

| Category | Subcategory | Method | Venue-Year | Multi-cause Robustness | Dual-manifestation Robustness | Dual-phase Robustness | Motivation Transformation Awareness | Generality | Data Accessibility | Scalability | Theoretical Grounding |
|----------|-------------|--------|------------|----------------------|-----------------------------|----------------------|-----------------------------------|-----------|-------------------|-------------|---------------------|
| **Adversarial Training** | **Gradient-based AT** | AdvTrain | RecSys'22 | △ | △ | △ | × | ○ | ○ | × | × |
| | | PamaCF | NeurIPS'24 | △ | △ | △ | × | ○ | ○ | × | ○ |
| | | VAT | RecSys'24 | △ | △ | △ | × | ○ | ○ | × | × |
| | | MVS | TOIS'23 | △ | △ | △ | × | ○ | ○ | × | × |
| | | Qian et al. | TKDD'25 | △ | △ | △ | × | ○ | ○ | × | × |
| | **Sequence-based AT** | AdvGraph | SIGIR'22 | △ | △ | △ | × | ○ | ○ | × | × |
| | | DARTS | CoRR'25 | △ | △ | △ | × | ○ | ○ | × | × |
| **Reinforcement Learning** | **Sequence Construction** | HRL | AAAI'19 | △ | △ | △ | × | × | ○ | × | × |
| | | SAR | RecSys'21 | △ | △ | △ | × | × | ○ | × | × |
| | | RLNF | SIGIR'21 | △ | △ | △ | × | × | ○ | × | × |
| | | HRL4Ba | ICASSP'22 | △ | △ | △ | × | × | ○ | × | × |
| | | MHRR | TSC'23 | △ | △ | △ | × | × | × | × | × |
| | **Feature Selection** | KERL | SIGIR'20 | △ | △ | △ | × | × | × | × | × |
| | | MARIS | SIGIR'22 | △ | △ | △ | × | × | × | × | × |
| **Distributionally Robust Optimization** | **Partition-based DRO** | S-DRO | WWW'22 | △ | △ | △ | ○ | ○ | ○ | ○ | △ |
| | | IDEA | SIGIR'25 | △ | △ | △ | ○ | ○ | ○ | ○ | △ |
| | **Distribution-based DRO** | DROS | SIGIR'23 | △ | △ | △ | ○ | ○ | ○ | ○ | △ |
| | | RSR | SIGIR'23 | △ | △ | △ | ○ | ○ | ○ | ○ | △ |
| | **Model-based DRO** | DePoD | TOIS'24 | △ | △ | △ | ○ | ○ | ○ | × | △ |
| | | E-NSDE | NeurIPS'24 | △ | △ | △ | ○ | ○ | ○ | × | △ |
| | | IDURL | SIGIR'25 | △ | △ | △ | ○ | ○ | × | × | △ |
| **Self-supervised Learning** | **Feature-level SSL** | DCRec | WWW'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | FCLRec | TKDE'24 | △ | △ | △ | × | × | × | △ | × |
| | | SGCL | TOIS'25 | △ | △ | △ | × | × | ○ | △ | × |
| | | BHGCL | TOIS'25 | △ | △ | △ | × | × | × | △ | × |
| | **Sequence-level SSL** | RAP | IJCAI'21 | △ | △ | △ | × | × | ○ | △ | × |
| | | CLEA | SIGIR'21 | △ | △ | △ | × | × | ○ | △ | × |
| | | ICL | WWW'22 | △ | △ | △ | × | × | ○ | × | × |
| | | BNCL | CIKM'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | CL4Rec | ICDE'22 | △ | △ | △ | × | × | ○ | ○ | × |
| | | ICSRec | WSDM'24 | △ | △ | △ | × | × | ○ | ○ | × |
| | | IOCLRec | AAAI'25 | △ | △ | △ | × | × | ○ | △ | × |
| | **Model-level SSL** | DuoRec | WSDM'22 | △ | △ | △ | × | ○ | ○ | △ | × |
| | | SHT | KDD'22 | △ | △ | △ | × | ○ | ○ | × | × |
| | | ContrastVAE | CIKM'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | AdaptSSR | NeurIPS'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | MCLRec | SIGIR'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | MStein | WWW'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | LMA4Rec | TKDE'24 | △ | △ | △ | × | × | ○ | △ | × |
| **Causal Learning** | **Counterfactual Sequence Intervention** | CauseRec | SIGIR'21 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | CASR | TKDE'23 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | PACIFIC | CIKM'24 | △ | △ | △ | × | × | ○ | △ | × |
| | **Structural Causal Modeling** | CoDeR | AAAI'25 | △ | △ | △ | × | × | ○ | × | × |
| | | CSRec | SIGIR'25 | △ | △ | △ | × | × | ○ | × | × |
| | | DePOI | SIGIR'25 | △ | △ | △ | × | × | × | × | × |
| **Curriculum Learning** | **Static Hardness Measurement** | GNNO | SIGIR'23 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | Diff4Rec | MM'23 | △ | △ | △ | × | ○ | ○ | △ | × |
| | | MELT | SIGIR'23 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | **Dynamic Hardness Measurement** | CCl | CIKM'21 | △ | △ | △ | × | ○ | × | ○ | × |
| | | HSD | CIKM'22 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | CDR | NeurIPS'21 | △ | △ | △ | × | ○ | ○ | ○ | × |
| | | EXHANS | WSDM'25 | △ | △ | △ | × | ○ | ○ | ○ | × |

# Inference-centric RSRs

| Category | Subcategory | Method | Venue-Year | Multi-cause Robustness | Dual-manifestation Robustness | Dual-phase Robustness | Motivation Transformation Awareness | Generality | Data Accessibility | Scalability | Theoretical Grounding |
|----------|-------------|--------|------------|----------------------|-----------------------------|----------------------|-----------------------------------|-----------|-------------------|-------------|---------------------|
| **Recommendation Calibration** | **Post-processing Calibration** | CaliRec | RecSys'18 | △ | △ | △ | × | ○ | × | × | × |
| | | Calib-Opt | RecSys'21 | △ | △ | △ | × | ○ | × | × | × |
| | | TecRec | WWW'21 | △ | △ | △ | × | ○ | × | × | × |
| | | MCF | WSDM'23 | △ | △ | △ | × | ○ | × | × | × |
| | | LeapRec | CIKM'24 | △ | △ | △ | × | ○ | × | × | × |
| | **End-to-End Calibration** | DACSR | Applied Science'22 | △ | △ | △ | × | × | × | △ | × |
| | | CSBR | Applied Intelligence'23 | △ | △ | △ | × | × | × | △ | × |
| | | CaliTune | RecSys'25 | △ | △ | △ | × | × | × | △ | × |
| **Multi-interest Disentanglement** | **Adaptive Interest Disentanglement** | MIND | CIKM'19 | △ | △ | △ | × | × | × | × | × |
| | | MCPRN | IJCAI'19 | △ | △ | △ | × | × | × | × | × |
| | | MRIF | SIGIR'20 | △ | △ | △ | × | × | ○ | × | × |
| | | MGNM | SIGIR'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | BaM | KDD'24 | △ | △ | △ | × | × | ○ | ○ | × |
| | | ComiRec | KDD'20 | △ | △ | △ | × | × | × | × | × |
| | **Regularized Interest Disentanglement** | IDSR | CIKM'20 | △ | △ | △ | × | × | × | × | × |
| | | MDSR | TOIS'21 | △ | △ | △ | × | × | × | × | × |
| | | CMI | SIGIR'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | Re4 | WWW'22 | △ | △ | △ | × | × | ○ | △ | × |
| | | TiMiRec | CIKM'22 | △ | △ | △ | × | × | ○ | × | × |
| | | REMI | RecSys'23 | △ | △ | △ | × | × | ○ | × | × |
| | | DisMIR | KDD'24 | △ | △ | △ | × | × | ○ | × | × |
| | **Knowledge-guided Interest Disentanglement** | MISSRec | MM'23 | △ | △ | △ | × | × | × | × | × |
| | | CoLT | RecSys'23 | △ | △ | △ | × | × | ○ | △ | × |
| | | Trinity | KDD'24 | △ | △ | △ | × | × | × | × | × |
| | | CoMoRec | MM'24 | △ | △ | △ | × | × | × | × | × |
| | | SimEmb | WSDM'24 | △ | △ | △ | × | × | × | × | × |
| | | HORAE | TOIS'25 | △ | △ | △ | × | × | ○ | × | × |




