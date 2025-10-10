<div id="top">

# GE2EAD

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


Collects papers on autonomous driving E2E learning and VLM/VLA, with organized research branches and trends in these fields.


## Table of Contents

- [Mindmap, Top Methods](#mindmap-top-methods)
- [Papers](#papers)
    - [Conventional End-to-End Methods](#conventional-end-to-end-methods)
    - [VLM-Centric End-to-End Methods](#vlm-centric-end-to-end-methods)
    - [Hybrid End-to-End Methods](#hybrid-end-to-end-methods)
- [Dataset](#dataset)
    - [Normal Dataset](#normal-dataset)
    - [Vision Language Dataset](#vision-language-dataset)
- [License](#License)
- [Citation](#citation)
- [Contact](#contact)

## Mindmap, Top Methods

<div align="center">
  <a href="https://github.com/AutoLab-SAI-SJTU/GE2EAD">
    <img 
      src="images/coggle.png" 
      alt="GE2EAD Mindmap Logo" 
      style="width: 100%; height: auto; max-width: 800px;" 
    >
  </a>
  <h3 align="center">GE2EAD Mindmap</h3>
</div>

<div align="center">
  <a href="https://github.com/AutoLab-SAI-SJTU/GE2EAD">
    <img 
      src="images/top_methods.png" 
      alt="GE2EAD Mindmap Logo" 
      style="width: 100%; height: auto; max-width: 800px;" 
    >
  </a>
  <h3 align="center">Top Methods</h3>
</div>

## Papers
### Conventional End-to-End Methods
<details open>
<summary>2025</summary>

* ​**EvaDrive**​: Evolutionary Adversarial Policy Optimization for End-to-End Autonomous Driving [[Paper](https://www.arxiv.org/pdf/2508.09158)] 
    

    Summary: This paper integrates trajectory generation and evaluation into a closed-loop system through an innovative multi-objective reinforcement learning framework and adversarial strategy optimization, significantly enhancing the robustness and flexibility of autonomous driving planning. It also supports diverse driving styles and has broad application potential.
* ​**ReconDreamer-RL**​: Enhancing Reinforcement Learning via Diffusion-based Scene Reconstruction [[Paper](https://arxiv.org/pdf/2508.08170)] [[Code](https://github.com/GigaAI-research/ReconDreamer-RL)] 
    

    Summary: ReconDreamer-RL is a framework that enhances end-to-end reinforcement learning through scene reconstruction and video diffusion technologies, aiming to optimize the training performance of autonomous driving models, especially in handling complex scenes and corner cases in closed-loop environments.
* ​**GMF-Drive**​: Gated Mamba Fusion with Spatial-Aware BEV Representation for End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2508.06113)] 
    

    Summary: GMF-Drive significantly enhances the efficiency of multimodal fusion and the performance of autonomous driving through innovative ​geometrically enhanced representations and spatially aware state-space models​.
* ​**DistillDrive**​: End-to-End Multi-Mode Autonomous Driving Distillation by Isomorphic Hetero-Source Planning Model [[Paper](https://www.arxiv.org/pdf/2508.05402)] [[Code](https://github.com/YuruiAI/DistillDrive)] 
    

    Summary: This paper employs a multimodal decoupling planning model based on structured scene representation as the teacher model, guiding the student model to learn ​multimodal motion features through distillation​, in order to address the limitations of existing end-to-end models in single-objective imitation learning.
* ​**GEMINUS**​: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving [[Paper](https://www.arxiv.org/pdf/2507.14456)] [[Code](https://github.com/newbrains1/GEMINUS)] 
    

    Summary: GEMINUS effectively combines global experts with scene-adaptive expert groups through dual-sensing routers, achieving a balance between adaptability and robustness in complex and diverse traffic scenarios.
* ​**DiVER**​: Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation [[Paper](https://arxiv.org/pdf/2507.04049)] 
    

    Summary: DiVER is an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories, effectively addressing the mode collapse problem inherent in imitation learning.
* ​**World4Drive**​: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model, **ICCV 2025** [[Paper](https://arxiv.org/pdf/2507.00603)] [[Project](https://github.com/ucaszyp/World4Drive)] 
    

    Summary: By simulating the evolution process of the ​physical world under different driving intentions​, the generation and evaluation of multimodal trajectories are achieved, which is close to the decision-making logic of human drivers.
* ​**FocalAD**​: Local Motion Planning for End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2506.11419)] 
    

    Summary: FocalAD refines planning by focusing on critical local neighbors and enhancing local motion representations.
* ​**GaussianFusion**​: Gaussian-Based Multi-Sensor Fusion for End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2506.00034)] [[Code](https://github.com/Say2L/GaussianFusion)] 
    

    Summary: Utilizing intuitive and compact Gaussian representations as intermediate carriers, GaussianFusion iteratively refine trajectory predictions through interactions with the rich spatial and semantic information within these Gaussians.
* ​**CogAD**​: Cognitive-Hierarchy Guided End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2505.21581)] 
    

    Summary: CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning.

* ​**DiffE2E**​: Rethinking End-to-End Driving with a Hybrid Action Diffusion and Supervised Policy [[Paper](https://arxiv.org/pdf/2505.19516)] [[Project](https://infinidrive.github.io/DiffE2E/)] 
    

    Summary: DiffE2 integrates a Transformer-based hybrid diffusion-supervised decoder and introduces a collaborative training mechanism, which effectively combines the advantages of diffusion and supervision strategies.
* ​**TransDiffuser**​: End-to-end Trajectory Generation with Decorrelated Multi-modal Representation for Autonomous Driving [[Paper](https://arxiv.org/pdf/2505.09315)] 
    

    Summary: TransDiffuser, an end-to-end generative trajectory model for autonomous driving based on "encoder-decoder", and introduces a multimodal representation decorrelation optimization mechanism to encourage sampling of more diverse trajectories from continuous space.
* ​**MomAD**​: Don’t Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2503.03125)] [[Code](https://github.com/adept-thu/MomAD)] 
    

    Summary: MomAD effectively alleviates the key challenges of trajectory mutation and perception instability in end-to-end autonomous driving through the ​momentum mechanism​. Trajectory momentum aims to stabilize and optimize trajectory prediction by keeping candidate trajectories aligned with historical trajectories.
* ​**Consistency**​: Predictive Planner for Autonomous Driving with Consistency Models [[Paper](https://arxiv.org/pdf/2502.08033)] 
    

    Summary: Consistency leverage the consistency model to build a predictive planner that samples from a joint distribution of ego and surrounding agents, conditioned on the ego vehicle’s navigational goal.
* ​**ARTEMIS**​: Autoregressive End-to-End Trajectory Planning with Mixture of Experts for Autonomous Driving [[Paper](https://arxiv.org/abs/2504.19580)] 
    

    Summary: Using the hybrid expert model MoE to improve E2E, the autoregressive planning module with MOE gradually generates trajectory waypoints through a sequential decision process, while dynamically selecting the expert network that best suits the current driving scenario.
* ​**TTOG**​: Two Tasks, One Goal: Uniting Motion and Planning for Excellent End To End Autonomous Driving Performance [[Paper](https://arxiv.org/pdf/2504.12667)] 
    

    Summary: TTOG introduces a new method to ​unify motion and planning tasks​, allowing the planning task to benefit from motion data, significantly improving the performance and generalization ability of the planning task.
* ​**DiffusionDrive**​: Truncated Diffusion Model for End-to-End Autonomous Driving, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2411.15139)] [[Code](https://github.com/hustvl/DiffusionDrive)] 
    

    Summary: For the first time, the diffusion model was introduced into the field of end-to-end autonomous driving, and a truncated diffusion strategy was proposed, which solved the problems of mode collapse and excessive computation when the traditional diffusion strategy was applied in traffic scenarios.
* ​**WoTE**​: End-to-End Driving with Online Trajectory Evaluation via BEV World Model [[Paper](https://arxiv.org/pdf/2504.01941)] [[Code](https://github.com/liyingyanUCAS/WoTE)] 
    

    Summary: The BEV world model is used to predict the future state of BEVs for trajectory evaluation. Compared with the image-level world model, the proposed BEV world model has lower latency and can be seamlessly supervised using an off-the-shelf BEV spatial traffic simulator.
* ​**DMAD**​: Divide and Merge: Motion and Semantic Learning in End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2502.07631)] [[Code](https://github.com/shenyinzhe/DMAD)] 
    

    Summary: A novel parallel detection, tracking, and prediction method that ​separates semantic learning from motion learning​. This architecture separates the gradient backpropagation between the two types of tasks to eliminate negative transfer, and merges similar tasks to exploit the correlation between tasks and promote positive transfer.
* ​**Centaur**​: Robust End-to-End Autonomous Driving with Test-Time Training [[Paper](https://arxiv.org/abs/2503.11650)] 
    

    Summary: Application of **Test-Time Training (TTT)** in End-to-End Autonomous Driving to enhance robustness.
* ​**Drive in Corridors**​: Enhancing the Safety of End-to-end Autonomous Driving via Corridor Learning and Planning [[Paper](https://arxiv.org/abs/2504.07507)] 
    

    Summary: The concept of a safe corridor in the field of robot planning is introduced into end-to-end autonomous driving as an explicit spatiotemporal constraint to enhance safety.
* ​**BridgeAD**​: Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction and Planning, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2503.14182v1)] [[Code](https://github.com/fudan-zvg/BridgeAD)] 
    

    Summary: End-to-end autonomous driving is enhanced by incorporating historical predictions of the current frame into the perception module, and incorporating historical predictions and planning of future frames into the motion planning module.
* ​**Hydra-MDP++**​: Advancing End-to-End Driving via Expert-Guided Hydra-Distillation [[Paper](https://arxiv.org/pdf/2503.12820)] [[Code](https://github.com/NVlabs/Hydra-MDP)] 
    

    Summary: Hydra-MDP++ introduces a new teacher-student knowledge distillation framework with a multi-head decoder that can learn from human demonstrations and rule-based experts.
* ​**DiffAD**​: A Unified Diffusion Modeling Approach for Autonomous Driving [[Paper](https://arxiv.org/pdf/2503.12170)] 
    

    Summary: DiffAD redefines autonomous driving as a ​conditional image generation task​. By rasterizing heterogeneous targets onto a unified bird’s-eye view (BEV) and modeling their latent distribution, DiffAD unifies various driving objectives and jointly optimizes all driving tasks in a single framework.
* ​**GoalFlow**​: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2503.05689)] [[Code](https://github.com/YvanYin/GoalFlow)] 
    

    Summary: It has come up with an end-to-end autonomous driving method called GoalFlow, which generates high-quality multimodal trajectories by introducing target points to constrain the generation process using flow matching.
* ​**HiP-AD**​: Hierarchical and Multi-Granularity Planning with Deformable Attention for Autonomous Driving in a Single Decoder, **ICCV 2025** [[Paper](https://arxiv.org/pdf/2503.08612)] [[Code](https://github.com/nullmax-vision/HiP-AD)] 
    

    Summary: A unified decoder is designed to take as input hybrid task queries (detection, map understanding, planning), allowing planning and perception tasks to exchange information in BEV space and planning queries to interact with image space.
* ​**LAW**​: Enhancing End-to-End Autonomous Driving with Latent World Model, **ICLR 2025** [[Paper](https://arxiv.org/pdf/2406.08481)] [[Code](https://github.com/BraveGroup/LAW)] 
    

    Summary: This paper proposes a self-supervised learning method based on the LAtent World Model (LAW) to optimize the scene feature representation and future trajectory prediction.
* ​**DriveTransformer**​: Unified Transformer for Scalable End-to-End Autonomous Driving, **ICLR 2025** [[Paper](https://arxiv.org/pdf/2503.07656)] [[Code](https://github.com/Thinklab-SJTU/DriveTransformer)] 
    

    Summary: A unified architecture without BEV is designed with a Decoder as the core, featuring task parallelism, sparse representation (task queries interact directly with raw sensor features), and stream processing.
* ​**UncAD**​: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty, **ICRA 2025** [[Paper](https://arxiv.org/pdf/2504.12826)] [[Code](https://github.com/pengxuanyang/UncAD)] 
    

    Summary: UncAD effectively utilize the map uncertainty to produce robust and safe planning results via Uncertainty-Guided Planning strategy and Uncertainty-Collision-Aware Planning Selection module.
* ​**RAD**​: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning [[Paper](https://arxiv.org/pdf/2502.13144)] [[Project](https://hgao-cv.github.io/RAD/)] 
    

    Summary: Establish a closed-loop reinforcement learning (RL) training paradigm based on 3DGS to build a realistic digital replica of the real physical world, enabling AD policies to explore the state space and handle out-of-distribution (OOD) scenarios.
* ​**OAD**​: Trajectory Offset Learning: A Framework for Enhanced End-to-End Autonomous Driving [[Paper](https://www.researchgate.net/publication/388891609_Trajectory_Offset_Learning_A_Framework_for_Enhanced_End-to-End_Autonomous_Driving)] [[Code](https://github.com/wzn-cv/OAD)] 
    

    Summary: OAD enhanced framework built upon the VAD architecture, which introduces a novel paradigm shift from direct trajectory prediction to ​trajectory offset learning​. Leverage trajectory vocabulary to learn offsets instead of learning trajectories directly.

</details>
<details open>
<summary>2024</summary>

- **GaussianAD**: Gaussian-Centric End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2412.10371)] [[Code](https://github.com/wzzheng/GaussianAD)]
   
    Summary: The author extensively and sparsely describes the scene by using 3D semantic Gaussian, efficiently performs 3D perception with sparse convolution, uses Gaussian 3D flow prediction, and plans the self-vehicle trajectory accordingly with the goal of future scene prediction.

- **MA2T**: Module-wise Adaptive Adversarial Training for End-to-end Autonomous Driving [[Paper](https://arxiv.org/pdf/2409.07321)]
   
    Summary: Adversative training is applied to end-to-end AD to improve the robustness under different adversative attacks by integrating module-level noise injection and dynamic weight accumulation adaptation.

- **Hint-AD**: Holistically Aligned Interpretability in End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2409.06702)] [[Project](https://air-discover.github.io/Hint-AD/)] [[Code](https://github.com/Robot-K/Hint-AD)]
   
    Summary: By combining the intermediate output and the token mixer subnetwork, the language generated by the model is aligned with the overall perception-prediction-planning output of the AD model.

- **DRAMA**: An Efficient End-to-end Motion Planner for Autonomous Driving with Mamba, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2408.03601)] [[Project](https://chengran-yuan.github.io/DRAMA/)] [[Code](https://github.com/Chengran-Yuan/DRAMA)]
   
    Summary: Using the Mamba-embedded encoder-decoder architecture, the encoder is used to fuse fuses features from the camera and LiDAR BEV images, and the decoder is used to generate motion trajectories.

- **PPAD**: Iterative Interactions of Prediction and Planning for End-to-end Autonomous Driving, **ECCV 2024** [[Paper](https://arxiv.org/pdf/2311.08100)] [[Code](https://github.com/zlichen/PPAD)]
   
    Summary: The prediction and planning processes are carried out alternately at each time step, rather than a single sequential process of prediction and planning.

- **BEV-Planner**: Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?, **CVPR 2024** [[Paper](https://arxiv.org/pdf/2312.03031)] [[Code](https://github.com/NVlabs/BEV-Planner)]
   
    Summary: This paper analyzes the problem of excessive reliance on the ego status in planning tasks by existing methods, proposes new baseline methods and evaluation metrics, and emphasizes the importance of developing more suitable datasets.

- **EfficientFuser**: Efficient Fusion and Task Guided Embedding for End-to-end Autonomous Driving [[Paper](https://arxiv.org/pdf/2407.02878)]
   
    Summary: Reduce required parameters and computation with EfficientViT lightweight neural networks.

- **UAD**: End-to-End Autonomous Driving without Costly Modularization and 3D Manual Annotation [[Paper](https://arxiv.org/pdf/2406.17680)]
   
    Summary: Using unsupervised frameworks eliminate the need for expensive 3D annotation and use self-supervised training strategies to enhance the planning robustness in the transition scene.

- **Hydra-MDP**: End-to-end Multimodal Planning with Multi-target Hydra-Distillation [[Paper](https://arxiv.org/pdf/2406.06978)] [[Code](https://github.com/NVlabs/Hydra-MDP)]
   
    Summary: The student model learns diverse trajectory candidates tailored for various evaluation metrics through the knowledge distillation of human teachers and rule-based teachers.

- **DualAD**: Disentangling the Dynamic and Static World for End-to-End Driving, **CVPR 2025** [[Paper](https://arxiv.org/pdf/2406.06264)] [[Code](https://github.com/TUM-AVS/DualAD)]
   
    Summary: A dual-stream architecture that decouples dynamic agents and static scene elements is used to compensate for the movement of self-vehicles and objects, enabling the system to better integrate information in the time dimension.

- **SparseDrive**: End-to-End Autonomous Driving via Sparse Scene Representation [[Paper](https://arxiv.org/pdf/2405.19620)] [[Code](https://github.com/swc-17/SparseDrive)]
   
    Summary: Sparse scene representation is used, while adding a parallel motion planner and using a hierarchical programming selection strategy to improve the performance of the model.

- **GAD**: GAD-Generative Learning for HD Map-Free Autonomous Driving [[Paper](https://arxiv.org/pdf/2405.00515)] [[Code](https://github.com/mr-d-self-driving/GAD)]
   
    Summary: The data-driven predictive - planning framework without high-precision maps goes beyond the planning methods of simple imitation or trajectory sampling.

- **SparseAD**: Sparse Query-Centric Paradigm for Efficient End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2404.06892)]
   
    Summary: Using the sparse query center paradigm to reduce computing costs and memory usage enables the utilization of longer historical information.

- **GenAD**: Generative End-to-End Autonomous Driving, **ECCV 2024** [[Paper](https://arxiv.org/pdf/2402.11502)] [[Code](https://github.com/wzzheng/GenAD)]
   
    Summary: Use a unified future trajectory generation model to perform motion prediction and planning simultaneously for introducing trajectory priors and higher-order interactions.

- **GraphAD**: Interaction Scene Graph for End-to-end Autonomous Driving [[Paper](https://arxiv.org/pdf/2403.19098)] [[Code](https://github.com/zhangyp15/GraphAD)]
   
    Summary: The graph model is adopted to describe the complex interactions in the traffic scene and introduce powerful prior knowledge.

- **ActiveAD**: Planning-Oriented Active Learning for End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2403.02877)]
   
    Summary: Utilize the active learning method oriented to planning to intelligently select the data that most needs annotation and improve the efficiency of data utilization.

- **VADv2**: End-to-End Vectorized Autonomous Driving via Probabilistic Planning [[Paper](https://arxiv.org/pdf/2402.13243)] [[Code](https://github.com/hustvl/vad)]
   
    Summary: Output the probability distribution of the action to deal with the uncertainty of the planning.
</details>

<details open>
<summary>2023</summary>

- **DriveAdapter**: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving, **ICCV 2023** [[Paper](https://arxiv.org/pdf/2308.00398)] [[Code](https://github.com/OpenDriveLab/DriveAdapter)]
   
    Summary: DriveAdapter decouples the perceptual learning of the student model from the planning knowledge of the teacher model by introducing an adapter module, avoiding the causal confusion problem in traditional behavior cloning methods and improving the efficiency and performance of the autonomous driving system.

- **VAD**: Vectorized Scene Representation for Efficient Autonomous Driving, **ICCV 2023** [[Paper](https://arxiv.org/pdf/2303.12077)] [[Code](https://github.com/hustvl/VAD)]
   
    Summary: The design of vectorized environmental representation improves the processing speed, and instance-level planning constraints enhance planning security.

- **ThinkTwice**: Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving, **CVPR 2023** [[Paper](https://arxiv.org/pdf/2305.06242)] [[Code](https://github.com/OpenDriveLab/ThinkTwice)]
   
    Summary: This paper refines the prediction results layer by layer through an extensible decoder layer, combining spatio-temporal prior knowledge and intensive supervision, which enhances driving safety and task completion rate, and also provides new ideas for planner design.

- **ReasonNet**: End-to-End Driving with Temporal and Global Reasoning, **CVPR 2023** [[Paper](https://arxiv.org/pdf/2305.10507)] [[Code](https://github.com/opendilab/DOS?tab=readme-ov-file)]
   
    Summary: Use temporal reasoning module to effectively fuse information from different frames and a transformer-based global reasoning module for better scene understanding. Release a dataset DOS, which consists of diverse occlusion scenarios in urban driving for systematic evaluation of occlusion events.

- **SuperDriverAI**: Towards Design and Implementation for End-to-End Learning-based Autonomous Driving [[Paper](https://arxiv.org/pdf/2305.10443)]
   
    Summary: Employing simple DNN network to predict steering angle, and the visual attention module improves interpretability.

- **UniAD**: Planning-oriented Autonomous Driving, **CVPR 2023** [[Paper](https://arxiv.org/pdf/2212.10156)] [[Code](https://arxiv.org/pdf/2212.10156)]
   
    Summary: The full-stack driving task is integrated in a query-based network. Different modules achieve feature complementarity and a global perspective, oriented towards the final planning task.

- **End-to-End Learning of Behavioural Inputs for Autonomous Driving in Dense Traffic**: End-to-End Learning of Behavioural Inputs for Autonomous Driving in Dense Traffic, **IROS 2023** [[Paper](https://arxiv.org/pdf/2310.14766)]
   
    Summary: By embedding a novel differentiable trajectory optimizer as the neural network layer, this method can dynamically adjust the behavioral input while ensuring the rapid convergence of the optimizer, thereby improving driving efficiency and reducing the collision rate.

- **CRCHFL**: Communication Resources Constrained Hierarchical Federated Learning for End-to-End Autonomous Driving, **IROS 2023** [[Paper](https://arxiv.org/pdf/2306.16169)]
   
    Summary: This paper proposes an optimization-driven communication resource-constrained hierarchical federated learning framework (CRCHFL), aiming to address the trade-off between limited communication resources and learning performance in end-to-end autonomous driving scenarios.

- **PPGeo**: Policy pre-training for autonomous driving via self-supervised geometric modeling, **ICLR 2023** [[Paper](https://arxiv.org/pdf/2301.01006)] [[Code](https://github.com/OpenDriveLab/PPGeo)]
   
    Summary: By performing self-supervised geometric modeling (pose/depth prediction and future ego-motion prediction) in stages on a large number of unlabeled YouTube driving videos, PPGeo pre-trained an encoder that can extract rich visual representations relevant to driving policies, thereby significantly improving the performance of visuo-motor driving tasks in data-constrained situations.

</details>

<details open>
<summary>Before 2023</summary>

- **MMFN**: Multi-Modal-Fusion-Net for End-to-End Driving, **IROS 2022** [[Paper](https://arxiv.org/pdf/2207.00186)] [[Code](https://github.com/Kin-Zhang/mmfn)]
   
    Summary: This paper improves the driving performance of the autonomous driving model in complex urban environments by integrating camera, LiDAR, high-definition Map (HD Map) and radar data.

- **KEMP**: Keyframe-Based Hierarchical End-to-End Deep Model for Long-Term Trajectory Prediction, **ICRA 2022** [[Paper](https://arxiv.org/pdf/2205.04624)]
   
    Summary: This paper predicts keyframes based on road context, and then fills in the intermediate states according to the keyframes and road context to generate complete trajectories, achieving the most advanced prediction performance.

- **TCP**: Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline, **NeurIPS 2022** [[Paper](https://arxiv.org/pdf/2206.08129)] [[Code](https://github.com/OpenPerceptionX/TCP)]
   
    Summary: This paper combines two mainstream prediction paradigms: trajectory planning and direct control. By designing a unified learning framework and interaction mechanism, it fully leverages the advantages of both and optimizes the final output through context fusion strategies.

- **ST-P3**: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning, **ECCV 2022** [[Paper](https://arxiv.org/pdf/2207.07601)] [[Code](https://github.com/OpenDriveLab/ST-P3)]
   
    Summary: ST-P3 proposes an interpretable end-to-end visual autonomous driving system, which realizes the joint spatiotemporal feature learning of perception, prediction and planning tasks by integrating self-centered alignment accumulation, dual-path prediction and time refinement units.

- **MP3**: A Unified Model to Map, Perceive, Predict and Plan, **CVPR 2021** [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.pdf)]
   
    Summary: By online predicting the map and dynamic agent status and using this information for motion planning, it achieves driving performance that is safer, more comfortable and has higher command compliance than existing methods without the need for high-precision maps.

- **Multitask-with-attention**: Multi-task Learning with Attention for End-to-end Autonomous Driving, **CVPR 2021** [[Paper](https://arxiv.org/pdf/2104.10753)] [[Code](https://github.com/KeishiIshihara/multitask-with-attention)]
   
    Summary: By integrating the Conditional Imitation Learning (CIL) framework and multi-task learning, the authors designed an attention-aware network to enhance the model's generalization ability and processing capacity for traffic signals.

- **Transfuser**: Multi-Modal Fusion Transformer for End-to-End Autonomous Driving, **CVPR 2021** [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.pdf)] [[Code](https://github.com/autonomousvision/transfuser)]
   
    Summary: TransFuser proposed a multimodal fusion Transformer based on the attention mechanism to integrate image and LiDAR data, thereby achieving a lower collision rate and better driving performance.

- **NEAT**: Neural Attention Fields for End-to-End Autonomous Driving, **ICCV 2021** [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chitta_NEAT_Neural_Attention_Fields_for_End-to-End_Autonomous_Driving_ICCV_2021_paper.pdf)] [[Code](https://github.com/autonomousvision/neat)]
   
    Summary: By learning the attention map from camera input to the spatio-temporal query location of BEV, the correlation problem of existing methods in BEV semantic prediction from camera input has been overcome, and robust and interpretable cloning of autonomous driving behavior has been achieved.

- **Fast-LiDARNet**: Efficient and Robust LiDAR-Based End-to-End Navigation, **ICRA 2021** [[Paper](https://arxiv.org/pdf/2105.09932)]
   
    Summary: This paper addresses the issues of high computational cost and insufficient model robustness in processing LiDAR data in existing methods, and enhances efficiency and reliability by optimizing the neural network structure and fusion algorithm.

- **IVMP**: Learning Interpretable End-to-End Vision-Based Motion Planning for Autonomous Driving with Optical Flow Distillation, **ICRA 2021** [[Paper](https://arxiv.org/pdf/2104.12861)] [[Project](https://sites.google.com/view/ivmp)]
   
    Summary: This paper achieves interpretable trajectory planning by predicting future bird's-eye view semantic maps, and simultaneously adopts optical flow distillation technology to enhance network performance and maintain real-time performance.

- **P3**: Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations, **ECCV 2020** [[Paper](https://arxiv.org/pdf/2008.05930)]
   
    Summary: A novel end-to-end autonomous driving system based on the differentiable probabilistic semantic occupancy layer is proposed, aiming to solve the problems of information loss in traditional perception modules and inconsistency among modules in multi-task learning.

- **DARB**: Exploring data aggregation in policy learning for vision-based urban autonomous driving, **CVPR 2020** [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf)] [[Code](https://github.com/autonomousvision/data_aggregation)]
   
    Summary: A novel data aggregation method is proposed. By sampling key states and using replay buffers that focus on uncertainties, the generalization ability and robustness of visual driving strategies in complex traffic scenarios are significantly improved.

- **Roach**: End-to-End Urban Driving by Imitating a Reinforcement Learning Coach, **ICCV 2021** [[Paper](https://arxiv.org/pdf/2108.08265)] [[Code](https://github.com/zhejz/carla-roach)]
   
    Summary: This paper proposes an end-to-end imitation learning framework based on the guidance of reinforcement learning experts, which effectively improves the performance of autonomous driving agents through intensive supervision signals and multi-task learning objectives.

- **LBC**: Learning by cheating, **CoRL 2019** [[Paper](https://arxiv.org/pdf/1912.12294)] [[Code](https://github.com/dotchen/LearningByCheating)]
   
    Summary: By using "privileged" agents as teachers to train pure visual perception autonomous driving systems, the driving performance in complex urban environments has been significantly improved.

- **CIL**: End-to-End driving via conditional imitation learning, **CoRL 2018** [[Paper](https://arxiv.org/pdf/1710.02410)] [[Code](https://github.com/carla-simulator/imitation-learning)]
   
    Summary: The conditional imitation learning method based on advanced commands enables the deep learning system that imitates human driving to respond to specific navigation instructions during the test stage while processing sensor inputs.

- **Drive in A Day**: Learning to drive in a day [[Paper](https://arxiv.org/pdf/1807.00412)] [[Code](https://github.com/sheelabhadra/Learning2Drive)]
   
    Summary: Deep reinforcement learning is applied to autonomous driving. Through autonomous exploration of vehicles and universal driving distance rewards, lane following can be learned merely based on monocular images.

- **CNN E2E**: End to End Learning for Self-Driving Cars [[Paper](https://arxiv.org/pdf/1604.07316)] [[Code](https://github.com/Nuclearstar/End-to-End-Learning-for-Self-Driving-Cars)]
   
    Summary: By training an end-to-end CNN to directly convert the original pixels of the camera into steering instructions, autonomous driving in various complex road conditions was achieved, and its potential advantages in performance and system scale compared with the traditional step-by-step method were proved.

- **Alvinn**: An autonomous land vehicle in a neural network, **NeurIPS 1988** [[Paper](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)]
   
    Summary: ALVINN is a three-layer backpropagation neural network. Through the input of cameras and laser rangefinders, it learns and outputs the direction in which vehicles travel along the road, demonstrating the adaptive potential to adjust processing capabilities according to different conditions.

</details>


<p align="right">(<a href="#top">back to top</a>)</p>    

### VLM-Centric End-to-End Methods
<details open>
<summary>2025</summary>

- **MCAM**​: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding [[paper](https://arxiv.org/pdf/2507.06072)] [[code](https://github.com/SixCorePeach/MCAM)]

     Summary: MCAM provides a new solution for behavior understanding and causal reasoning in autonomous driving videos by integrating multimodal feature extraction, causal analysis, and vision-language converters.

- **AutoDrive-R²**​: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving  [[paper](https://arxiv.org/pdf/2509.01944)]


     Summary: AutoDrive-R² enhances the reasoning and self-reflection capabilities of autonomous driving systems simultaneously by incorporating self-reflective thought chain processing and physics-based reinforcement learning.

- **DriveAgent-R1**​: Advancing VLM-based Autonomous Driving with Hybrid Thinking and Active Perception [[paper](https://arxiv.org/pdf/2507.20879)]

     Summary: DriveAgent-R1 offers an innovative solution for the field of autonomous driving through a hybrid thinking and active perception mechanism, significantly enhancing the reliability and safety of decision-making in complex scenarios, while opening up new directions for future research.

- **NavigScene:​** Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving​ [[paper](https://arxiv.org/pdf/2507.05227)]

     Summary: NavigScene significantly enhances the performance of autonomous driving systems by providing global navigation information, making them closer to the navigation capabilities of human drivers in complex and unknown environments.

- **ADRD**​: LLM-DRIVEN AUTONOMOUS DRIVING BASED ON RULE-BASED DECISION SYSTEMS [[Paper](https://arxiv.org/pdf/2506.14299)]

     Summary: ADRD (LLM-Driven Autonomous Driving with Rule-Based Decision Systems), a framework that leverages large language models to automatically generate rule-based driving policies, aims to achieve efficient, explainable, and robust autonomous driving decisions.

- **AutoVLA**​: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning [[Paper](https://arxiv.org/pdf/2506.13757)] [[project](https://autovla.github.io/)] [[code](https://github.com/ucla-mobility/AutoVLA)]

     Summary: Reinforcement learning-based post-training methods and adaptive fast and slow thinking capabilities significantly improve planning performance

- **Poutine**​: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving [[Paper](https://arxiv.org/abs/2506.11234)]

     Summary: Poutine shows that both VLT pre-training and RL fine-tuning are critical to achieving strong driving performance in the long tail. This is a 3B parameter visual language model (VLM) designed for end-to-end autonomous driving in long-tail driving scenarios, trained with self-supervised visual language track (VLT) next tag prediction Poutine-Base

- **ReCogDrive**​: A Reinforced Cognitive Framework for End-to-End Autonomous Driving [[Paper](https://arxiv.org/abs/2506.08052)] [[project](https://xiaomi-research.github.io/recogdrive/)] [[code](https://github.com/xiaomi-research/recogdrive)]

     Summary: ReCogDrive, an autonomous driving system that combines the VLM with a diffusion planner

- **AD-EE**​: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving [[Paper](https://arxiv.org/pdf/2506.05404)]

     Summary: AD-EE is an early exit framework that incorporates domain characteristics of autonomous driving and uses causal reasoning to identify the optimal exit layer.

- **FastDrive: ​**Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving [[Paper](https://arxiv.org/pdf/2506.05442)]

     Summary: Introducing NuScenes-S, a structured and concise benchmark dataset, FastDrive, a compact VLM baseline with 90 million parameters that can understand structured and concise descriptions and efficiently generate machine-friendly driving decisions

- **HMVLM**​: Multistage Reasoning-Enhanced Vision-Language Model for Long-Tailed Driving Scenarios​ [[Paper](https://arxiv.org/pdf/2506.05883)]

     Summary: HaoMo implements the slow branch of cognitively inspired fast-slow architecture. The fast controller outputs low-level steering, throttle, and brake commands, while the slow planner (a large visual-language model) generates high-level intent.

- **S4-Driver**​: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation, ​**CVPR2025**​ [[Paper](https://openaccess.thecvf.com//content/CVPR2025/papers/Xie_S4-Driver_Scalable_Self-Supervised_Driving_Multimodal_Large_Language_Model_with_Spatio-Temporal_CVPR_2025_paper.pdf)]

     Summary: S4-Driver is a scalable self-supervised motion planning algorithm with spatiotemporal visual representations

- **DiffVLA**​: Vision-Language Guided Diffusion Planning for Autonomous Driving [[Paper](https://arxiv.org/pdf/2505.19381)]

     Summary: Diff-VLA introduces a novel hybrid sparse-dense diffusion policy, enhanced by the integration of a Vision-Language Model (VLM)

- **X-Driver**​:Explainable Autonomous Driving with Vision-Language Models [[Paper](https://arxiv.org/pdf/2505.05098)]

     Summary: X-Driver is a unified multimodal large language model (MLLM) framework for closed-loop autonomous driving that utilizes Chain of Thought (CoT) reasoning and autoregressive modeling to improve both perception and decision-making performance.

- **DriveGPT4-V2**​: Harnessing Large Language Model Capabilities for Enhanced Closed-Loop Autonomous Driving, ​**CVPR2025**​ [[Paper](https://openaccess.thecvf.com//content/CVPR2025/papers/Xu_DriveGPT4-V2_Harnessing_Large_Language_Model_Capabilities_for_Enhanced_Closed-Loop_Autonomous_CVPR_2025_paper.pdf)]

     Summary: Different from the previous study DriveGPT4-V1, which focused on open-loop tasks, this study explores the ability of LLM in enhancing closed-loop autonomous driving and uses an expert LLM as a teacher for online policy supervision.

- **DriveMind**​: A Dual-VLM based Reinforcement Learning Framework for Autonomous Driving [[Paper](https://arxiv.org/abs/2506.00819)]

     Summary: A dynamic dual-VLM architecture is proposed, which combines a static contrastive VLM encoder with a novelty-triggered VLM encoder-decoder to solve the semantic rigidity problem of traditional fixed cues.

- **ReasonPlan**​: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving [[Paper](https://arxiv.org/abs/2505.20024)] [[Code](https://github.com/Liuxueyi/ReasonPlan)]

     Summary: ReasonPlan is an MLLM fine-tuning framework specifically designed for closed-loop driving, enabling comprehensive reasoning through a self-supervised Next Scene Prediction task and a supervised Decision Chain-of-Thought process.

- **FutureSightDrive**​: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving [[Paper](https://arxiv.org/abs/2505.17685)] [[Code](https://github.com/missTL/FSDrive)]

     Summary: FutureSightDrive proposes a spatio-temporal CoT reasoning method to enable the model to think visually.

- **PADriver**​: Towards Personalized Autonomous Driving [[Paper](https://arxiv.org/abs/2505.05240)]

     Summary: Based on a multimodal large language model (MLLM), PADriver takes streaming video frames and personalized text prompts as input to actively perform scene understanding, danger level assessment, and action decision-making.

- **LDM**​: Unlock the Power of Unlabeled Data in Language Driving Model, ​**ICRA2025**​ [[Paper](https://arxiv.org/abs/2503.10586)]

     Summary: Dynamic self-supervised pre-training framework, semi-supervised knowledge distillation architecture

- **DriveMoE**​: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving [[Paper](https://arxiv.org/abs/2505.16278)] [[Project](https://thinklab-sjtu.github.io/DriveMoE/)] [[Code](https://github.com/Thinklab-SJTU/DriveMoE)]

     Summary: A new VLM-AD framework based on MoE

- **DriveMonkey**​:Extending Large Vision-Language Model for Diverse Interactive Tasks in Autonomous Driving [[Paper](https://arxiv.org/abs/2505.08725)] [[Code](https://github.com/zc-zhao/DriveMonkey)]

     Summary: Use a series of learnable queries to seamlessly integrate the LVLM with the spatial processor, which is designed as a plug-and-play component and can be initialized with a pre-trained 3D detector to improve 3D perception

- **AgentThink**​: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving [[Paper](https://arxiv.org/abs/2505.15298)]

     Summary: AgentThink combines Chain-of-Thought (CoT) reasoning with dynamic agent-style tool invocation for autonomous driving tasks for the first time.

- **DSDrive**​: Distilling Large Language Model for Lightweight End-to-End Autonomous Driving with Unified Reasoning and Planning [[Paper](https://arxiv.org/pdf/2505.05360)]

     Summary: DSDrive uses a distillation method to enhance lightweight LLM as the core of the AD system

- **LightEMMA**​: Lightweight End-to-end Multimodal Autonomous Driving [[Paper](https://arxiv.org/abs/2505.00284)] [[Code](https://github.com/michigan-traffic-lab/LightEMMA)]

     Summary: LightEMMA is a lightweight, end-to-end multimodal model designed for autonomous driving, enabling efficient and comprehensive perception and decision-making.

- **Towards Human-Centric Autonomous Driving**​: A Fast-Slow Architecture Integrating Large Language Model Guidance with Reinforcement Learning [[Paper](https://arxiv.org/abs/2505.06875)]

     Summary: A "fast and slow" decision-making framework that combines a large language model (LLM) for high-level instruction parsing and a reinforcement learning (RL) agent for low-level real-time decision-making.

- **DriveSOTIF**​: Advancing Perception SOTIF Through Multimodal Large Language Models [[Paper](https://arxiv.org/abs/2505.07084)]

     Summary: The first innovative fusion of multimodal large language models (MLLMs) and SOTIF risk recognition

- **Actor-Reasoner**​: Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions [[Paper](https://arxiv.org/abs/2503.00502)] [[Code](https://github.com/FanGShiYuu/Actor-Reasoner)]

- **MPDrive**​: Improving Spatial Understanding with Marker-Based Prompt Learning for Autonomous Driving, ​**CVPR2025**​ [[Paper](https://arxiv.org/abs/2504.00379)]

     Summary: By using detection experts to overlay numerical labels on target regions to create labeled images, we transform complex text coordinate generation into text-based visual label prediction.

- **V3LMA**​: Visual 3D-enhanced Language Model for Autonomous Driving [[Paper](https://arxiv.org/pdf/2505.00156)]

     Summary: Approach improves 3D scene understanding by combining Large Language Models (LLMs) with vision-language models (LVLMs).

- **OpenDriveVLA**​: Towards End-to-end Autonomous Driving with Large Vision Language Action Model [[Paper](https://arxiv.org/abs/2503.23463v1)] [[Project](https://drivevla.github.io/)] [[Code](https://github.com/DriveVLA/OpenDriveVLA)]

     Summary: OpenDriveVLA is built on an open-source pre-trained large-scale vision-language model (VLM) to generate reliable driving actions conditioned on 3D environment perception, ego vehicle state, and driver commands.

- **SimLingo**​: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment, ​**CVPR2025**​ [[Paper](https://openaccess.thecvf.com//content/CVPR2025/papers/Renz_SimLingo_Vision-Only_Closed-Loop_Autonomous_Driving_with_Language-Action_Alignment_CVPR_2025_paper.pdf)] [[Project](https://www.katrinrenz.de/simlingo/)] [[Code](https://github.com/RenzKa/simlingo)]

     Summary: SimLingo is a vision-language-action model unifying the tasks of autonomous driving, vision-language understanding and language-action alignment.

- **SAFEAUTO**​: KNOWLEDGE-ENHANCED SAFE AUTONOMOUS DRIVING WITH MULTIMODAL FOUNDATION MODELS , ​**ICLR2025**​ [[Paper](https://arxiv.org/abs/2503.00211)] [[Code](https://github.com/AI-secure/SafeAuto)]

     Summary: SAFEAUTO introduces the Place-Dependent Cross-Entropy (PDCE) loss function, specifically designed to improve the accuracy of low-level control signal predictions by treating numerical values as textual sequences.

- **NuGrounding**​: A Multi-View 3D Visual Grounding Framework in Autonomous Driving [[Paper](https://arxiv.org/abs/2503.22436)]

     Summary: NuGrounding introduces a novel paradigm that seamlessly integrates the instruction comprehension capabilities of multimodal large language models (MLLMs) with the precise localization abilities of specialized detection models.

- **CoT-Drive**​: Efficient Motion Forecasting for Autonomous Driving with LLMs and Chain-of-Thought Prompting [[Paper](https://arxiv.org/pdf/2503.07234)]

     Summary: Use LLMs and chaining cues to do prediction tasks

- **CoLMDriver**​: LLM-based Negotiation Benefits Cooperative Autonomous Driving [[Paper](https://arxiv.org/abs/2503.08683)] [[Code](https://github.com/cxliu0314/CoLMDriver)]

     Summary: The first full-process collaborative driving system based on a large language model, capable of effective language-based negotiation and real-time driving control.

- **AlphaDrive**​: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning [[Paper](https://arxiv.org/abs/2503.07608)] [[Code](https://github.com/hustvl/AlphaDrive)]

     Summary: AlphaDrive is the first framework to integrate GRPO-based RL and planning reasoning into autonomous driving

- **TrackingMeetsLMM**​: Tracking Meets Large Multimodal Models for Driving Scenario Understanding [[Paper](https://arxiv.org/abs/2503.14498)] [[Code](https://github.com/mbzuai-oryx/TrackingMeetsLMM)]

     Summary: Introduced a novel method to embed tracking information into LMMs to enhance their spatiotemporal understanding of driving scenarios

- **BEVDriver**​: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving [[Paper](https://arxiv.org/abs/2503.03074)]

     Summary: Directly utilize the original BEV features generated by LiDAR and camera to eliminate the dependence on pre-predicted path points. Use two PIDs to control the lateral and longitudinal directions to bridge the gap between high-level decision-making and low-level planning.

- **DynRsl-VLM**​: Enhancing Autonomous Driving Perception with Dynamic Resolution Vision-Language Models [[Paper](https://arxiv.org/pdf/2503.11265)]

     Summary:  DynRsl-VLM incorporates a dynamic resolution image input processing approach that captures all entity feature information within an image while ensuring that the image input remains computationally tractable for the Vision Transformer (ViT).

- **Sce2DriveX**​: A Generalized MLLM Framework for Scene-to-Drive Learning [[Paper](https://arxiv.org/abs/2502.14917)]

     Summary: Sce2DriveX utilizes multimodal joint learning from local scene videos and global BEV maps to deeply understand long-range spatiotemporal relationships and road topology, enhancing its comprehensive perception and reasoning capabilities in 3D dynamic/static scenes and achieving driving generalization across scenes.

- VLM-Assisted Continual learning for Visual Question Answering in Self-Driving  [[Paper](https://arxiv.org/pdf/2502.00843)]

     Summary: It introduces a novel continual learning framework that integrates vision-language models (VLMs) with selective memory replay and knowledge distillation, further strengthened by regularization of task-specific projection layers.

- **LeapVAD**​: A Leap in Autonomous Driving via Cognitive Perception and Dual-Process Thinking [[Paper](https://arxiv.org/pdf/2501.08168)] [[Project](https://pjlab-adg.github.io/LeapVAD/)] [[Code](https://github.com/PJLab-ADG/LeapVAD)]

     Summary: LeapAD is a dual-process, closed-loop autonomous driving system that enables continuous learning, adaptation, and improvement over time.
</details>

<details open>
<summary>2024</summary>

- **VLM-RL**​: A Unified Vision Language Model and Reinforcement Learning Framework for Safe Autonomous Driving[[Paper](https://arxiv.org/abs/2412.15544)][[Project](https://www.huang-zilin.com/VLM-RL-website/)][[Code](https://github.com/zihaosheng/VLM-RL)]

    Symmary: VLM-RL is the first work in the autonomous driving field to unify VLMs with RL for end-to-end driving policy learning in the CARLA simulator.

- **GPVL**: Generative Planning with 3D-vision Language Pre-training for End-to-End Autonomous Driving, **AAAI 2025** [[Paper](https://arxiv.org/pdf/2501.08861)] [[Code](https://github.com/ltp1995/GPVL)]
   Summary: A generative planning framework for autonomous driving using a 3D vision-language pre-training paradigm.
- **CALMM-Drive**: Confidence-Aware Autonomous Driving with Large Multimodal Model [[Paper](https://arxiv.org/abs/2412.04209)]
   
   Summary: The CALMM-Drive approach integrates driving task-specific Chain-of-Thought (CoT) reasoning with Top-K confidence elicitation to improve the accuracy and reliability of decision-making.

- **WiseAD**: Knowledge Augmented End-to-End Autonomous Driving with Vision-Language Model [[Paper](https://arxiv.org/abs/2412.09951)] [[Code](https://github.com/wyddmw/WiseAD)]
   
   Summary: WiseAD is a specialized vision-language model (VLM) designed for end-to-end autonomous driving, capable of performing driving reasoning, action justification, object recognition, risk analysis, providing driving suggestions, and trajectory planning across a wide range of scenarios.

- **OpenEMMA**: Open-Source Multimodal Model for End-to-End Autonomous Driving, **WACV 2025** [[Paper](https://arxiv.org/abs/2412.15208)] [[Code](https://github.com/taco-group/OpenEMMA)]
  
   Summary: OpenEMMA leverages existing open source modules and pre-trained MLLMs to replicate the capabilities of EMMA in trajectory planning and perception.

- **FeD**​: Feedback-Guided Autonomous Driving, ​**CVPR2024**​[[Paper](https://fedaltothemetal.github.io/resources/FeD_v1.pdf)][[Project](https://fedaltothemetal.github.io/)]

    Summary: Achieving the First Perceptual-Motion End-to-End Training and Evaluation of an LLM-Based Driving Model

- **LeapAD**​: Continuously learning, adapting, and improving: A dual-process approach to autonomous driving, ​**NeurIPS 2024**​[[Paper](https://arxiv.org/abs/2405.15324)][[Project](https://pjlab-adg.github.io/LeapAD/)][[Code](https://github.com/PJLab-ADG/LeapAD)]

- **DriveMM**​: All-in-One Large Multimodal Model for Autonomous Driving[[Paper](https://arxiv.org/abs/2412.07689)][[Project](https://zhijian11.github.io/DriveMM/)][[Code](https://github.com/zhijian11/DriveMM)]

    Summary: DriveMM is robustly designed with the general capability to perform a wide variety of autonomous driving (AD) tasks and demonstrates strong generalization performance, enabling effective transfer to new datasets.

- Explanation for Trajectory Planning using Multi-modal Large Language Model for Autonomous Driving​**, ECCV2024**​[[Paper](https://arxiv.org/abs/2411.09971)]

    Summary:  Leveraging the newly collected dataset, we take the future planning trajectory of the ego vehicle as input.

- **LaVida Drive**: Vision-Text Interaction VLM for Autonomous Driving with Token Selection, Recovery and Enhancement [[Paper](https://arxiv.org/abs/2411.12980)]
   
   Summary: An innovative VQA framework designed to support fine-grained perception of high-resolution visual inputs in dynamic driving environments while integrating temporal information.

- **EMMA**: End-to-End Multimodal Model for Autonomous Driving [[Paper](https://arxiv.org/abs/2410.23262)]
   
   Summary: EMMA directly maps raw camera sensor data into various driving-specific outputs, including planner trajectories, perception objects, and road graph elements.

- **DriVLMe**: Enhancing LLM-based Autonomous Driving Agents with Embodied and Social Experiences, **IROS 2024** [[Paper](https://arxiv.org/pdf/2406.03008)] [[Project](https://sled-group.github.io/driVLMe/)] [[Code](https://github.com/sled-group/driVLMe/tree/main)]
   
   Summary: DriVLMe is a video-language model-based agent designed to enable natural and effective communication between humans and autonomous vehicles, allowing the vehicles to perceive their surroundings and navigate the environment more intuitively.

- **OccLLaMA**: An Occupancy-Language-Action Generative World Model for Autonomous Driving [[Paper](https://arxiv.org/abs/2409.03272)]
   
   Summary: OccLLaMA is a unified 3D occupancy-language-action generative world model that integrates various VLA (vision-language-action) related tasks.

- ​**MiniDrive**​: More Efficient Vision-Language Models with Multi-Level 2D Features as Text Tokens for Autonomous Driving[[Paper](https://arxiv.org/abs/2409.07267)]


    Summary: By combining the Feature Engineering Mixture of Experts (FEMoE) module with a dynamic instruction adapter, our approach addresses the limitation of previous methods, which could only generate static visual token embeddings for a given image.

- ​**RDA-Driver**​:Making Large Language Models Better Planners with Reasoning-Decision Alignment, ​**2024ECCV**​[[Paper](https://arxiv.org/abs/2408.13890)]


    Summary: We develop an end-to-end decision model based on a multimodal enhanced LLM that simultaneously performs CoT reasoning and enforces planning outcomes.

- ​**EC-Drive**​:Edge-Cloud Collaborative Motion Planning for Autonomous Driving with Large Language Models, ​**2024ICCT**​[[Paper](https://arxiv.org/abs/2408.09972)][[Project](https://sites.google.com/view/ec-drive)]


    Summary: EC-Drive utilizes drift detection algorithms to selectively upload critical data, including new obstacles and traffic pattern changes, to the cloud for processing by GPT-4, while routine data is efficiently managed by smaller LLMs on edge devices.
- **V2X-VLM**​: End-to-End V2X Cooperative Autonomous Driving Through Large Vision-Language Models[[Paper](https://arxiv.org/abs/2408.09251)][[Project](https://www.huang-zilin.com/V2X-VLM-website/)][[Code](https://github.com/zilin-huang/V2X-VLM)]


    Summary: This study aims to propose pioneering E2E vehicle-infrastructure cooperative autonomous driving (VICAD) framework leveraging large VLMs to enhance collaborative situational awareness, decision-making, and overall driving performances.
​- **Cube-LLM**​: Language-Image Models with 3D Understanding[[Paper](https://arxiv.org/abs/2405.03685)][[Project](https://janghyuncho.github.io/Cube-LLM/)]


    Summary: Cube-LLM, a pre-trained visual language model for autonomous driving, can infer 3D indoor and outdoor scenes from a single image
​- **VLM-MPC**​: Vision Language Foundation Model (VLM)-Guided Model Predictive Controller (MPC) for Autonomous Driving[[Paper](https://arxiv.org/abs/2408.04821)]

    Summary: VLM-MPC combines the Model Predictive Controller (MPC) with VLM to evaluate how model-based control could enhance VLM decision-making.

- **SimpleLLM4AD**​: An End-to-End Vision-Language Model with Graph Visual Question Answering for Autonomous Driving, ​**IEIT Systems**​[[Paper](https://arxiv.org/abs/2407.21293)]

    Summary: SimpleLLM4AD reimagines the traditional autonomous driving pipeline by structuring the task into four interconnected stages: perception, prediction, planning, and behavior.

- **AsyncDriver**​: Asynchronous Large Language Model Enhanced Planner for Autonomous Driving, ​**ECCV 2024**​[[Paper](https://arxiv.org/abs/2406.14556)][[Code](https://github.com/memberRE/AsyncDriver)]

    Summary: AsyncDriver is a novel asynchronous, LLM-enhanced closed-loop framework that utilizes scene-aware instruction features generated by a large language model (LLM) to guide real-time planners in producing accurate and controllable trajectory predictions.

- ​**AD-H**​: AUTONOMOUS DRIVING WITH HIERARCHICAL AGENTS , ​**ICLR2025**​[[Paper](https://openreview.net/pdf/e15ef4c8e8f4e0d2db875b42314bcc25546c73dc.pdf)]

    Summary: A hierarchical framework that facilitates collaboration between two agents: the MLLM-based planner and the controller.

- **CarLLaVA**​: Vision language models for camera-only closed-loop driving[[Paper](https://arxiv.org/abs/2406.10165)][[Project](https://www.youtube.com/watch?v=E1nsEgcHRuc)]


    Summary: CarLLaVA uses a semi-disentangled output representation of both path predictions and waypoints, getting the advantages of the path for better lateral control and the waypoints for better longitudinal control.

- ​**PlanAgent**​: A Multi-modal Large Language Agent for Closed-loop Vehicle Motion Planning[[Paper](https://arxiv.org/abs/2406.01587)]


    Summary: PlanAgent is the first closed-loop mid-to-mid(use bev, no raw sensor) autonomous driving planning agent system based on a Multi-modal Large Language Model.

- ​**Atlas ​**​: Is a 3D-Tokenized LLM the Key to Reliable Autonomous Driving?[[Paper](https://arxiv.org/abs/2405.18361#)]


    Summary: A DETR-style 3D perceptron is introduced as a 3D tokenizer, which connects LLM with a single-layer linear projector.

- ​**Driving with Regulation**​: Interpretable Decision-Making for Autonomous Vehicles with Retrieval-Augmented Reasoning via LLM[[Paper](https://arxiv.org/abs/2410.04759)]


    Summary:Traffic Regulation Retrieval (TRR) agent based on Retrieval Augmented Generation (RAG) to automatically retrieve relevant traffic rules and guidelines from a wide range of regulatory documents and related records based on the context of the autonomous vehicle

- ​**OmniDrive**​: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning, ​**CVPR 2025**​[[Paper](https://arxiv.org/abs/2405.01533)][[Code](https://github.com/NVlabs/OmniDrive)]


    Summary: The  features a novel 3D multimodal LLM design that uses sparse queries to lift and compress visual representations into 3D.

- ​**Co-driver**​: VLM-based Autonomous Driving Assistant with Human-like Behavior and Understanding for Complex Road Scenes[[Paper](https://arxiv.org/html/2405.05885v1)]


    Summary: This is an automated driving assistance system that provides adjustable driving behavior for autonomous vehicles based on an understanding of complex road scenarios, including safety distances, weather, lighting conditions, road surfaces, and locations.

- ​**AgentsCoDriver**​: Large Language Model Empowered Collaborative Driving with Lifelong Learning[[Paper](https://arxiv.org/abs/2404.06345)]


    Summary: Multiple vehicles are capable of collaborative driving It can accumulate knowledge, lessons, and experiences over time by constantly interacting with its environment, enabling lifelong learning

- ​**EM-VLM4AD**​: Multi-Frame, Lightweight & Efficient Vision-Language Models for Question Answering in Autonomous Driving, ​**CVPR2024**​[[Paper](https://arxiv.org/abs/2403.19838)][[Code](https://github.com/akshaygopalkr/EM-VLM4AD)]


    Summary: EM-VLM4AD is an efficient and lightweight multi-frame vision-language model designed to perform visual question answering for autonomous driving applications.

- ​**LeGo-Drive**​: Language-enhanced Goal-oriented Closed-Loop End-to-End Autonomous Driving, ​**IROS2024**​[[Paper](https://arxiv.org/abs/2403.20116)][[Project](https://reachpranjal.com/lego-drive/)][[Code](https://github.com/reachpranjal/lego-drive)]


    Summary: A novel planning-guided end-to-end LLM-based goal point navigation solution that predicts and improves the desired state by dynamically interacting with the environment and generating a collision-free trajectory.

- ​**Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving**​, ​**ICCMA2024**​[[Paper](https://arxiv.org/abs/2402.13602v3)]


    Summary: Regarding the "location of the object," "speed of our car," "distance to the object," and "our car’s direction" are fed into the large language model for mathematical calculations within CARLA. After formulating these calculations based on overcoming weather conditions, precise control values for brake and speed are generated.

- ​**VLAAD**​: Vision and Language Assistant for Autonomous Driving, ​**WACV2024**​[[Paper](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Park_VLAAD_Vision_and_Language_Assistant_for_Autonomous_Driving_WACVW_2024_paper.pdf)]


    Summary: Aiming to enhance the explainability of autonomous driving systems.

- ​**ELM**​: Embodied Understanding of Driving Scenarios, ​**ECCV2024**​[[Paper](https://arxiv.org/abs/2403.04593)]


    Summary: we introduce the Embodied Language Model (ELM), a comprehensive framework tailored for agents' understanding of driving scenes with large spatial and temporal spans.

- ​**RAG-Driver**​: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model,​**​ RSS2024**​[[Paper](https://arxiv.org/abs/2402.10828)][[Project](https://yuanjianhao508.github.io/RAG-Driver/)][[Code](https://github.com/YuanJianhao508/RAG-Driver)]


    Summary: RAG-Driver is a novel retrieval-augmented, multimodal large language model that utilizes in-context learning to enable high-performance, interpretable, and generalizable autonomous driving.

- ​**BEV-TSR**​: Text-Scene Retrieval in BEV Space for Autonomous Driving,​**​ AAAI-2025**​[[Paper](https://arxiv.org/abs/2401.01065)]


    Summary: Focus on enhancing the semantic capabilities of BEV representations

- ​**LLaDA**​: Driving Everywhere with Large Language Model Policy Adaptation, ​**CVPR2024**​[[Paper](https://arxiv.org/abs/2402.05932)][[Project](https://boyiliee.github.io/llada/)][[Code](https://github.com/Boyiliee/LLaDA-AV)]


    Summary: Traffic Rule Extractor (TRE), which aims to organize and filter the inputs (initial plan+unique traffic code) and feed the output into the frozen LLMs to obtain the final new plan.

</details>

<details open>
<summary>2023</summary>

- **LingoQA**: Visual Question Answering for Autonomous Driving, **ECCV 2024** [[Paper](https://arxiv.org/abs/2312.14115)] [[Code](https://github.com/wayveai/LingoQA/)]
   
    Summary: This is a novel framework for integrating LLM into AD systems, enabling them to follow user instructions by generating code that leverages established functional primitives.
- **LaMPilot**: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs, **CVPR 2024** [[Paper](https://arxiv.org/abs/2312.04372)] [[Project](https://github.com/PurdueDigitalTwin/LaMPilot)]
   
    Summary: This is a novel framework for integrating LLM into AD systems, enabling them to follow user instructions by generating code that leverages established functional primitives.
- **LLM-ASSIST**: Enhancing Closed-Loop Planning with Language-Based Reasoning [[Paper](https://arxiv.org/pdf/2401.00125)] [[Project](https://llmassist.github.io/)]
   
    Summary: LLM-Planner takes over scenarios that PDM-Closed cannot handle.
- **DriveLM**: Driving with Graph Visual Question Answering, **ECCV 2024 Oral** [[Paper](https://arxiv.org/pdf/2312.14150)] [[Code](https://github.com/OpenDriveLab/DriveLM)]
   
    Summary: Graph VQA involves formulating Perception, Prediction, Planning reasoning as a series of questionanswer pairs (QAs) in a directed graph.
- **DriveMLM**: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving [[Paper](https://arxiv.org/abs/2312.09245)] [[Code](https://github.com/OpenGVLab/DriveMLM)]
   
    Summary: Design an MLLM planner for decision prediction, and develop a data engine that can effectively generate decision states and corresponding explanation annotation for model training and evaluation.
- **LiDAR-LLM**: Exploring the Potential of Large Language Models for 3D LiDAR Understanding [[Paper](https://arxiv.org/abs/2312.14074)] [[Project](https://sites.google.com/view/lidar-llm)]
   
    Summary: Take raw LiDAR data as input and leverage LLM’s superior inference capabilities to fully understand outdoor 3D scenes.
- **Talk2BEV**: Language-enhanced Bird's-eye View Maps for Autonomous Driving [[Paper](https://arxiv.org/abs/2310.02251)] [[Project](https://llmbev.github.io/talk2bev/)] [[Code](https://github.com/llmbev/talk2bev)]
   
    Summary: Large-scale visual language model (LVLM) combined with BEV map to achieve visual reasoning, spatial understanding and decision making.
- **Talk2Drive**: Personalized Autonomous Driving with Large Language Models: Field Experiments [[Paper](https://arxiv.org/abs/2312.09397)] [[Project](https://www.youtube.com/watch?v=4BWsfPaq1Ro)]
   
    Summary: capable of translating natural verbal commands into executable controls and learning to satisfy personal preferences for safety, efficiency, and comfort with a proposed memory module.
- **LMDrive**: Closed-Loop End-to-End Driving with Large Language Models, **CVPR 2024** [[Paper](https://arxiv.org/abs/2312.07488)] [[Code](https://github.com/opendilab/LMDrive)]
   
    Summary: LMDrive, the very first work to leverage LLMs for closed-loop end-to-end autonomous driving.
- **Reason2Drive**: Towards Interpretable and Chain-based Reasoning for Autonomous Driving, **ECCV 2024** [[Paper](https://arxiv.org/abs/2312.03661)] [[Code](https://github.com/fudan-zvg/Reason2Drive)]
   
    Summary: Introduce a straightforward yet effective framework that enhances existing VLMs with two new components: a prior tokenizer and an instructed vision decoder.
- **CAVG**: GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models [[Paper](https://arxiv.org/abs/2312.03543)] [[Code](https://github.com/Petrichor625/Talk2car_CAVG)]
   
    Summary: Utilize five encoders: Text, Image, Context, and Cross-Modal—with: with a Multimodal decoder to predict object bounding box.
- **Dolphins**: Multimodal Language Model for Driving, **ECCV 2024** [[Paper](https://arxiv.org/abs/2312.00438)] [[Project](https://vlm-driver.github.io/)] [[Code](https://github.com/SaFoLab-WISC/Dolphins)]
   
    Summary: Dolphins is adept at processing multimodal inputs comprising video (or image) data, text instructions, and historical control signals to generate informed outputs corresponding to the provided instructions.
- **Agent-Driver**: A Language Agent for Autonomous Driving, **COLM 2024** [[Paper](https://arxiv.org/abs/2311.10813)] [[Project](https://usc-gvl.github.io/Agent-Driver/)] [[Code](https://github.com/USC-GVL/Agent-Driver)]
   
    Summary: Agent-Driver changes the traditional autonomous driving pipeline by introducing a versatile tool library accessible through function calls, cognitive memory for common sense and experiential knowledge for decision-making, and a reasoning engine capable of thought chain reasoning, task planning, motion planning, and self-reflection.
- **Empowering Autonomous Driving with Large Language Models**: A Safety Perspective, **ICLR 2024** [[Paper](https://arxiv.org/abs/2312.00812)] [[Code](https://github.com/wangyixu14/llm_conditioned_mpc_ad)]
   
    Summary: Deploys the LLM as an intelligent decision-maker in planning, incorporating safety verifiers for contextual safety learning to enhance overall AD performance and safety.
- **ChatGPT as Your Vehicle Co-Pilot**: An Initial Attempt [[Paper](https://ieeexplore.ieee.org/document/10286969)]
   
    Summary: Design a universal framework that embeds LLMs as a vehicle "Co-Pilot" of driving, which can accomplish specific driving tasks with human intention satisfied based on the information provided.
- **Receive, Reason, and React**: Drive as You Say with Large Language Models in Autonomous Vehicles, **ITSM 2024** [[Paper](https://arxiv.org/abs/2310.08034)]
   
    Summary: Utilize LLMs’ linguistic and contextual understanding abilities with specialized tools to integrate the language and reasoning capabilities of LLMs into autonomous vehicles.
- **LanguageMPC**: Large Language Models as Decision Makers for Autonomous Driving [[Paper](https://arxiv.org/pdf/2310.03026)]
   
    Summary: Leverage LLMs to provide high-level decisions through chain-of-thought.Convert high-level decisions into mathematical representations to guide the bottom-level controller(MPC).
- **Driving with LLMs**: Fusing Object-Level Vector Modality for Explainable Autonomous Driving [[Paper](https://arxiv.org/abs/2310.01957)] [[Code](https://github.com/wayveai/Driving-with-LLMs)]
   
    Summary: Propose a unique object-level multimodal LLM architecture(Llama2+Lora), using only vectorized representations as input.
- **DriveGPT4**: Interpretable End-to-end Autonomous Driving via Large Language Model, **RAL** [[Paper](https://tonyxuqaq.github.io/assets/pdf/2024_RAL_DriveGPT4.pdf)] [[Project](https://tonyxuqaq.github.io/projects/DriveGPT4/)]
   
    Summary: DriveGPT4 represents the pioneering effort to leverage LLMs for the development of an interpretable end-to-end autonomous driving solution.
- **GPT-Driver**: Learning to Drive with GPT, **NeurIPS 2023** [[Paper](https://arxiv.org/abs/2310.01415)] [[Project](https://pointscoder.github.io/projects/gpt_driver/index.html)] [[Code](https://github.com/PointsCoder/GPT-Driver)]
   
    Summary: Represent the planner inputs and outputs as language tokens, and leverage the LLM to generate driving trajectories through a language description of coordinate positions.
- **DiLu**: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models, **ICLR 2024** [[Paper](https://arxiv.org/pdf/2309.16292)] [[Project](https://pjlab-adg.github.io/DiLu/)] [[Code](https://github.com/PJLab-ADG/DiLu)]
   
    Summary: Propose the DiLu framework, which combines a Reasoning and a Reflection module to enable the system to perform decision-making based on common-sense knowledge and evolve continuously.
- **Drive as You Speak**: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles [[Paper](https://arxiv.org/abs/2309.10228)]
   
    Summary: In this paper, we present a novel framework that leverages Large Language Models (LLMs) to enhance autonomous vehicles’ decision-making processes. By integrating LLMs’ natural language capabilities and contextual understanding, specialized tools usage, synergizing reasoning, and acting with various modules on autonomous vehicles, this framework aims to seamlessly integrate the advanced language and reasoning capabilities of LLMs into autonomous vehicles.
- **HiLM-D**: Enhancing MLLMs with Multi-Scale High-Resolution Details for Autonomous Driving, **IJCV** [[Paper](https://arxiv.org/abs/2309.05186)]
   
    Summary: ROLISP that aims to identify, explain and localize the risk object for the ego-vehicle meanwhile predicting its intention and giving suggestions. Propose HiLM-D (Towards High-Resolution Understanding in MLLMs for Autonomous Driving), an efficient method to incorporate HR information into MLLMs for the ROLISP task.
- **SurrealDriver**: Designing LLM-powered Generative Driver Agent Framework based on Human Drivers' Driving-thinking Data [[Paper](https://arxiv.org/abs/2309.13193)]
   
    Summary: The framework uses post-drive self-reported driving thought data from human drivers as demonstration and feedback to build a human-like generative driving agent.
- **Drive Like a Human**: Rethinking Autonomous Driving with Large Language Models [[Paper](https://arxiv.org/abs/2307.07162)] [[Code](https://github.com/PJLab-ADG/DriveLikeAHuman?tab=readme-ov-file)]
   
    Summary: Identify three key abilities: Reasoning, Interpretation and Memorization (accumulate experience and self-reflection).
- **ADAPT**: Action-aware Driving Caption Transformer, **ICRA 2023** [[Paper](https://arxiv.org/abs/2302.00673)] [[Code](https://github.com/jxbbb/ADAPT)]
   
    Summary: propose a multi-task joint training framework that aligns both the driving action captioning task and the control signal prediction task.

</details>

<p align="right">(<a href="#top">back to top</a>)</p> 

### Hybrid End-to-End Methods
<details open>
<summary>2025</summary>

- **ReAL-AD:**Towards Human-Like Reasoning in End-to-End Autonomous Driving [[paper](https://arxiv.org/pdf/2507.12499)][[project](https://4dvlab.github.io/project_page/realad)]


    Summary: ReAL-AD introduces a human-like hierarchical reasoning mechanism and innovatively integrates the vision-language model organically into the decision-making hierarchy, significantly improving the planning accuracy and safety of the end-to-end autonomous driving system.

- ​**VLAD**​: A VLM-Augmented Autonomous Driving Framework with Hierarchical Planning and Interpretable Decision Process, ​**ITSC2025**​[[Paper](https://arxiv.org/abs/2507.01284)]


    Summary: Combining a fine-tuned VLM with a state-of-the-art end-to-end system, VAD

- ​**LeAD**​: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving[[Paper](https://arxiv.org/abs/2507.05754)]


    Summary: The high-frequency E2E subsystem maintains real-time perception-planning-control cycles, while the low-frequency LLM module enhances scenario comprehension through multi-modal perception fusion with HD maps and derives optimal decisions via chain-of-thought (CoT) reasoning when baseline planners encounter capability limitations.

- ​**NetRoller**​: Interfacing General and Specialized Models for End-to-End Autonomous Driving[[Paper](https://arxiv.org/abs/2506.14589)][[code](https://github.com/Rex-sys-hk/NetRoller)]


    Summary: NetRoller explores the design of adapters to facilitate seamless integration of Vision Language Model (VLM, aka GM) and Specialized Driving Model (SM)

- ​**SOLVE**​: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving, ​**CVPR2025**​[[Paper](https://arxiv.org/pdf/2505.16805?)]


    Summary: SOLVE combines VLM and end-to-end networks through feature-level collaboration (shared visual encoder) and trajectory-level collaboration.

- ​**VERDI**​: VLM-Embedded Reasoning for Autonomous Driving[[Paper](https://arxiv.org/pdf/2505.15925)]


    Summary: Align E2E perception prediction planning with VLM thought chain output, and refine VLM reasoning into E2E

- ​**ALN-P3**​: Unified Language Alignment for Perception, Prediction, and Planning in Autonomous Driving[[Paper](https://arxiv.org/pdf/2505.15158)]


    Summary: A unified co-refinement framework that introduces cross-modal alignment between a "fast" vision-based autonomous driving system and a "slow" language-driven reasoning module.

- ​**VLM-E2E**​: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion[[Paper](https://arxiv.org/pdf/2502.18042?)]


    Summary: The textual representation is integrated into the bird's eye view (BEV) features using VLM for semantic supervision, which enables the model to learn richer feature representations that explicitly capture the driver's attention semantics.

- **DIMA:** Distilling Multi-modal Large Language Models for Autonomous Driving, ​**CVPR2025**​[[Paper](https://arxiv.org/pdf/2501.09757)]


    Summary: Transferring knowledge from a multimodal large language model (MLLM) to a vision-based end-to-end planner via knowledge distillation
</details>


<details open>
<summary>2024</summary>

- **VLM-AD**​: End-to-End Autonomous Driving through Vision-Language Model Supervision[[Paper](https://arxiv.org/pdf/2412.14446)]

    Summary: Using the reasoning annotations generated by VLM as supervisory signals, the planning accuracy of the E2E model is significantly improved and the collision rate is reduced.

- **FASIONAD** : FAst and Slow FusION Thinking Systems for Human-Like Autonomous Driving with Adaptive Feedback​**​ ​**​[[Paper](https://arxiv.org/pdf/2411.18013)]

    Summary: The fast and slow systems of VLMAD and E2EAD, the fast system efficiently manages routine navigation tasks through fast, data-driven path planning, while the slow system handles complex reasoning and decision-making in unfamiliar or challenging scenarios.

- ​**Senna**​: Bridging Large Vision-Language Models and End-to-End Autonomous Driving[[Paper](https://arxiv.org/abs/2410.22313)][[Code](https://github.com/hustvl/Senna)]


    Summary: Senna is an autonomous driving system that integrates a Large Vision-Language Model with an end-to-end model to improve planning safety, robustness and generalization.

- ​**Hint-AD**​: Holistically Aligned Interpretability in End-to-End Autonomous Driving, ​**CoRL 2024**​[[Paper](https://arxiv.org/abs/2409.06702)][[Project](https://air-discover.github.io/Hint-AD/)][[Code](https://github.com/Robot-K/Hint-AD)]


    Summary: An integrated autonomous driving-language system that generates language that is aligned with the overall perception-prediction-planning output of the autonomous driving model

- ​**DriveVLM**​: The Convergence of Autonomous Driving and Large Vision-Language Models, ​**CoRL2024**​[[Paper](https://arxiv.org/abs/2402.12289)][[Project](https://tsinghua-mars-lab.github.io/DriveVLM/)]


    Summary: DriveVLM-Dual, a hybrid system that incorporates DriveVLM and a traditional autonomous pipeline.

- ​**DME-Driver**​: Integrating Human Decision Logic and 3D Scene Perception in Autonomous Driving, ​**AAAI2025**​[[Paper](https://arxiv.org/abs/2401.03641)]


    Summary:  DME-Driver utilizes a powerful vision language model as the decision-maker and a planning-oriented perception model as the control signal generator.

- ​**VLP**​: Vision Language Planning for Autonomous Driving,​**​ CVPR 2024**​[[Paper](https://arxiv.org/abs/2401.05577)]


    Summary: Propose VLP, a Vision Language Planning model, which is composed of novel components ALP and SLP, aiming to improve the ADS from self-driving BEV reasoning and self-driving decision-making aspects, respectively.

</details>
<p align="right">(<a href="#top">back to top</a>)</p> 


## Dataset
### Normal Dataset
<details open>
<summary>Normal Dataset</summary>

- **KITTI**: Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite CVPR 2012 [[Paper](https://ieeexplore.ieee.org/document/6248074)][[Dataset](https://www.cvlibs.net/datasets/kitti/)]
  
  - Summary: A total of 61 scenes were recorded using a 64-beam LiDAR, a stereo camera pair, a color camera, and GPS/IMU; the training set comprises 7,481 labeled frames and the test set comprises 7,518 frames.
  - Support Tasks: 3D Object Detection, 3D Multi-Object Tracking, Semantic Segmentation, Instance Segmentation

- **nuScenes**: A Multimodal Dataset for Autonomous Driving CVPR 2020 [[Paper](https://arxiv.org/abs/1903.11027)][[Dataset](https://www.nuscenes.org/)]

  - Summary: A total of 1,000 scenes (20 seconds each) were collected, including raw data from 6 cameras, 5 radars, and 1 LiDAR sensor; the training set comprises 28,130 samples, the validation set comprises 6,019 samples, and the test set comprises 6,008 samples.
  - Support Tasks: 3D Detection, 3D Multi-Object Tracking, LiDAR Semantic Segmentation, Panoptic Segmentation & Panoptic Tracking, Planning

- **Waymo Open Dataset** Scalability in Perception for Autonomous Driving: Waymo Open Dataset CVPR 2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)][[Dataset](https://waymo.com/intl/zh-cn/research/scalability-in-perception-for-autonomous-driving-waymo-open-dataset/)]

  - Summary: A total of 1,150 scenes (20 seconds each) were collected, including raw data from 5 LiDAR sensors and 5 cameras; the training‐validation set comprises 1,000 scenes, and the test set comprises 150 scenes.
  - Support Tasks: 3D Detection, 3D Tracking

- **Argoverse**: Argoverse: 3D Tracking and Forecasting with Rich Maps, CVPR 2019 [[Paper](https://arxiv.org/abs/1911.02620)][[Dataset](https://www.argoverse.org/)] Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting, NeurIPS 2023 [[Paper](https://arxiv.org/abs/2301.00493)][[Dataset](https://www.argoverse.org/)]


  - Summary: A Sensor Dataset of 1,000 3D-annotated sequences (lidar, ring cameras, stereo cameras, HD maps), a Lidar Dataset of 20,000 unlabeled sequences, a Motion Forecasting Dataset of 250,000 scenarios, and a Map Change Dataset of 1,000 scenarios (200 depicting HD-map changes), all collected across six cities.
  - Support Tasks: 3D Object Detection, 3D Tracking, Motion Forecasting, Map Change Detection

- **Lyft**  One Thousand and One Hours: Self-driving Motion Prediction Dataset ArXiv 2020 [[Paper](https://arxiv.org/abs/2006.14480)][[Dataset](https://level5.lyft.com/)]

  - Summary: Over 1,000 hours of driving data collected in Las Vegas and Palo Alto using a ring of 6 cameras, a 64-beam LiDAR, and GPS/IMU; annotated for agent trajectories and intent.
  - Support Tasks: Motion Prediction
 
- **ONCE** One Million Scenes for Autonomous Driving: ONCE Dataset NeurIPS 2021 [[Paper](https://arxiv.org/abs/2106.11037)][[Dataset](https://once-for-auto-driving.github.io/index.html)]

  - Summary: A million LiDAR sweeps with 28,000 frames annotated (32-beam LiDAR, 5 cameras, GNSS/IMU); training set comprises 25,000 annotated frames, validation 1,500, test 1,500.
  - Support Tasks: 3D Object Detection, 3D Tracking, Semantic Segmentation

- **Mapillary Vistas** The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes ICCV 2017 [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)][[Dataset](https://www.mapillary.com/dataset/vistas)]

  - Summary: A total of 25,000 high-resolution images from diverse street scenes; training set comprises 18,000 images, validation 2,000, test 5,000; pixel-level annotations over 66 object classes.
  - Support Tasks: Semantic Segmentation, Instance Segmentation

- **BDD100K**: A Diverse Driving Dataset for Heterogeneous Multitask Learning CVPR 2020 [[Paper](https://arxiv.org/abs/1805.04687)][[Dataset](https://github.com/bdd100k/bdd100k)]

  - Summary: A total of 100,000 front-camera images; training set comprises 70,000 images, validation 10,000, test 20,000; annotations include bounding boxes, instance masks, drivable areas, lanes, and tracking IDs.
  - Support Tasks: Object Detection, Instance Segmentation, Drivable Area Segmentation, Lane Marking, Multi-Object Tracking

- **ApolloScape**: The ApolloScape Open Dataset for Autonomous Driving and Its Application CVPR 2018 Workshops [[Paper](https://arxiv.org/abs/1803.06184)][[Dataset](https://github.com/ApolloScapeAuto/dataset-api?tab=readme-ov-file#data-download)]

  - Summary: A total of 143,000 images with pixel-level labels and 2 million LiDAR point clouds (40-line LiDAR, 6 cameras); training set comprises 100,000 images, validation 20,000, test 23,000.
  - Support Tasks: Semantic Segmentation, Instance Segmentation, Stereo Matching, Optical Flow, Visual Odometry


</details>

<p align="right">(<a href="#top">back to top</a>)</p>   

### Vision Language Dataset
<details open>
<summary>2025</summary>

- **nuScenesR²-6K**: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving[[Paper](https://arxiv.org/pdf/2509.01944)]

	Summary: nuScenesR²-6K is the first innovative CoT dataset for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation.

- ​**Bench2ADVLM**​: A Closed-Loop Benchmark for Vision-language Models in Autonomous Driving[[Paper](https://arxiv.org/abs/2508.02028)]

    Summary: Bench2ADVLM, a unified hierarchical closed-loop evaluation framework for real-time, interactive assessment of ADVLMs across both simulation and physical platforms, uses routes provided in the Bench2Drive.

- ​​**VLADBench**​: Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving[[Paper](https://arxiv.org/pdf/2503.21505)][[Project](https://github.com/Depth2World/VLADBench?tab=readme-ov-file)][[Dataset](https://huggingface.co/datasets/depth2world/VLADBench)]

    Summary:  VLADBench is a challenging and finegrained dataset featuring close-form QAs that progress from static foundational knowledge and elements to advanced reasoning for dynamic on-road situations. The elaborate VLADBench spans 5 key domains: Traffic Knowledge Understanding, General Element Recognition, Traffic Graph Generation, Target Attribute Comprehension, and Ego Decision-Making and Planning.

- **NuInteract ​**:Extending Large Vision-Language Model for Diverse Interactive Tasks in Autonomous Driving [[Paper](https://arxiv.org/abs/2505.08725v1)][[Project](https://github.com/zc-zhao/DriveMonkey)][[Dataset](https://github.com/zc-zhao/DriveMonkey/releases/download/NuInteract_Dataset/NuInteract.zip)]


    Summary:  NuInteract is constructed based on nuScene. It encompasses 239K images (six single-view images and one surrounding view image for each frame) with high-quality dense captions and 1.3M data across diverse interactive language-based tasks, resulting a total of 1.5M image-text pairs.


- ​**Drive-R1**​: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning[[Paper](https://arxiv.org/abs/2506.18234)]

    Summary:  Drive-R1 is designed to bridges the scenario reasoning and motion planning for AD and firstly undergoes the supervised finetuning on a elaborate dataset containing both long and short COT data. Drive-R1 is encouraged to reason step-by-step from visual input to final planning decisions.

- ​**DriveAction**​: A Benchmark for Exploring Human-like Driving Decisions in VLA Models[[Paper](https://arxiv.org/abs/2506.05667)][[Project](https://huggingface.co/datasets/LiAuto-DriveAction/drive-action)][[Dataset](https://huggingface.co/datasets/LiAuto-DriveAction/drive-action)]

    Summary:  The first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios and leveraging real-world driving data proactively collected by users of production-level autonomous vehicles.

- ​**STSBench**​: A Spatio-temporal Scenario Benchmark for Multi-modal Large Language Models in Autonomous Driving[[Paper](https://arxiv.org/abs/2506.06218)][[Project](https://github.com/LRP-IVC/STSBench/tree/main)][[Dataset](https://huggingface.co/datasets/ivc-lrp/STSBench)]

    Summary:  STSnu is the first benchmark that evaluates the spatio-temporal reasoning capabilities of VLMs based on comprehensive 3D perception and evaluates driving expert VLMs for end-to-end driving, operating on videos from multi-view cameras or LiDAR.

- **DRAMA-ROLISP** HiLM-D: Enhancing MLLMs with Multi-Scale High-Resolution Details for Autonomous Driving, IJCV2025. [[Paper](https://arxiv.org/abs/2309.05186)][[Project](https://github.com/xmed-lab/HiLM-D?tab=readme-ov-file)][[Dataset](https://github.com/xmed-lab/HiLM-D?tab=readme-ov-file)]

    Summary:  HiLM-D, a resource-efficient framework that enhances visual information processing in MLLMs for ROLISP (Risk Object Localization and Intention and Suggestion Prediction) .

- **WOMD-Planning-ADE Benchmark** S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation. CVPR2025. [[Paper](https://arxiv.org/abs/2505.24139)]

- ​**ImpromptuVLA**​: Open Weights and Open Data for Driving Vision-Language-Action Models [[Paper](https://arxiv.org/abs/2505.23757)][[Project](https://github.com/ahydchh/Impromptu-VLA)][[Dataset](https://huggingface.co/datasets/aaaaaap/unstructed)][[Model](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main)]

    Summary: The Impromptu VLA Dataset includes over 80,000 meticulously curated video clips, distilled from over 2M source clips sourced from 8 open-source large-scale datasets.

- **DriveBench**: Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives. ICCV2025. [[Link](https://drive-bench.github.io/)][[Paper](https://arxiv.org/abs/2501.04003)][[Project](https://github.com/drive-bench/toolkit)][[Dataset](https://huggingface.co/datasets/drive-bench/arena)]

    Summary: DriveBench is a benchmark dataset designed to evaluate VLM reliability across 17 settings, encompassing 19,200 frames, 20,498 question-answer pairs, three question types, four mainstream driving tasks, and a total of 12 popular VLMs.

- **SimLingo: ​**Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment, CVPR2025.[[Paper](https://arxiv.org/abs/2503.09594)][[Project](https://github.com/RenzKa/simlingo)][[Dataset](https://huggingface.co/datasets/RenzKa/simlingo)]

    Summary: Proposed model trained in this dataset that can handle three different tasks: (1) closed-loop driving, (2) vision-language understanding, and (3) language-action alignment.

- ​**WOMD-Reasoning**​: A Large-Scale Dataset for Interaction Reasoning in Driving, ICML 2025. [[Paper](https://arxiv.org/abs/2407.04281)][[Project](https://github.com/yhli123/WOMD-Reasoning)][[Dataset](https://waymo.com/open/download)]

    Summary: WOMD-Reasoning is a large-scale Q&As dataset built on WOMD focusing on describing and reasoning traffic rule-induced interactions in driving scenarios，and presents 3 million Q&As on real-world driving scenarios.

- ​**OmniDrive**​: LLM-Agent for Autonomous Driving with 3D Perception, Reasoning and Planning, CVPR 2025. [[Paper](https://arxiv.org/abs/2405.01533)][[Project](https://github.com/NVlabs/OmniDrive)][[Dataset](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0)][[Model](https://huggingface.co/exiawsh/OmniDrive/tree/main)]

    Summary: OmniDrive is a holistic vision-language dataset that aligns agent models with 3D driving tasks through counterfactual reasoning.

- ​**CODA-LM**​: Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases, WACV 2025. [[Paper](https://arxiv.org/abs/2404.10595)][[Project](https://github.com/DLUT-LYZ/CODA-LM)][[Dataset](https://huggingface.co/collections/KaiChen1998/coda-lm-6726500ab7d88dbcf9dc3fd0)]

    Summary: CODA-LM is the very first benchmark for the automatic evaluation of LVLMs for self-driving corner cases.

- **CoVLA: ​**Comprehensive Vision-Language-Action Dataset for Autonomous Driving, WACV 2025 Oral. [[Paper](https://arxiv.org/abs/2408.10845)][[Project](https://turingmotors.github.io/covla-ad/)][[Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset)]

    Summary: CoVLA Dataset is an dataset comprising real-world driving videos spanning more than 80 hours. This dataset leverages a novel, scalable approach based on automated data processing and a caption generation pipeline to generate accurate driving trajectories paired with detailed natural language descriptions of driving environments and maneuvers.

- **nuPrompt ​**: Language Prompt for Autonomous Driving. AAAI 2025. [[Paper](https://arxiv.org/abs/2309.04379)][[Project](https://github.com/wudongming97/Prompt4Driving)][[Dataset](https://github.com/wudongming97/Prompt4Driving/releases/download/v1.0/nuprompt_v1.0.zip)][[Model](https://github.com/wudongming97/Prompt4Driving/releases/download/v1.0/f3_prompttrack_e12.pth)]

    Summary: NuPrompt is the first object-centric language prompt set for driving scenes within 3D, multi-view, and multi-frame space, which expands nuScenes dataset by constructing a total of 40,147 language descriptions, each referring to an average of 7.4 object tracklets.

- ​**Robusto-1 Dataset**​: Comparing Humans and VLMs on real out-of-distribution Autonomous Driving VQA from Peru[[Paper](https://arxiv.org/pdf/2503.07587)][[Dataset](https://huggingface.co/datasets/Artificio/robusto-1/tree/main)]

    Summary: The Robusto-1 dataset that uses dashcam video data from Peru, a country with one of the “worst” (aggressive) drivers in the world, a high traffic index, and a high ratio of bizarre to non–bizarre street objects likely never seen in training. The motivation is preliminarly testing at a cognitive level how well Foundational Visual-Language Models (VLMs) compare to Humans in Driving.

- **DrivingVQA Dataset**  RIV-CoT: Retrieval-Based Interleaved Visual Chain-of-Thought in Real-World Driving Scenarios. [[Paper](https://arxiv.org/abs/2501.04671)][[Project](https://github.com/vita-epfl/RIV-CoT)][[Dataset](https://huggingface.co/datasets/EPFL-DrivingVQA/DrivingVQA)]

    Summary: DrivingVQA is a visual question answering dataset derived from driving theory exams, which contains 3,931 multiple-choice problems with expert-written explanations and grounded entities relevant to the reasoning process.

- ​**DriveLMM-o1**​: A Step-by-Step Reasoning Dataset and Large Multimodal Model for Driving Scenario Understanding. [[Paper](https://arxiv.org/abs/2503.10621v1)][[Project](https://github.com/ayesha-ishaq/DriveLMM-o1)][[Dataset](https://huggingface.co/datasets/ayeshaishaq/DriveLMMo1)][[Model](https://huggingface.co/ayeshaishaq/DriveLMMo1)]

    Summary: DriveLMM-o1 is a dataset designed to advance step-wise visual reasoning for autonomous driving, which features over 18k VQA examples in the training set and more than 4k in the test set, each enriched with step-by-step reasoning to ensure logical inference in autonomous driving scenarios.

</details>

<details open>
<summary>2024</summary>

- ​**DriveLM**​: Driving with Graph Visual Question Answering, ECCV 2024.[[Paper](https://arxiv.org/abs/2312.14150)][[Project](https://github.com/OpenDriveLab/DriveLM)][[Dataset](https://huggingface.co/datasets/OpenDriveLab/DriveLM)]

    Summary: DriveLM proposed a task, Graph VQA, which is more similar to the human reasoning process. And instantiate datasets (DriveLM-Data) built upon nuScenes and CARLA, and propose a VLM-based baseline approach (DriveLM-Agent) for jointly performing Graph VQA and end-to-end driving.

- ​**LMDrive**​: Closed-Loop End-to-End Driving with Large Language Models [[Link](https://hao-shao.com/projects/lmdrive.html)][[Paper](https://arxiv.org/abs/2312.07488)][[Project](https://github.com/opendilab/LMDrive)][[Dataset](https://huggingface.co/datasets/OpenDILabCommunity/LMDrive)]

    Summary: LMDrive uniquely processes and integrates multi-modal sensor data with natural language instructions, enabling interaction with humans and navigation software in realistic instructional settings. To facilitate further research in language-based closed-loop autonomous driving, the author also publicly release the corresponding dataset which includes approximately 64K instruction-following data clips, and the LangAuto benchmark that tests the system's ability to handle complex instructions and challenging driving scenarios.

- ​**DriveCoT**​: Integrating Chain-of-Thought Reasoning  with End-to-End Driving [[Link](https://drivecot.github.io/index.html)][[Paper](https://arxiv.org/abs/2403.16996)][[Project](https://drivecot.github.io/index.html)][[Dataset](https://drivecot.github.io/download.html)]

    Summary: The author collect a comprehensive end-to-end driving dataset named DriveCoT, leveraging the CARLA simulator. It contains sensor data, control decisions, and chain-of-thought labels to indicate the reasoning process.

- **NuScenes-QA: ​**A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario, AAAI 2024. [[Paper](https://arxiv.org/pdf/2305.14836)][[Project](https://github.com/qiantianwen/NuScenes-QA)][[Dataset](https://drive.google.com/drive/folders/1jIkICT23wZWZYPrWCa0x-ubjpClSzOuU)]

    Summary: NuScenes-QA is the first benchmark for VQA in the autonomous driving scenario, encompassing 34K visual scenes and 460K question-answer pairs.

- **NuScenes-MQA: ​**Integrated Evaluation of Captions and QA for Autonomous  Driving Datasets using Markup Annotations, WACV2024. [[Paper](https://arxiv.org/abs/2312.06352)][[Project](https://github.com/turingmotors/NuScenes-MQA)][[Dataset](https://drive.google.com/drive/u/0/folders/1PQy0qhTtbdueIVlVnn4jC6xvANZUynRZ)]

    Summary:  NuScenes-MQA dataset is annotated by a novel annotation technique Markup-QA,  in which QAs are enclosed within markups.

- ​**Talk2BEV**​:Language-enhanced Bird’s-eye View Maps for Autonomous Driving, ICRA 2024. [[Paper](https://arxiv.org/abs/2310.02251)][[Project](https://github.com/llmbev/talk2bev)][[Dataset](https://drive.google.com/file/d/1B5Uong8xYGRDkufR33T9sCNyNdRzPxc4/view?usp=sharing)]

    Summary: Talk2BEV-Bench, a benchmark for evaluating LVLMs for AD applications with human-annotated ground-truth for object attributes, semantics, visual reasoning, spatial understanding, and decision-making.

- ​**DriveGPT4**​: Interpretable End-to-end Autonomous Driving via Large Language Model, RA-L 2024. [[Paper](https://arxiv.org/abs/2310.01412)][[Project](https://drive.google.com/drive/folders/1PsGL7ZxMMz1ZPDS5dZSjzjfPjuPHxVL5)][[Dataset](https://drive.google.com/drive/folders/1PsGL7ZxMMz1ZPDS5dZSjzjfPjuPHxVL5)]

    Summary: DriveGPT4 develop a new visual instruction tuning dataset for interpretable autonomous driving with the assistance of ChatGPT.

- ​**ContextVLM**​: Zero-Shot and Few-Shot Context Understanding for Autonomous Driving using Vision Language Models, ITSC 2024. [[Paper](https://arxiv.org/abs/2409.00301)][[Project](https://github.com/ssuralcmu/ContextVLM)][[Dataset](https://drive.google.com/drive/folders/1bUNDIlhgTjcxt0WaOGVRRuI0e5_cWJDG)]

    Summary: A large dataset named  DrivingContexts with a combination of hand-annotated  and machine annnotated labels to improve VLMs for  better context recognition.Its images are from muti-data sources.

- ​**LingoQA**​: Visual Question Answering for Autonomous Driving, ECCV 2024. [[Paper](https://arxiv.org/abs/2312.14115)][[Project](https://github.com/wayveai/LingoQA)][[Dataset](https://github.com/wayveai/LingoQA)]

    Summary: LingoQA Datasets with 419.9k QA pair stands out with its freeform questions and answers, covering not just perception but also driving reasoning from the drivers directly, broadening the scope of autonomousdriving datasets.

- ​**Rank2Tell**​: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning, WACV2024. [[Paper](https://arxiv.org/abs/2309.06597)][[Dataset](https://usa.honda-ri.com/rank2tell)]

    Summary: Rank2Tell, a multi-modal ego-centric dataset for Ranking the importance level and Telling the reason for the importance.Using various close and open-ended visual question answering, the dataset provides dense annotations of various semantic, spatial, temporal, and relational attributes of various important objects in complex traffic scenarios.

- ​**MAPLM**​: A Real-World Large-Scale Vision-Language Dataset for Map and Traffic Scene Understanding, CVPR2024. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MAPLM_A_Real-World_Large-Scale_Vision-Language_Benchmark_for_Map_and_Traffic_CVPR_2024_paper.pdf)][[Project](https://github.com/LLVM-AD/MAPLM?tab=readme-ov-file#official-open-source-datasets-of-1st-workshop-on-large-language-vision-models-for-autonomous-driving-llvm-ad-in-wacv-2024)][[Dataset](https://drive.google.com/drive/folders/1cqFjBH8MLeP6nKFM0l7oV-Srfke-Mx1R)]

    Summary: A new vision-language benchmark that can be used to finetune traffic and HD map domain-specific foundation models.

- ​**NuInstruct**​: Holistic Autonomous Driving Understanding by Bird’s-Eye-View Injected Multi-Modal Large Models, CVPR2024. [[Paper](https://arxiv.org/pdf/2401.00988)][[Project](https://github.com/xmed-lab/NuInstruct)][[Dataset](https://drive.google.com/file/d/1ybwvgnPFRVwmEAJudWwnGQ8wLitzQ82O/view?usp=sharing)]

    Summary: NuInstruct, a novel dataset with 91K multi-view video-QA pairs across 17 subtasks, where each task demands holistic information ( e.g., temporal, multi-view, and spatial), significantly elevating the challenge level.

- **DriveVLM（SUP-AD dataset）**: The Convergence of Autonomous Driving and Large Vision-Language Models, CoRL 2024. [[paper](https://arxiv.org/abs/2402.12289)][[Project](https://tsinghua-mars-lab.github.io/DriveVLM/)]

    Summary: carry out a comprehensive data mining and annotation pipeline to construct an in-house SUP-AD dataset for the scene understanding and planning task.

- ​**SURDS**​: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models [[Paper](https://arxiv.org/abs/2411.13112)][[Project](https://github.com/XiandaGuo/Drive-MLLM)][[Dataset](https://huggingface.co/datasets/bonbon-rj/SURDS/tree/main)]

    Summary: SURDS, a large-scale benchmark designed to systematically evaluate the spatial reasoning capabilities of vision language models (VLMs). Built on the nuScenes dataset, SURDS comprises 41,080 vision-question-answer training instances and 9,250 evaluation samples, spanning six spatial categories: orientation, depth estimation, pixel-level localization, pairwise distance, lateral ordering, and front-behind relations.
</details>

<details open>
<summary>2023</summary>

- ​**DriveMLM**​: Aligning Multi-Modal Large Language Models with  Behavioral Planning States for Autonomous Driving [[Paper](https://arxiv.org/abs/2312.09245)][[Project](https://github.com/OpenGVLab/DriveMLM)]

    Summary: The author design an effective data engine to collect a dataset that includes decision state and corresponding explanation annotation for model training and evaluation.。

- ​**Reason2Drive**​: Towards Interpretable and Chain-based Reasoning for Autonomous Driving [[Paper](https://arxiv.org/abs/2312.03661)][[Project](https://github.com/fudan-zvg/reason2drive)][[Dataset\_mini](https://drive.google.com/file/d/16IInbGqEzg4UcNhTlxVA9tS6tOTi4wet/view?usp=sharing)]

    Summary: Reason2Drive, a benchmark dataset with over 600K video-text pairs, aimed at facilitating the study of interpretable reasoning in complex driving environments.

- **Refer-KITTI dataset**: Referring Multi-Object Tracking. CVPR2023. [[Paper](https://arxiv.org/abs/2303.03366)][[Project](https://github.com/wudongming97/RMOT)][[Dataset](https://github.com/wudongming97/RMOT)]

    Summary: The author proposes a new and general referring understanding task, termed referring multi-object tracking (RMOT).To push forward RMOT, they construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI, which provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects.

- ​**DRAMA**​: Joint Risk Localization and Captioning in Driving, WACV2023. [[Paper](https://arxiv.org/abs/2209.10767)][[Project](https://usa.honda-ri.com/drama)][[Dataset](https://usa.honda-ri.com/drama)]

    Summary: DRAMA (Driving Risk Assessment Mechanism with A captioning module), which consists of 17,785 interactive driving scenarios collected in Tokyo, Japan. Our DRAMA dataset accommodates video- and object-level questions on driving risks with associated important objects to achieve the goal of visual captioning as a free-form language description utilizing closed and open-ended responses for multi-level questions, which can be used to evaluate a range of visual captioning capabilities in driving scenarios.

</details>

<details open>
<summary>Before 2023</summary>

- ​**SUTD-TrafficQA**​: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events, CVPR 2021.[[Paper](https://arxiv.org/abs/2103.15538)][[Project](https://github.com/SUTDCV/SUTD-TrafficQA)][[Dataset](https://sutdcv.github.io/SUTD-TrafficQA/#/download)]

    Summary: SUTD-TrafficQA takes the form of video QA based on the collected 10,080 in-the-wild videos and annotated 62,535 QA pairs, for benchmarking the cognitive capability of causal inference and event understanding models in complex traffic scenarios.

- **BDD-OIA dataset**: Explainable Object-induced Action Decision for Autonomous Vehicles, CVPR 2020. [[Paper](https://arxiv.org/pdf/2003.09405)][[Project](https://github.com/Twizwei/bddoia_project)][[Dataset](https://drive.google.com/file/d/1WFiwRi_sMA_McZnkbEjh8Rnl-Im7_9Mk/view)]

    Summary: A large dataset annotated for both driving commands and explanations.

- **HAD**: Grounding Human-to-Vehicle Advice for Self-driving Vehicles, CVPR 2019. [[Paper](https://arxiv.org/abs/1911.06978)][[Dataset](https://usa.honda-ri.com/HAD#Downloadthedataset)]

    Summary: The Honda Research Institute-Advice Dataset (HAD) is driving data with natural language advices. The driving videos are gathered from HDD dataset, a large-scale naturalistic driving dataset collected in San Francisco Bay Area.

- **BDD-X dataset**: Textual Explanations for Self-Driving Vehicles, ECCV 2018.[[Paper](https://arxiv.org/abs/1807.11546)][[Project](https://github.com/JinkyuKimUCB/explainable-deep-driving)][[Dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset)]

    Summary: Berkeley DeepDrive eXplanation (BDD-X) dataset is a large-scale dataset with over 6,984 video clips annotated with driving descriptions.

- ​**Talk2Car**​:**​ ​**Taking Control of Your Self-Driving Car, EMNLP 2019. [[Paper](https://arxiv.org/abs/1909.10838)][[Project](https://github.com/talk2car/Talk2Car)]

    Summary: The first object referral dataset that contains commands written in natural language for self-driving cars, provides a detailed comparison with related datasets such as ReferIt, RefCOCO, RefCOCO+, RefCOCOg, Cityscape-Ref and CLEVR-Ref.

</details>

<p align="right">(<a href="#top">back to top</a>)</p>   

## License



<p align="right">(<a href="#top">back to top</a>)</p>    


## Citation
If you find this project useful in your research, please consider citing:
```BibTeX

```

<p align="right">(<a href="#top">back to top</a>)</p>    

## Contact

<p align="right">(<a href="#top">back to top</a>)</p>    


<!-- links -->
[your-project-path]:AutoLab-SAI-SJTU/GE2EAD
[contributors-shield]: https://img.shields.io/github/contributors/AutoLab-SAI-SJTU/GE2EAD.svg?style=flat-square
[contributors-url]: https://github.com/AutoLab-SAI-SJTU/GE2EAD/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AutoLab-SAI-SJTU/GE2EAD.svg?style=flat-square
[forks-url]: https://github.com/AutoLab-SAI-SJTU/GE2EAD/network/members
[stars-shield]: https://img.shields.io/github/stars/AutoLab-SAI-SJTU/GE2EAD.svg?style=flat-square
[stars-url]: https://github.com/AutoLab-SAI-SJTU/GE2EAD/stargazers
[issues-shield]: https://img.shields.io/github/issues/AutoLab-SAI-SJTU/GE2EAD.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/AutoLab-SAI-SJTU/GE2EAD.svg
[license-shield]: https://img.shields.io/github/license/AutoLab-SAI-SJTU/GE2EAD.svg?style=flat-square
[license-url]: https://github.com/AutoLab-SAI-SJTU/GE2EAD/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian