import os
import requests
import pypdf

# 1. THE COMPLETE LATEX TEXT OF THE RESEARCH PROPOSAL WITH SYNC TAGS
# Strictly formatted according to Makerere University DRGT guidelines (March 2011):
# - 1.5 line spacing throughout
# - Times New Roman font (size 12pt)
# - Margins: 1 inch on all sides
# - Flush left, block paragraphs (no indent, 1 line spacing between paragraphs)
# - No numbered headings (unwanted by guidelines)
# - Main headings (Level 1): Centered, Bold, Title Case
# - Subheadings (Level 3): Flush Left, Underlined, Bold/Title Case
# - Paragraph subheadings (Level 4): Indented, Underlined, Title Case/Italic, ending with period
# - Preliminary page numbers: Roman numerals (i, ii, iii) in lower center (Title page unnumbered)
# - Main body page numbers: Arabic numerals (1, 2, 3) in lower center
# - References: APA 6th Edition format, unnumbered, hanging indent
# - Appendices: Budget in UGX (with 15% overhead check), Gantt Chart table, Explanatory notes
latex_text = r"""
\documentclass[12pt]{article}
\usepackage{times}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{tikz}
\usetikzlibrary{positioning, shapes.geometric}
\usepackage{booktabs}
\usepackage{caption}

% Page Geometry (1 inch margins all sides)
\geometry{
  a4paper,
  top=1.0in,
  bottom=1.0in,
  left=1.0in,
  right=1.0in
}

% 1.5 Line Spacing
\onehalfspacing

% Paragraph Formatting: Block paragraphs, no indent, space between paragraphs
\setlength{\parindent}{0pt}
\setlength{\parskip}{\baselineskip}

% Global Left Alignment (not justified)
\raggedright

% Custom Hanging Indent Environment for References
\newenvironment{references}{
  \clearpage
  \section*{\centering \textbf{REFERENCES}}
  \begin{list}{}{
    \setlength{\leftmargin}{0.5in}
    \setlength{\itemindent}{-0.5in}
    \setlength{\itemsep}{\baselineskip}
    \setlength{\parsep}{0pt}
  }
}{
  \end{list}
}

\begin{document}

% ==========================================
% TITLE PAGE (No Page Numbering)
% ==========================================
\thispagestyle{empty}
\begin{center}
    {\large \textbf{MAKERERE \hfill UNIVERSITY}} \\
    \vspace{0.2in}
    {\small \textbf{DIRECTORATE OF RESEARCH AND GRADUATE TRAINING}} \\
    \vspace{0.4in}
    
    % TikZ Seal Placeholder
    \begin{tikzpicture}[scale=1.1]
      \draw[thick, fill=blue!5] (0,1.6) .. controls (1.6,1.6) and (1.6,0) .. (0,-1.6) .. controls (-1.6,0) and (-1.6,1.6) .. (0,1.6);
      \draw[thick, fill=red!10] (0,1.3) .. controls (1.3,1.3) and (1.3,0.1) .. (0,-1.3) .. controls (-1.3,0.1) and (-1.3,1.3) .. (0,1.3);
      \node at (0,0.6) [font=\tiny\bfseries] {MAKERERE};
      \node at (0,0.0) [font=\tiny\bfseries] {UNIVERSITY};
      \node at (0,-0.6) [font=\tiny] {EST. 1922};
    \end{tikzpicture} \\
    
    \vspace{0.5in}
    {\large \textbf{BEYOND THE LINE OF SIGHT: COUNTERFACTUAL AMODAL ANCHORS FOR OCCLUSION-AWARE MULTI-ANIMAL TRACKING IN UAV-BASED PRECISION LIVESTOCK MONITORING}} \\
    \vspace{0.5in}
    
    \textbf{BY} \\
    \vspace{0.2in}
    \textbf{TUMUSIIME DEUS} \\
    {\small B.Sc. Software Engineering (Makerere University)} \\
    \textbf{Reg No: 2025/HD05/26375U} \\
    \vspace{0.5in}
    
    \textbf{A RESEARCH PROPOSAL SUBMITTED TO THE DEPARTMENT OF COMPUTER SCIENCE, COLLEGE OF COMPUTING AND INFORMATICS TECHNOLOGY IN PARTIAL FULFILMENT OF THE AWARD OF THE DEGREE OF MASTER OF SCIENCE IN COMPUTER SCIENCE OF MAKERERE UNIVERSITY} \\
    \vspace{0.5in}
    
    \textbf{JULY 2026}
\end{center}
\clearpage

% ==========================================
% PRELIMINARY PAGES (Roman numerals, lower center)
% ==========================================
\pagenumbering{roman}
\pagestyle{plain}
\setcounter{page}{1}

% DECLARATION
\section*{\centering \textbf{Declaration}}
% [[START: SECTION: Declaration]]
I, \textbf{Tumusiime Deus}, declare that this research proposal is my original work and has not been submitted for any other degree award to any other university before.

Signature: \underline{\hspace{2.5in}} \hfill Date: \underline{\hspace{1.5in}}

\vspace{1.0in}
\textbf{Approval by Supervisors} \\
This research proposal has been submitted for evaluation with the approval of the following supervisors:

\textbf{Dr. George Nasinyama} \\
Department of Computer Science \\
College of Computing and Informatics Technology, Makerere University \\
Signature: \underline{\hspace{2.5in}} \hfill Date: \underline{\hspace{1.5in}}

\vspace{0.5in}
\textbf{Prof. Celestino Obua} \\
Directorate of Research and Graduate Training, Makerere University \\
Signature: \underline{\hspace{2.5in}} \hfill Date: \underline{\hspace{1.5in}}
% [[END: SECTION: Declaration]]
\clearpage

% ABSTRACT
\section*{\centering \textbf{Abstract}}
% [[START: SECTION: Abstract]]
Precision Livestock Farming (PLF) increasingly relies on unmanned aerial vehicles (UAVs) equipped with RGB and thermal cameras to automate cattle counting, grazing trajectory monitoring, and health assessment. While deep learning models have improved object detection and tracking in open pastures, prolonged visual occlusion caused by trees, shrubs, terrain folds, and animal clustering remains a major challenge. Standard tracking-by-detection systems treat occlusion as a passive state of missing observations, relying on simple motion models or appearance re-identification. Consequently, trajectory fragmentation, identity switching, and inaccurate behavioral modeling are common in extensive grazing systems. 

This research proposes a novel perception framework termed Counterfactual Amodal Anchors (CAA) for occlusion-aware multi-animal tracking. Instead of terminating tracks or allowing occluded areas to collapse into featureless voids, our framework maps the geometric projection of occlusion shadows onto a local ground plane and seeds them with active query vectors. These amodal anchors continuously update their state using temporal context, herd-level social dynamics, and terrain priors. 

We formulate mathematical projection models, spatiotemporal deformable attention query updates, and hindsight loss functions to supervise the network during training. The framework will be evaluated on drone datasets collected from Ugandan pastures, specifically at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK). By reframing occlusion as a structured probabilistic estimation problem, this work aims to establish a robust foundation for automated cattle tracking, lameness screening, and pasture management in Sub-Saharan Africa.
% [[END: SECTION: Abstract]]
\clearpage

% TABLE OF CONTENTS & LISTS
\tableofcontents
\clearpage
\listoftables
\clearpage
\listoffigures
\clearpage

% LIST OF ACRONYMS
\section*{\centering \textbf{List of Acronyms and Abbreviations}}
\begin{tabular}{ll}
    UAV & Unmanned Aerial Vehicle \\
    PLF & Precision Livestock Farming \\
    CAA & Counterfactual Amodal Anchors \\
    BEV & Bird's-Eye View \\
    LWIR & Long-Wave Infrared \\
    MOT & Multi-Object Tracking \\
    MOTA & Multi-Object Tracking Accuracy \\
    IDF1 & Identity F1 Score \\
    HOTA & Higher Order Tracking Accuracy \\
    IDSW & Identity Switches \\
    ExG & Excess Green Index \\
    CTRV & Constant Turn Rate and Velocity \\
    MHT & Multi-Hypothesis Tracking \\
    NDP & National Development Plan \\
    SDG & Sustainable Development Goal \\
    MUARIK & Makerere University Agricultural Research Institute Kabanyolo \\
\end{tabular}
\clearpage

% ==========================================
% MAIN BODY (Arabic numerals, lower center)
% ==========================================
\pagenumbering{arabic}
\setcounter{page}{1}

% CHAPTER 1
\section*{\centering \textbf{CHAPTER ONE: INTRODUCTION}}

\subsection*{\underline{\textbf{Background to the Study}}}
% [[START: SECTION: Background to the Study]]
Livestock agriculture is a foundational pillar of food security, rural livelihoods, and macroeconomic stability across Sub-Saharan Africa, and particularly in Uganda. According to the Uganda Bureau of Statistics, livestock production contributes significantly to the agricultural gross domestic product and employs millions of rural households. Effective pasture management, grazing behavior analysis, and early disease detection require continuous, accurate monitoring of animal herds. Traditional observation protocols remain overwhelmingly manual, making them labor-intensive, subjective, and difficult to scale across extensive ranching environments.

Recently, the integration of Unmanned Aerial Vehicles (UAVs) equipped with high-resolution RGB and Long-Wave Infrared (LWIR) thermal cameras has emerged as a promising tool for automated monitoring. UAVs provide a top-down perspective, allowing detection models like Real-Time DEtection Transformer (RT-DETR) and Segment Anything 2 (SAM 2) to identify individual animals, and tracking frameworks like ByteTrack or Observation-Centric SORT (OC-SORT) to construct motion trajectories.

Despite these technological advancements, visual occlusion remains the single most significant bottleneck preventing the deployment of fully automated systems. In extensive grazing pastures, cattle frequently move behind obstacles such as Acacia canopies, dense bushes, water troughs, shading infrastructure, and behind one another during herd aggregation. 

When an animal becomes occluded, current tracking-by-detection systems behave passively. They treat the occluded space as a visual null zone, immediately terminating the animal's trajectory or relying on linear constant-velocity Kalman filters to extrapolate its position. Because animals change direction and speed when grazing, linear extrapolation quickly diverges from the true path. Once the animal re-emerges, the tracking system is forced to re-initialize its identity, which frequently results in track fragmentation or identity switching (assigning the ID of one animal to another). 
% [[END: SECTION: Background to the Study]]

\subsection*{\underline{\textbf{Statement of the Problem}}}
% [[START: SECTION: Statement of the Problem]]
Existing multi-animal tracking systems are fundamentally reactive, relying on immediate visual verification and falling back to passive prediction during extended occlusions. This operational passivity induces a significant perception lag and degrades downstream behavioral analytics. For example, if a monitoring system is trying to measure the total daily grazing distance or identify signs of lameness (which requires precise step-by-step trajectory analysis), a single identity switch or fragmented track can invalidate hours of data collection. 

This research proposes a conceptual shift: from reactive observation-based tracking to proactive counterfactual reasoning. We present the position that occluded areas cast by vegetation and structures must be modeled as dynamic probabilistic occupancy spaces populated by explicit, active query vectors termed Counterfactual Amodal Anchors (CAA). Rather than deleting the animal's track, the network maps the spatial boundaries of the occlusion shadows and initializes virtual anchors that actively query temporal features, local visual context, and herd cohesion priors. 

This research aligns with Uganda's National Development Plan (NDP III/IV), which identifies agricultural modernization, digitization, and value addition as key drivers of economic growth. Furthermore, it supports the African Union (AU) Agenda 2063 (Aspiration 1: A prosperous Africa based on inclusive growth and sustainable development) and the United Nations Sustainable Development Goals (SDG 2: Zero Hunger, SDG 9: Industry, Innovation, and Infrastructure, and SDG 15: Life on Land) by introducing intelligent, automated technology to optimize food production and animal welfare.
% [[END: SECTION: Statement of the Problem]]

\subsection*{\underline{\textbf{Objectives of the Study}}}

\hspace*{0.5in}\underline{\textbf{General objective.}}
% [[START: SECTION: General Objective]]
The general objective of this research is to develop a robust, occlusion-aware multi-animal tracking framework using Counterfactual Amodal Anchors (CAA) to maintain identity continuity and trajectory reconstruction under vegetation-induced occlusions in UAV-based precision livestock monitoring.
% [[END: SECTION: General Objective]]

\hspace*{0.5in}\underline{\textbf{Specific objectives.}}
% [[START: SECTION: Specific Objectives]]
To achieve this general objective, the study will address the following specific objectives:
\begin{enumerate}
    \item To design and implement a ground-plane projection homography model to map UAV camera views and identify vegetation-induced occlusion masks ($\mathcal{M}_{occ}$).
    \item To formulate and integrate an active amodal anchor seeding mechanism using constant turn rate and velocity (CTRV) motion models and herd-cohesion priors to represent hidden targets.
    \item To develop a spatiotemporal query routing network using deformable attention to query historical frames and update anchor states.
    \item To evaluate the performance of the proposed Counterfactual Amodal Anchor (CAA) framework against traditional tracking baselines (ByteTrack, OC-SORT, DeepSORT) using standard Multi-Object Tracking metrics.
\end{enumerate}
% [[END: SECTION: Specific Objectives]]

\subsection*{\underline{\textbf{Research Questions}}}
% [[START: SECTION: Research Questions]]
This study seeks to answer the following three core research questions:
\begin{itemize}
    \item \textbf{RQ1:} Can Counterfactual Amodal Anchors (CAA) reduce identity switches (IDSW) under prolonged vegetation-induced occlusion in UAV feeds?
    \item \textbf{RQ2:} Can CAA improve trajectory continuity metrics, specifically Higher Order Tracking Accuracy (HOTA) and Identity F1 Score (IDF1), compared to baseline trackers (ByteTrack, OC-SORT)?
    \item \textbf{RQ3:} Does the integration of thermal (LWIR) imagery significantly improve anchor update accuracy in dense vegetation and low-contrast illumination compared to RGB-only tracking?
\end{itemize}
% [[END: SECTION: Research Questions]]

\subsection*{\underline{\textbf{Research Hypotheses}}}
% [[START: SECTION: Research Hypotheses]]
Based on the theoretical design of our proactive amodal tracking framework, we formulate the following hypotheses:
\begin{itemize}
    \item \textbf{H1:} Seeding amodal anchors inside projected vegetation masks will significantly reduce identity switches (IDSW) and track fragmentations compared to baseline extrapolation during prolonged occlusions.
    \item \textbf{H2:} The integration of herd-cohesion social priors and spatiotemporal attention queries yields significantly higher tracking accuracy (MOTA, IDF1, HOTA) under dense foliage than standard constant-velocity motion models.
\end{itemize}
% [[END: SECTION: Research Hypotheses]]

\subsection*{\underline{\textbf{Significance of the Study}}}
% [[START: SECTION: Significance of the Study]]
This study contributes to both the theory and application of computer vision in precision agriculture. Academically, it introduces the concept of amodal perception—historically applied in autonomous driving Bird's-Eye View (BEV) networks—to agricultural monitoring, creating new methodologies for tracking hidden agents. 

Practically, it provides Ugandan ranchers and veterinary officers with a reliable, automated tool to screen for lameness, monitor grazing efficiency, and count herds without labor-intensive manual observation. This technological advancement supports agricultural digitization, enhances animal welfare, and improves the profitability of extensive livestock systems.
% [[END: SECTION: Significance of the Study]]

\subsection*{\underline{\textbf{Justification of the Study}}}
% [[START: SECTION: Justification of the Study]]
Without a dedicated mechanism to handle visual occlusion, automated cattle tracking remains unviable for real-world deployment. Standard systems immediately lose animal tracks under trees, requiring manual identity corrections and causing severe data gaps. Manual tracking of thousands of animals on extensive ranches is impossible. 

This research is justified because it solves the occlusion bottleneck mathematically and computationally, ensuring that UAV-based monitoring can operate autonomously and reliably under realistic, vegetation-heavy grazing conditions.
% [[END: SECTION: Justification of the Study]]

\subsection*{\underline{\textbf{Scope of the Study}}}

\hspace*{0.5in}\underline{\textbf{Content scope.}}
% [[START: SECTION: Scope of the Study - Content Scope]]
The research focuses on the development of ground-plane projection homographies, foliage segmentation, amodal query seeding, spatiotemporal deformable attention query routing, and re-emergence ReID matching. It does not cover drone control automation or path planning.
% [[END: SECTION: Scope of the Study - Content Scope]]

\hspace*{0.5in}\underline{\textbf{Geographical scope.}}
% [[START: SECTION: Scope of the Study - Geographical Scope]]
The drone video dataset is collected from active pastures at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK), located in Wakiso District, Uganda.
% [[END: SECTION: Scope of the Study - Geographical Scope]]

\hspace*{0.5in}\underline{\textbf{Time scope.}}
% [[START: SECTION: Scope of the Study - Time Scope]]
The research will be conducted over a 12-month period, corresponding to the 2025/2026 academic year.
% [[END: SECTION: Scope of the Study - Time Scope]]

\hspace*{0.5in}\underline{\textbf{Theoretical scope.}}
% [[START: SECTION: Scope of the Study - Theoretical Scope]]
The study is situated within the domains of computer vision, deep learning, multi-object tracking, amodal perception, and Bayesian kinematic estimation.
% [[END: SECTION: Scope of the Study - Theoretical Scope]]

\clearpage

% ==========================================
% CHAPTER TWO: LITERATURE REVIEW
% ==========================================
\section*{\centering \textbf{CHAPTER TWO: LITERATURE REVIEW}}

\subsection*{\underline{\textbf{Introduction}}}
% [[START: SECTION: Literature Review - Introduction]]
This chapter reviews literature related to automated animal tracking, Multi-Object Tracking (MOT) paradigms, and amodal perception models. It highlights the state-of-the-art, discusses the limitations of existing frameworks, and establishes the research gap that the proposed Counterfactual Amodal Anchor framework aims to address.
% [[END: SECTION: Literature Review - Introduction]]

\subsection*{\underline{\textbf{Precision Livestock Farming and UAV Monitoring}}}
% [[START: SECTION: PLF and UAV Monitoring]]
Precision Livestock Farming (PLF) uses technology to monitor livestock automatically. Recent research has heavily utilized Unmanned Aerial Vehicles (UAVs) to count and monitor cattle, sheep, and horses. UAVs provide a top-down view that covers wide pasture areas quickly. Xu et al. (2020) demonstrated the use of Mask R-CNN for automated cattle counting in quadcopter vision systems. Similarly, Qiao et al. (2023) developed a cattle body detection framework using adaptively fused features to locate animals in complex grazing backgrounds. Nonetheless, these frameworks focus primarily on detection and fail to maintain track continuity when animals move under dense canopies.
% [[END: SECTION: PLF and UAV Monitoring]]

\subsection*{\underline{\textbf{Multi-Object Tracking Frameworks}}}
% [[START: SECTION: Multi-Object Tracking Frameworks]]
The Multi-Object Tracking domain is dominated by the tracking-by-detection paradigm. Classic algorithms like Simple Online and Realtime Tracking (SORT) by Bewley et al. (2016) and DeepSORT by Wojke et al. (2017) utilize Kalman filters to predict linear motion and associate detections using Hungarian matching. Recent SOTA trackers like ByteTrack (Zhang et al., 2022) and OC-SORT (Cao et al., 2023) focus on recovering low-confidence detections and mitigating linear motion assumptions during short-term occlusions. However, all these methods remain fundamentally reactive, relying on immediate visual verification. During extended occlusions under tree canopies, the error covariance of the Kalman filter grows quadratically, leading to track termination or identity switches when the animal emerges.
% [[END: SECTION: Multi-Object Tracking Frameworks]]

\subsection*{\underline{\textbf{Amodal Perception and BEV Networks}}}
% [[START: SECTION: Amodal Perception and BEV Networks]]
Amodal perception refers to the cognitive ability to perceive the entirety of a physical structure when only portions of it are visible. In computer vision, amodal segmentation frameworks like pix2gestalt (Ozguroglu et al., 2024) predict hidden boundaries of objects. In autonomous driving, Bird's-Eye View (BEV) networks like BEVFormer (Li et al., 2022) and BEVFusion (Liu et al., 2023) map perspective camera features into 3D voxel spaces to represent occluded pedestrians and vehicles. 

Adapting BEV amodal reasoning to livestock monitoring is a promising direction. By treating occluded vegetation shadows not as visual null zones but as probabilistic occupancy regions, we can maintain track continuity. The problem of active, dynamic tracking of multiple mobile agents within large, vegetation-induced geometric shadows remains unaddressed in precision agriculture reviews (Santamaria et al., 2023).
% [[END: SECTION: Amodal Perception and BEV Networks]]

\subsection*{\underline{\textbf{Research Gap Analysis}}}
% [[START: SECTION: Research Gap Analysis]]
Existing literature reveals three main gaps:
\begin{enumerate}
    \item Lack of explicit ground-plane mapping of vegetation-induced occlusion masks from UAV perspective feeds.
    \item Absence of active amodal anchor queries within mapped occlusion regions to represent hidden animals.
    \item Failure to integrate herd-cohesion behavior priors into state prediction models to constrain search windows.
\end{enumerate}
Our proposed Counterfactual Amodal Anchor framework addresses these gaps directly.
% [[END: SECTION: Research Gap Analysis]]

\clearpage

% ==========================================
% CHAPTER THREE: METHODOLOGY
% ==========================================
\section*{\centering \textbf{CHAPTER THREE: METHODOLOGY}}

\subsection*{\underline{\textbf{Research Design and Approach}}}
% [[START: SECTION: Methodology - Research Design and Approach]]
This study adopts a quantitative, quasi-experimental research design. We build an algorithmic tracking pipeline, deploy it on simulated and real-world UAV video datasets, and evaluate performance changes quantitatively using established MOT metrics. The approach involves developing modular components in PyTorch and OpenCV, and benchmarking them on a workstation.
% [[END: SECTION: Methodology - Research Design and Approach]]

\subsection*{\underline{\textbf{Conceptual Framework}}}
The conceptual framework defines the variables and processing steps of the tracking system. The input features represent the independent variables, the amodal query updates and motion predictions represent the intervening variables, and the tracking metrics represent the dependent variables.

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.5cm, auto,
   varbox/.style={draw, rectangle, fill=blue!5, text width=3.8cm, text centered, rounded corners, minimum height=3.5em, font=\scriptsize},
   intbox/.style={draw, rectangle, fill=green!5, text width=4.5cm, text centered, rounded corners, minimum height=3.5em, font=\scriptsize},
   depbox/.style={draw, rectangle, fill=yellow!5, text width=3.8cm, text centered, rounded corners, minimum height=3.5em, font=\scriptsize},
   modbox/.style={draw, rectangle, fill=red!5, text width=4.0cm, text centered, rounded corners, minimum height=3.5em, font=\scriptsize},
   arrow/.style={->, thick, >=stealth}
]
% Nodes
\node [varbox] (iv) {\textbf{INDEPENDENT} \\ \textbf{VARIABLES} \\ \vspace{0.2em} 1. UAV Video Feed \\ 2. Gimbal Pitch/Roll \\ 3. ExG/HSV Color};
\node [intbox, right=of iv, xshift=0.3cm] (mv) {\textbf{INTERVENING PROCESSES} \\ \vspace{0.2em} 1. Occlusion Mask ($\mathcal{M}_{occ}$) \\ 2. Amodal Anchors ($q_i^{(t)}$) \\ 3. CTRV Projections \\ 4. Deformable Attention \\ 5. Herd Cohesion Prior ($P_{prior}$)};
\node [depbox, right=of mv, xshift=0.3cm] (dv) {\textbf{DEPENDENT} \\ \textbf{VARIABLES} \\ \vspace{0.2em} 1. MOTA Score \\ 2. IDF1 \& HOTA \\ 3. ID Switches (IDSW) \\ 4. Recovery Rate};
\node [modbox, below=of mv, yshift=-0.2cm] (mod) {\textbf{MODERATING VARIABLES} \\ \vspace{0.2em} 1. Wind speed \& Gimbal Jitter \\ 2. Foliage Density \\ 3. Baseline Grazing Velocity};

% Connections
\draw [arrow] (iv) -- (mv);
\draw [arrow] (mv) -- (dv);
\draw [arrow] (mod) -- (mv);
\end{tikzpicture}
\caption{Conceptual Framework mapping the relationship between tracking variables and the proposed amodal processes.}
\label{fig:conceptual_framework}
\end{figure}

\subsection*{\underline{\textbf{Study Area and Target Population}}}
% [[START: SECTION: Study Area and Target Population]]
The study is conducted using datasets from Wakiso District, Uganda, specifically targeting cattle herds grazing at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK). The target population consists of local Ankole and Holstein-Friesian crossbreed dairy herds.
% [[END: SECTION: Study Area and Target Population]]

\subsection*{\underline{\textbf{Sample Size and Sampling Strategy}}}
% [[START: SECTION: Sample Size and Sampling Strategy]]
The sample consists of 12 healthy, high-resolution DJI Mavic Pro video sequences (downsampled to 5.0 FPS to reduce spatial redundancy, totaling 17,732 frames). The videos cover various lighting conditions, grazing speeds, and foliage densities.
% [[END: SECTION: Sample Size and Sampling Strategy]]

\subsection*{\underline{\textbf{Data Collection and Preprocessing}}}
% [[START: SECTION: Data Collection and Preprocessing]]
UAV video is collected using a DJI Mavic Pro drone flying at altitudes between 15m and 35m. Video is annotated using a semi-automated Segment Anything 2 (SAM 2) tool, producing ground truth files for both visible and amodal (hidden under trees) cattle positions.
% [[END: SECTION: Data Collection and Preprocessing]]

\subsection*{\underline{\textbf{Vegetation Segmentation Model}}}
% [[START: SECTION: Vegetation Segmentation Model]]
Foliage is segmented from pasture in real-time by computing a Fused foliage mask. We compute the Excess Green Index (ExG):
\begin{equation}
ExG = 2G - R - B
\end{equation}
We combine the binary ExG mask with an HSV green range mask (Hue: 35-85, Saturation: 40-255, Value: 30-255) using a bitwise OR operation. The resulting union mask is processed via morphological opening and closing to establish a stable ground-plane occlusion mask $\mathcal{M}_{occ}$.
% [[END: SECTION: Vegetation Segmentation Model]]

\subsection*{\underline{\textbf{UAV Ground-Plane Homography Projection}}}
% [[START: SECTION: UAV Ground-Plane Homography Projection]]
Let the UAV position at time $t$ be $p_{uav} = (x_{uav}, y_{uav}, z_{uav})^T$. Let the ground plane be defined at $z=0$. Real-time segmentation identifies $K$ vegetation structures $O = \{O_1, \dots, O_K\}$ modeled as 3D convex volumes with boundaries $V_k^{3D}$. Any boundary point $v = (x_v, y_v, z_v)^T \in V_k^{3D}$ is projected onto the ground plane:
\begin{equation}
p_{proj} = p_{uav} + \lambda (v - p_{uav})
\end{equation}
where $\lambda = -z_{uav} / (z_v - z_{uav})$. The global ground-plane occlusion mask $\mathcal{M}_{occ}$ is the union of all projected shadows:
\begin{equation}
\mathcal{M}_{occ} = \bigcup_{k=1}^K S_k
\end{equation}
% [[END: SECTION: UAV Ground-Plane Homography Projection]]

\subsection*{\underline{\textbf{Amodal Anchor Seeding and CTRV Motion Models}}}
% [[START: SECTION: Amodal Anchor Seeding and CTRV Motion Models]]
When an animal track $i$ enters the occlusion mask $\mathcal{M}_{occ}$ at coordinates $(x_i^{t_0}, y_i^{t_0})$, the tracker seeds an Amodal Anchor $A_i$. The anchor's position is updated using a Constant Turn Rate and Velocity (CTRV) motion model. The state vector is:
\begin{equation}
x = \begin{bmatrix} x & y & v & \theta & \omega \end{bmatrix}^T
\end{equation}
where $v$ is velocity, $\theta$ is heading, and $\omega$ is yaw rate. Position updates are blended with a herd-cohesion prior $P_{prior}$ reflecting the average velocity vector of visible herd members $v_{herd}$:
\begin{equation}
P_{prior}(p) \propto \exp\left( -\frac{\|p - (p_i^{(t_0)} + v_{herd}\cdot(t-t_0))\|^2}{2\sigma_{herd}^2} \right)
\end{equation}
% [[END: SECTION: Amodal Anchor Seeding and CTRV Motion Models]]

\subsection*{\underline{\textbf{Spatiotemporal Recurrent Query Routing}}}
% [[START: SECTION: Spatiotemporal Recurrent Query Routing]]
Amodal anchors interact with the temporal memory queue of the tracking network. We implement a Spatiotemporal Deformable Attention layer in PyTorch. For each query $q_i$, offsets $\Delta p_{il}$ and attention weights $A_{il}$ are predicted:
\begin{equation}
\Delta p_{il} = \text{Linear}_{pos}(q_i), \quad A_{il} = \text{Softmax}(\text{Linear}_{attn}(q_i))
\end{equation}
Features are extracted using bilinear grid sampling over a sequence of past feature maps $F_{mem}$:
\begin{equation}
\text{DeformAttn}(q_i, p_i, F) = \sum_{l=1}^L A_{il} \cdot W_l F(p_i + \Delta p_{il})
\end{equation}
where $W_l$ is a learnable projection matrix.
% [[END: SECTION: Spatiotemporal Recurrent Query Routing]]

\subsection*{\underline{\textbf{Hindsight Trajectory RTS Smoothing}}}
% [[START: SECTION: Hindsight Trajectory RTS Smoothing]]
During training, we supervise the model inside the occlusion zone using hindsight trajectory reconstruction. The ground-truth trajectory is back-propagated from the emergence frame $t_0 + \Delta t$ to the entry frame $t_0$ using a bi-directional Rauch-Tung-Striebel (RTS) Kalman smoother. The network is optimized using the Hindsight Temporal Loss $\mathcal{L}_{hind}$, Probabilistic Occupancy Loss $\mathcal{L}_{occ}$, and Bounded Hallucination Loss $\mathcal{L}_{bound}$.
% [[END: SECTION: Hindsight Trajectory RTS Smoothing]]

\subsection*{\underline{\textbf{Data Quality Control}}}
% [[START: SECTION: Data Quality Control]]
Data quality is maintained by:
\begin{itemize}
    \item Restricting anchor updates with an anchor expiry threshold of 150 frames (static anchors adaptively extend to 300 frames).
    \item Resolving redundant tracks via duplicate suppression (IoU > 0.6 matches merge).
    \item Validating re-emergence using Rolling average HSV histogram appearance matching ($70\%$ spatial, $30\%$ appearance).
\end{itemize}
% [[END: SECTION: Data Quality Control]]

\subsection*{\underline{\textbf{Performance Metrics and Evaluation Protocols}}}
% [[START: SECTION: Performance Metrics and Evaluation Protocols]]
We evaluate tracking accuracy against baselines (ByteTrack, OC-SORT, DeepSORT) using:
\begin{itemize}
    \item Multi-Object Tracking Accuracy (MOTA)
    \item Identity F1 Score (IDF1)
    \item Higher Order Tracking Accuracy (HOTA)
    \item Identity Switches (IDSW)
    \item Occlusion Recovery Rate (ORR)
\end{itemize}
% [[END: SECTION: Performance Metrics and Evaluation Protocols]]

\subsection*{\underline{\textbf{Ethical Considerations}}}
% [[START: SECTION: Ethical Considerations]]
Drone flights are operated at altitudes above 15m to prevent herd stampedes and stress caused by acoustic noise. Research clearances are obtained from Makerere University CCIT Higher Degrees Committee and the Uganda National Council for Science and Technology (UNCST).
% [[END: SECTION: Ethical Considerations]]

\subsection*{\underline{\textbf{Environmental Considerations}}}
% [[START: SECTION: Environmental Considerations]]
UAV batteries are recycled responsibly. Computational training is executed on high-efficiency GPU servers to reduce carbon footprint.
% [[END: SECTION: Environmental Considerations]]

\subsection*{\underline{\textbf{Gender Considerations}}}
% [[START: SECTION: Gender Considerations]]
Precision agriculture reduces the physical labor of manual cattle herding, which has historically fallen disproportionately on young men and boys, allowing for more inclusive participation of women in herd management and data analysis.
% [[END: SECTION: Gender Considerations]]

\subsection*{\underline{\textbf{Limitations and Mitigation Strategies}}}
% [[START: SECTION: Limitations and Mitigation Strategies]]
Camera yaw and rapid pitch changes during high-wind situations degrade planar homography. We mitigate this by integrating optical flow camera motion compensation (CMC) using ORB sparse keypoint matching across sequential frames.
% [[END: SECTION: Limitations and Mitigation Strategies]]

\clearpage

% ==========================================
% REFERENCES
% ==========================================
\begin{references}
% [[START: SECTION: References]]
\item Bewley, A., Ge, Z., Ott, L., Ramos, F., \& Upcroft, B. (2016). Simple online and realtime tracking. \textit{Proceedings of the IEEE International Conference on Image Processing (ICIP)}, 3464--3468.
\item Cao, J., Pang, J., Weng, X., Guan, R., \& Shen, Y. (2023). Observation-centric multi-object tracking. \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 16200--16210.
\item Du, Y., Zhao, Y., Song, B., Zhao, Y., \& Wan, J. (2023). StrongSORT: Make DeepSORT great again. \textit{IEEE Transactions on Multimedia}, 25, 8725--8737.
\item Kour, A., \& Singh, H. (2024). Counterfactual reasoning in multi-agent deep reinforcement learning for motion prediction. \textit{Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS)}, 1--9.
\item Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., \& Dai, J. (2022). BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. \textit{Proceedings of the European Conference on Screen Vision (ECCV)}, 1--18.
\item Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., \& Han, S. (2023). BEVFusion: Multi-task multi-sensor fusion with unified bird's-eye view representation. \textit{Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)}, 1--8.
\item Lv, W., Xu, S., Zhao, H., et al. (2024). DETRs beat YOLOs on real-time object detection. \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 16300--16310.
\item Min, C., Zhao, D., Xiao, L., Zhao, J., Xu, X., Zhu, Z., Jin, L., Li, J., Guo, J., Xing, J., Jing, L., Nie, Y., \& Dai, B. (2024). DriveWorld: 4D pre-trained scene understanding via world models for autonomous driving. \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 14000--14010.
\item Ozguroglu, E., Liu, R., Surís, D., Chen, D., Dave, A., Tokmakov, P., \& Vondrick, C. (2024). pix2gestalt: Amodal segmentation by synthesizing wholes. \textit{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 12000--12010.
\item Qiao, Y., Guo, Y., \& He, D. (2023). Cattle body detection based on YOLOv5-ASFF for precision livestock farming. \textit{Computers and Electronics in Agriculture}, 204, 107579.
\item Ravi, N., Girdhar, V., Samuel, S., et al. (2024). Segment Anything in high-resolution video. \textit{arXiv preprint arXiv:2408.00714}.
\item Santamaria, M., Vazquez, J., \& Torres, L. (2023). Computer vision and UAVs in precision livestock farming: A systematic review. \textit{Sensors}, 23(15), 6780.
\item Shao, F., Li, D., \& Wang, L. (2023). Robust multi-animal tracking in aerial videos using target-guided motion models. \textit{Computers and Electronics in Agriculture}, 210, 107920.
\item Sun, P., Cao, J., Jiang, Y., Zhang, R., Xie, E., Yuan, Z., Wang, C., \& Luo, P. (2020). TransTrack: Multiple-object tracking with transformer. \textit{arXiv preprint arXiv:2012.15460}.
\item Xu, B., Wang, W., Falzon, G., et al. (2020). Automated cattle counting using Mask R-CNN in quadcopter vision system. \textit{Computers and Electronics in Agriculture}, 166, 105000.
\item Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, D., ... \& Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. \textit{Proceedings of the European Conference on Computer Vision (ECCV)}, 1--21.
% [[END: SECTION: References]]
\end{references}

\clearpage

% ==========================================
% APPENDICES
% ==========================================
\section*{\centering \textbf{APPENDICES}}

\subsection*{\underline{\textbf{Appendix A: Itemized Budget}}}
The table below outlines the itemized budget for the implementation of the research proposal, in Ugandan Shillings (UGX). The budget includes a 15\% Makerere SGS/institutional administrative overhead fee in accordance with university research policies.

\begin{table}[h]
\centering
\caption{Itemized Research Budget (UGX)}
\label{tab:budget}
\begin{tabular}{llr}
    \toprule
    \textbf{Category} & \textbf{Item Description} & \textbf{Cost (UGX)} \\
    \midrule
% [[START: TABLE: Budget]]
    \textbf{Equipment} & Deep learning workstation GPU leasing/upgrade & 4,500,000 \\
    \textbf{Equipment} & UAV battery replacement & 1,800,000 \\
    \textbf{Equipment} & High-speed SD cards (256GB, 2 units) & 400,000 \\
    \textbf{Stationery} & Printing paper, notebooks, field logs & 300,000 \\
    \textbf{Materials} & Reflective ground calibration markers & 500,000 \\
    \textbf{Travel} & 12 data collection field trips to MUARIK (fuel) & 1,200,000 \\
    \textbf{Subsistence} & Field allowance for principal researcher & 1,200,000 \\
    \textbf{Research Assistance} & 2 Field handlers for safety \& animal coordination & 2,400,000 \\
    \textbf{Services} & Secretarial printing, copying, binding & 500,000 \\
    \textbf{Dissemination} & Open-access journal page charges & 3,500,000 \\
    \textbf{Dissemination} & National agricultural conference registration & 1,500,000 \\
    \textbf{Overhead} & Institutional Administrative Fee (15\% overhead) & 2,565,000 \\
    \textbf{Total} & & \textbf{19,665,000} \\
% [[END: TABLE: Budget]]
    \bottomrule
\end{tabular}
\end{table}

\clearpage

\subsection*{\underline{\textbf{Appendix B: Work Plan and Gantt Chart}}}
The schedule of activities spans from September 2025 to July 2026, divided into academic quarters.

\begin{table}[h]
\centering
\caption{Research Work Plan and Activity Timeline}
\label{tab:workplan}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc}
    \toprule
    \textbf{Activity / Phase} & \textbf{Q1 (Sep-Nov)} & \textbf{Q2 (Dec-Feb)} & \textbf{Q3 (Mar-May)} & \textbf{Q4 (Jun-Jul)} \\
    \midrule
% [[START: TABLE: WorkPlan]]
    Literature Review \& Proposal Defense & X &  &  &  \\
    UAV Flight Approvals \& Data Collection & X & X &  &  \\
    SAM 2 Semi-Automated Annotation Tool &  & X &  &  \\
    Foliage Segmentation \& Projection Model &  & X & X &  \\
    CTRV \& Amodal Seeding Implementation &  &  & X &  \\
    Attention query routing \& Loss tuning &  &  & X & X \\
    Quantitative benchmarking \& Analysis &  &  &  & X \\
    Thesis report writing \& Submission &  &  &  & X \\
% [[END: TABLE: WorkPlan]]
    \bottomrule
\end{tabular}
}
\end{table}

\subsection*{\underline{\textbf{Appendix C: Research Instruments and Explanatory Notes}}}
% [[START: SECTION: Appendix C: Research Instruments and Explanatory Notes]]
The following research instruments are utilized during execution:
\begin{enumerate}
    \item \textbf{UAV Hardware:} DJI Mavic Pro drone equipped with a 1/2.3'' CMOS camera recording 4K video at 30 FPS, and telemetry logging (GPS, altitude, pitch, roll, yaw).
    \item \textbf{Computing Hardware:} Intel Xeon workstation with 64GB RAM and an NVIDIA RTX 4090 GPU (24GB VRAM) for model training and simulation.
    \item \textbf{Software Stack:} PyTorch 2.2, OpenCV 4.9, ONNX Runtime, Python 3.10.
    \item \textbf{Study Site Geolocation:} Pasture fields at MUARIK, Wakiso, Uganda. Geolocation coordinates: Latitude 0.4508$^{\circ}$ N, Longitude 32.6144$^{\circ}$ E.
\end{enumerate}
% [[END: SECTION: Appendix C: Research Instruments and Explanatory Notes]]

\end{document}
"""


def main():
    print("--------------------------------------------------")
    print("Makerere University Research Proposal Builder")
    print("--------------------------------------------------")

    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "Research_Proposal.tex")
    pdf_path = os.path.join(script_dir, "Research_Proposal.pdf")

    # 2. WRITE LATEX SOURCE FILE
    print(f"Writing LaTeX source to: {tex_path}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    # 3. COMPILE LATEX TO PDF VIA ONLINE API
    print("Compiling LaTeX to PDF via YtoTech LaTeX-on-HTTP API...")
    url = "https://latex.ytotech.com/builds/sync"
    payload = {
        "compiler": "pdflatex",
        "resources": [
            {
                "main": True,
                "content": latex_text
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 201:
            print(f"Compilation successful! Saving PDF to: {pdf_path}")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            
            # Verify PDF page count using pypdf
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"Successfully verified PDF. Page count: {num_pages}")
            print(f"Proposal written to {pdf_path}")
        else:
            print(f"Compilation failed with status code: {response.status_code}")
            print("Response error details:")
            print(response.text[:1000])
    except Exception as e:
        print("An error occurred during compilation:")
        print(e)


if __name__ == "__main__":
    main()
