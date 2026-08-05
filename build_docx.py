import os
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def add_level_1_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.5
    return p

def add_level_3_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True
    run.underline = True
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    return p

def add_level_4_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True
    run.underline = True
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    return p

def add_paragraph(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    return p

def add_marker(doc, text):
    # Hidden marker for sync script (styled in light grey)
    p = doc.add_paragraph()
    run = p.add_run(f"[[{text}]]")
    run.font.name = 'Courier New'
    run.font.size = Pt(9.5)
    run.font.color.rgb = docx.shared.RGBColor(160, 160, 160)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    return p

def main():
    doc = docx.Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.page_width = Inches(8.27) # A4 width
        section.page_height = Inches(11.69) # A4 height

    # ==========================================
    # TITLE PAGE
    # ==========================================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    
    run = p.add_run("MAKERERE UNIVERSITY\n")
    run.bold = True
    run.font.size = Pt(14)
    
    run = p.add_run("DIRECTORATE OF RESEARCH AND GRADUATE TRAINING\n\n\n\n")
    run.bold = True
    run.font.size = Pt(11)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("BEYOND THE LINE OF SIGHT: COUNTERFACTUAL AMODAL ANCHORS FOR OCCLUSION-AWARE MULTI-ANIMAL TRACKING IN UAV-BASED PRECISION LIVESTOCK MONITORING\n\n\n\n")
    run_title.bold = True
    run_title.font.size = Pt(13)
    
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_author = p_author.add_run("BY\n\nTUMUSIIME DEUS\n")
    run_author.bold = True
    run_author.font.size = Pt(12)
    
    run_sub = p_author.add_run("B.Sc. Software Engineering (Makerere University)\n")
    run_sub.font.size = Pt(11)
    
    run_reg = p_author.add_run("Reg No: 2025/HD05/26375U\n\n\n\n")
    run_reg.bold = True
    run_reg.font.size = Pt(12)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("A RESEARCH PROPOSAL SUBMITTED TO THE DEPARTMENT OF COMPUTER SCIENCE, COLLEGE OF COMPUTING AND INFORMATICS TECHNOLOGY IN PARTIAL FULFILMENT OF THE AWARD OF THE DEGREE OF MASTER OF SCIENCE IN COMPUTER SCIENCE OF MAKERERE UNIVERSITY\n\n\n\n")
    run_sub.bold = True
    run_sub.font.size = Pt(11)
    
    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_date = p_date.add_run("JULY 2026")
    run_date.bold = True
    run_date.font.size = Pt(12)
    
    doc.add_page_break()

    # ==========================================
    # DECLARATION
    # ==========================================
    add_marker(doc, "SECTION: Declaration")
    add_level_1_heading(doc, "Declaration")
    add_paragraph(doc, "I, Tumusiime Deus, declare that this research proposal is my original work and has not been submitted for any other degree award to any other university before.")
    add_paragraph(doc, "Signature: __________________________          Date: __________________________")
    
    add_paragraph(doc, "Approval by Supervisors\nThis research proposal has been submitted for evaluation with the approval of the following supervisors:")
    
    add_paragraph(doc, "Dr. George Nasinyama\nDepartment of Computer Science\nCollege of Computing and Informatics Technology, Makerere University\nSignature: __________________________          Date: __________________________")
    add_paragraph(doc, "Prof. Celestino Obua\nDirectorate of Research and Graduate Training, Makerere University\nSignature: __________________________          Date: __________________________")
    doc.add_page_break()

    # ==========================================
    # ABSTRACT
    # ==========================================
    add_marker(doc, "SECTION: Abstract")
    add_level_1_heading(doc, "Abstract")
    add_paragraph(doc, "Precision Livestock Farming (PLF) increasingly relies on unmanned aerial vehicles (UAVs) equipped with RGB and thermal cameras to automate cattle counting, grazing trajectory monitoring, and health assessment. While deep learning models have improved object detection and tracking in open pastures, prolonged visual occlusion caused by trees, shrubs, terrain folds, and animal clustering remains a major challenge. Standard tracking-by-detection systems treat occlusion as a passive state of missing observations, relying on simple motion models or appearance re-identification. Consequently, trajectory fragmentation, identity switching, and inaccurate behavioral modeling are common in extensive grazing systems.")
    add_paragraph(doc, "This research proposes a novel perception framework termed Counterfactual Amodal Anchors (CAA) for occlusion-aware multi-animal tracking. Instead of terminating tracks or allowing occluded areas to collapse into featureless voids, our framework maps the geometric projection of occlusion shadows onto a local ground plane and seeds them with active query vectors. These amodal anchors continuously update their state using temporal context, herd-level social dynamics, and terrain priors.")
    add_paragraph(doc, "We formulate mathematical projection models, spatiotemporal deformable attention query updates, and hindsight loss functions to supervise the network during training. The framework will be evaluated on drone datasets collected from Ugandan pastures, specifically at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK). By reframing occlusion as a structured probabilistic estimation problem, this work aims to establish a robust foundation for automated cattle tracking, lameness screening, and pasture management in Sub-Saharan Africa.")
    doc.add_page_break()

    # ==========================================
    # CHAPTER ONE: INTRODUCTION
    # ==========================================
    add_level_1_heading(doc, "CHAPTER ONE: INTRODUCTION")
    
    add_marker(doc, "SECTION: Background to the Study")
    add_level_3_heading(doc, "Background to the Study")
    add_paragraph(doc, "Livestock agriculture is a foundational pillar of food security, rural livelihoods, and macroeconomic stability across Sub-Saharan Africa, and particularly in Uganda. According to the Uganda Bureau of Statistics, livestock production contributes significantly to the agricultural gross domestic product and employs millions of rural households. Effective pasture management, grazing behavior analysis, and early disease detection require continuous, accurate monitoring of animal herds. Traditional observation protocols remain overwhelmingly manual, making them labor-intensive, subjective, and difficult to scale across extensive ranching environments.")
    add_paragraph(doc, "Recently, the integration of Unmanned Aerial Vehicles (UAVs) equipped with high-resolution RGB and Long-Wave Infrared (LWIR) thermal cameras has emerged as a promising tool for automated monitoring. UAVs provide a top-down perspective, allowing detection models like Real-Time DEtection Transformer (RT-DETR) and Segment Anything 2 (SAM 2) to identify individual animals, and tracking frameworks like ByteTrack or Observation-Centric SORT (OC-SORT) to construct motion trajectories.")
    add_paragraph(doc, "Despite these technological advancements, visual occlusion remains the single most significant bottleneck preventing the deployment of fully automated systems. In extensive grazing pastures, cattle frequently move behind obstacles such as Acacia canopies, dense bushes, water troughs, shading infrastructure, and behind one another during herd aggregation.")
    add_paragraph(doc, "When an animal becomes occluded, current tracking-by-detection systems behave passively. They treat the occluded space as a visual null zone, immediately terminating the animal's trajectory or relying on linear constant-velocity Kalman filters to extrapolate its position. Because animals change direction and speed when grazing, linear extrapolation quickly diverges from the true path. Once the animal re-emerges, the tracking system is forced to re-initialize its identity, which frequently results in track fragmentation or identity switching (assigning the ID of one animal to another).")
    
    add_marker(doc, "SECTION: Statement of the Problem")
    add_level_3_heading(doc, "Statement of the Problem")
    add_paragraph(doc, "Existing multi-animal tracking systems are fundamentally reactive, relying on immediate visual verification and falling back to passive prediction during extended occlusions. This operational passivity induces a significant perception lag and degrades downstream behavioral analytics. For example, if a monitoring system is trying to measure the total daily grazing distance or identify signs of lameness (which requires precise step-by-step trajectory analysis), a single identity switch or fragmented track can invalidate hours of data collection.")
    add_paragraph(doc, "This research proposes a conceptual shift: from reactive observation-based tracking to proactive counterfactual reasoning. We present the position that occluded areas cast by vegetation and structures must be modeled as dynamic probabilistic occupancy spaces populated by explicit, active query vectors termed Counterfactual Amodal Anchors (CAA). Rather than deleting the animal's track, the network maps the spatial boundaries of the occlusion shadows and initializes virtual anchors that actively query temporal features, local visual context, and herd cohesion priors.")
    add_paragraph(doc, "This research aligns with Uganda's National Development Plan (NDP III/IV), which identifies agricultural modernization, digitization, and value addition as key drivers of economic growth. Furthermore, it supports the African Union (AU) Agenda 2063 (Aspiration 1: A prosperous Africa based on inclusive growth and sustainable development) and the United Nations Sustainable Development Goals (SDG 2: Zero Hunger, SDG 9: Industry, Innovation, and Infrastructure, and SDG 15: Life on Land) by introducing intelligent, automated technology to optimize food production and animal welfare.")
    
    add_level_3_heading(doc, "Objectives of the Study")
    
    add_marker(doc, "SECTION: General Objective")
    add_level_4_heading(doc, "General Objective")
    add_paragraph(doc, "The general objective of this research is to develop a robust, occlusion-aware multi-animal tracking framework using Counterfactual Amodal Anchors (CAA) to maintain identity continuity and trajectory reconstruction under vegetation-induced occlusions in UAV-based precision livestock monitoring.")
    
    add_marker(doc, "SECTION: Specific Objectives")
    add_level_4_heading(doc, "Specific Objectives")
    add_paragraph(doc, "To achieve this general objective, the study will address the following specific objectives:")
    add_paragraph(doc, "1. To design and implement a ground-plane projection homography model to map UAV camera views and identify vegetation-induced occlusion masks (M_occ).\n"
                    "2. To formulate and integrate an active amodal anchor seeding mechanism using constant turn rate and velocity (CTRV) motion models and herd-cohesion priors to represent hidden targets.\n"
                    "3. To develop a spatiotemporal query routing network using deformable attention to query historical frames and update anchor states.\n"
                    "4. To evaluate the performance of the proposed Counterfactual Amodal Anchor (CAA) framework against traditional tracking baselines (ByteTrack, OC-SORT, DeepSORT) using standard Multi-Object Tracking metrics.")
    
    add_marker(doc, "SECTION: Research Questions")
    add_level_3_heading(doc, "Research Questions")
    add_paragraph(doc, "This study seeks to answer the following three core research questions:\n"
                    "- RQ1: Can Counterfactual Amodal Anchors (CAA) reduce identity switches (IDSW) under prolonged vegetation-induced occlusion in UAV feeds?\n"
                    "- RQ2: Can CAA improve trajectory continuity metrics, specifically Higher Order Tracking Accuracy (HOTA) and Identity F1 Score (IDF1), compared to baseline trackers (ByteTrack, OC-SORT)?\n"
                    "- RQ3: Does the integration of thermal (LWIR) imagery significantly improve anchor update accuracy in dense vegetation and low-contrast illumination compared to RGB-only tracking?")
    
    add_marker(doc, "SECTION: Research Hypotheses")
    add_level_3_heading(doc, "Research Hypotheses")
    add_paragraph(doc, "Based on the theoretical design of our proactive amodal tracking framework, we formulate the following hypotheses:\n"
                    "- H1: Seeding amodal anchors inside projected vegetation masks will significantly reduce identity switches (IDSW) and track fragmentations compared to baseline extrapolation during prolonged occlusions.\n"
                    "- H2: The integration of herd-cohesion social priors and spatiotemporal attention queries yields significantly higher tracking accuracy (MOTA, IDF1, HOTA) under dense foliage than standard constant-velocity motion models.")
    
    add_marker(doc, "SECTION: Significance of the Study")
    add_level_3_heading(doc, "Significance of the Study")
    add_paragraph(doc, "This study contributes to both the theory and application of computer vision in precision agriculture. Academically, it introduces the concept of amodal perception—historically applied in autonomous driving Bird's-Eye View (BEV) networks—to agricultural monitoring, creating new methodologies for tracking hidden agents.")
    add_paragraph(doc, "Practically, it provides Ugandan ranchers and veterinary officers with a reliable, automated tool to screen for lameness, monitor grazing efficiency, and count herds without labor-intensive manual observation. This technological advancement supports agricultural digitization, enhances animal welfare, and improves the profitability of extensive livestock systems.")
    
    add_marker(doc, "SECTION: Justification of the Study")
    add_level_3_heading(doc, "Justification of the Study")
    add_paragraph(doc, "Without a dedicated mechanism to handle visual occlusion, automated cattle tracking remains unviable for real-world deployment. Standard systems immediately lose animal tracks under trees, requiring manual identity corrections and causing severe data gaps. Manual tracking of thousands of animals on extensive ranches is impossible.")
    add_paragraph(doc, "This research is justified because it solves the occlusion bottleneck mathematically and computationally, ensuring that UAV-based monitoring can operate autonomously and reliably under realistic, vegetation-heavy grazing conditions.")
    
    add_level_3_heading(doc, "Scope of the Study")
    
    add_marker(doc, "SECTION: Scope of the Study - Content Scope")
    add_level_4_heading(doc, "Content Scope")
    add_paragraph(doc, "The research focuses on the development of ground-plane projection homographies, foliage segmentation, amodal query seeding, spatiotemporal deformable attention query routing, and re-emergence ReID matching. It does not cover drone control automation or path planning.")
    
    add_marker(doc, "SECTION: Scope of the Study - Geographical Scope")
    add_level_4_heading(doc, "Geographical Scope")
    add_paragraph(doc, "The drone video dataset is collected from active pastures at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK), located in Wakiso District, Uganda.")
    
    add_marker(doc, "SECTION: Scope of the Study - Time Scope")
    add_level_4_heading(doc, "Time Scope")
    add_paragraph(doc, "The research will be conducted over a 12-month period, corresponding to the 2025/2026 academic year.")
    
    add_marker(doc, "SECTION: Scope of the Study - Theoretical Scope")
    add_level_4_heading(doc, "Theoretical Scope")
    add_paragraph(doc, "The study is situated within the domains of computer vision, deep learning, multi-object tracking, amodal perception, and Bayesian kinematic estimation.")
    
    doc.add_page_break()

    # ==========================================
    # CHAPTER TWO: LITERATURE REVIEW
    # ==========================================
    add_level_1_heading(doc, "CHAPTER TWO: LITERATURE REVIEW")
    
    add_marker(doc, "SECTION: Literature Review - Introduction")
    add_level_3_heading(doc, "Introduction")
    add_paragraph(doc, "This chapter reviews literature related to automated animal tracking, Multi-Object Tracking (MOT) paradigms, and amodal perception models. It highlights the state-of-the-art, discusses the limitations of existing frameworks, and establishes the research gap that the proposed Counterfactual Amodal Anchor framework aims to address.")
    
    add_marker(doc, "SECTION: PLF and UAV Monitoring")
    add_level_3_heading(doc, "Precision Livestock Farming and UAV Monitoring")
    add_paragraph(doc, "Precision Livestock Farming (PLF) uses technology to monitor livestock automatically. Recent research has heavily utilized Unmanned Aerial Vehicles (UAVs) to count and monitor cattle, sheep, and horses. UAVs provide a top-down view that covers wide pasture areas quickly. Xu et al. (2020) demonstrated the use of Mask R-CNN for automated cattle counting in quadcopter vision systems. Similarly, Qiao et al. (2023) developed a cattle body detection framework using adaptively fused features to locate animals in complex grazing backgrounds. Nonetheless, these frameworks focus primarily on detection and fail to maintain track continuity when animals move under dense canopies.")
    
    add_marker(doc, "SECTION: Multi-Object Tracking Frameworks")
    add_level_3_heading(doc, "Multi-Object Tracking (MOT) Frameworks")
    add_paragraph(doc, "The Multi-Object Tracking domain is dominated by the tracking-by-detection paradigm. Classic algorithms like Simple Online and Realtime Tracking (SORT) by Bewley et al. (2016) and DeepSORT by Wojke et al. (2017) utilize Kalman filters to predict linear motion and associate detections using Hungarian matching. Recent SOTA trackers like ByteTrack (Zhang et al., 2022) and OC-SORT (Cao et al., 2023) focus on recovering low-confidence detections and mitigating linear motion assumptions during short-term occlusions. However, all these methods remain fundamentally reactive, relying on immediate visual verification. During extended occlusions under tree canopies, the error covariance of the Kalman filter grows quadratically, leading to track termination or identity switches when the animal emerges.")
    
    add_marker(doc, "SECTION: Amodal Perception and BEV Networks")
    add_level_3_heading(doc, "Amodal Perception and Bird's-Eye View (BEV) Networks")
    add_paragraph(doc, "Amodal perception refers to the cognitive ability to perceive the entirety of a physical structure when only portions of it are visible. In computer vision, amodal segmentation frameworks like pix2gestalt (Ozguroglu et al., 2024) predict hidden boundaries of objects. In autonomous driving, Bird's-Eye View (BEV) networks like BEVFormer (Li et al., 2022) and BEVFusion (Liu et al., 2023) map perspective camera features into 3D voxel spaces to represent occluded pedestrians and vehicles.")
    add_paragraph(doc, "Adapting BEV amodal reasoning to livestock monitoring is a promising direction. By treating occluded vegetation shadows not as visual null zones but as probabilistic occupancy regions, we can maintain track continuity. The problem of active, dynamic tracking of multiple mobile agents within large, vegetation-induced geometric shadows remains unaddressed in precision agriculture reviews (Santamaria et al., 2023).")
    
    add_marker(doc, "SECTION: Research Gap Analysis")
    add_level_3_heading(doc, "Research Gap Analysis")
    add_paragraph(doc, "Existing literature reveals three main gaps:\n"
                    "1. Lack of explicit ground-plane mapping of vegetation-induced occlusion masks from UAV perspective feeds.\n"
                    "2. Absence of active amodal anchor queries within mapped occlusion regions to represent hidden animals.\n"
                    "3. Failure to integrate herd-cohesion behavior priors into state prediction models to constrain search windows.\n"
                    "Our proposed Counterfactual Amodal Anchor framework addresses these gaps directly.")
    
    doc.add_page_break()

    # ==========================================
    # CHAPTER THREE: METHODOLOGY
    # ==========================================
    add_level_1_heading(doc, "CHAPTER THREE: METHODOLOGY")
    
    add_marker(doc, "SECTION: Methodology - Research Design and Approach")
    add_level_3_heading(doc, "Research Design and Approach")
    add_paragraph(doc, "This study adopts a quantitative, quasi-experimental research design. We build an algorithmic tracking pipeline, deploy it on simulated and real-world UAV video datasets, and evaluate performance changes quantitatively using established MOT metrics. The approach involves developing modular components in PyTorch and OpenCV, and benchmarking them on a workstation.")
    
    # Note: Conceptual framework diagram can't easily be drawn as TikZ in Word, 
    # but we can place a text note representing it
    add_paragraph(doc, "[Conceptual Framework Variables Flowchart represented in LaTeX PDF]")
    
    add_marker(doc, "SECTION: Study Area and Target Population")
    add_level_3_heading(doc, "Study Area and Target Population")
    add_paragraph(doc, "The study is conducted using datasets from Wakiso District, Uganda, specifically targeting cattle herds grazing at the Makerere University Agricultural Research Institute Kabanyolo (MUARIK). The target population consists of local Ankole and Holstein-Friesian crossbreed dairy herds.")
    
    add_marker(doc, "SECTION: Sample Size and Sampling Strategy")
    add_level_3_heading(doc, "Sample Size and Sampling Strategy")
    add_paragraph(doc, "The sample consists of 12 healthy, high-resolution DJI Mavic Pro video sequences (downsampled to 5.0 FPS to reduce spatial redundancy, totaling 17,732 frames). The videos cover various lighting conditions, grazing speeds, and foliage densities.")
    
    add_marker(doc, "SECTION: Data Collection and Preprocessing")
    add_level_3_heading(doc, "Data Collection and Preprocessing")
    add_paragraph(doc, "UAV video is collected using a DJI Mavic Pro drone flying at altitudes between 15m and 35m. Video is annotated using a semi-automated Segment Anything 2 (SAM 2) tool, producing ground truth files for both visible and amodal (hidden under trees) cattle positions.")
    
    add_marker(doc, "SECTION: Vegetation Segmentation Model")
    add_level_3_heading(doc, "Vegetation Segmentation Model")
    add_paragraph(doc, "Foliage is segmented from pasture in real-time by computing a Fused foliage mask. We compute the Excess Green Index (ExG) as: ExG = 2G - R - B. We combine the binary ExG mask with an HSV green range mask (Hue: 35-85, Saturation: 40-255, Value: 30-255) using a bitwise OR operation. The resulting union mask is processed via morphological opening and closing to establish a stable ground-plane occlusion mask M_occ.")
    
    add_marker(doc, "SECTION: UAV Ground-Plane Homography Projection")
    add_level_3_heading(doc, "UAV Ground-Plane Homography Projection")
    add_paragraph(doc, "Let the UAV position at time t be p_uav = (x_uav, y_uav, z_uav)^T. Let the ground plane be defined at z=0. Real-time segmentation identifies K vegetation structures O = {O1, ..., OK} modeled as 3D convex volumes with boundaries V_k^{3D}. Any boundary point v = (x_v, y_v, z_v)^T in V_k^{3D} is projected onto the ground plane: p_proj = p_uav + lambda * (v - p_uav) where lambda = -z_uav / (z_v - z_uav). The global ground-plane occlusion mask M_occ is the union of all projected shadows.")
    
    add_marker(doc, "SECTION: Amodal Anchor Seeding and CTRV Motion Models")
    add_level_3_heading(doc, "Amodal Anchor Seeding and CTRV Motion Models")
    add_paragraph(doc, "When an animal track i enters the occlusion mask M_occ at coordinates (x_i^{t_0}, y_i^{t_0}), the tracker seeds an Amodal Anchor A_i. The anchor's position is updated using a Constant Turn Rate and Velocity (CTRV) motion model. The state vector is: [x, y, v, theta, omega]^T where v is velocity, theta is heading, and omega is yaw rate. Position updates are blended with a herd-cohesion prior P_prior reflecting the average velocity vector of visible herd members v_herd.")
    
    add_marker(doc, "SECTION: Spatiotemporal Recurrent Query Routing")
    add_level_3_heading(doc, "Spatiotemporal Recurrent Query Routing")
    add_paragraph(doc, "Amodal anchors interact with the temporal memory queue of the tracking network. We implement a Spatiotemporal Deformable Attention layer in PyTorch. For each query q_i, offsets delta_p_il and attention weights A_il are predicted using linear projections. Features are extracted using bilinear grid sampling over a sequence of past feature maps F_mem.")
    
    add_marker(doc, "SECTION: Hindsight Trajectory RTS Smoothing")
    add_level_3_heading(doc, "Hindsight Trajectory RTS Smoothing")
    add_paragraph(doc, "During training, we supervise the model inside the occlusion zone using hindsight trajectory reconstruction. The ground-truth trajectory is back-propagated from the emergence frame t_0 + delta_t to the entry frame t_0 using a bi-directional Rauch-Tung-Striebel (RTS) Kalman smoother. The network is optimized using the Hindsight Temporal Loss, Probabilistic Occupancy Loss, and Bounded Hallucination Loss.")
    
    add_marker(doc, "SECTION: Data Quality Control")
    add_level_3_heading(doc, "Data Quality Control")
    add_paragraph(doc, "Data quality is maintained by restricting anchor updates with an anchor expiry threshold of 150 frames (static anchors adaptively extend to 300 frames), resolving redundant tracks via duplicate suppression (IoU > 0.6 matches merge), and validating re-emergence using rolling average HSV histogram appearance matching (70% spatial, 30% appearance).")
    
    add_marker(doc, "SECTION: Performance Metrics and Evaluation Protocols")
    add_level_3_heading(doc, "Performance Metrics and Evaluation Protocols")
    add_paragraph(doc, "We evaluate tracking accuracy against baselines (ByteTrack, OC-SORT, DeepSORT) using: Multi-Object Tracking Accuracy (MOTA), Identity F1 Score (IDF1), Higher Order Tracking Accuracy (HOTA), Identity Switches (IDSW), and Occlusion Recovery Rate (ORR).")
    
    add_marker(doc, "SECTION: Ethical Considerations")
    add_level_3_heading(doc, "Ethical Considerations")
    add_paragraph(doc, "Drone flights are operated at altitudes above 15m to prevent herd stampedes and stress caused by acoustic noise. Research clearances are obtained from Makerere University CCIT Higher Degrees Committee and the Uganda National Council for Science and Technology (UNCST).")
    
    add_marker(doc, "SECTION: Environmental Considerations")
    add_level_3_heading(doc, "Environmental Considerations")
    add_paragraph(doc, "UAV batteries are recycled responsibly. Computational training is executed on high-efficiency GPU servers to reduce carbon footprint.")
    
    add_marker(doc, "SECTION: Gender Considerations")
    add_level_3_heading(doc, "Gender Considerations")
    add_paragraph(doc, "Precision agriculture reduces the physical labor of manual cattle herding, which has historically fallen disproportionately on young men and boys, allowing for more inclusive participation of women in herd management and data analysis.")
    
    add_marker(doc, "SECTION: Limitations and Mitigation Strategies")
    add_level_3_heading(doc, "Limitations and Mitigation Strategies")
    add_paragraph(doc, "Camera yaw and rapid pitch changes during high-wind situations degrade planar homography. We mitigate this by integrating optical flow camera motion compensation (CMC) using ORB sparse keypoint matching across sequential frames.")
    
    doc.add_page_break()

    # ==========================================
    # REFERENCES
    # ==========================================
    add_marker(doc, "SECTION: References")
    add_level_1_heading(doc, "REFERENCES")
    refs = [
        "Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. Proceedings of the IEEE International Conference on Image Processing (ICIP), 3464-3468.",
        "Cao, J., Pang, J., Weng, X., Guan, R., & Shen, Y. (2023). Observation-centric multi-object tracking. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16200-16210.",
        "Du, Y., Zhao, Y., Song, B., Zhao, Y., & Wan, J. (2023). StrongSORT: Make DeepSORT great again. IEEE Transactions on Multimedia, 25, 8725-8737.",
        "Kour, A., & Singh, H. (2024). Counterfactual reasoning in multi-agent deep reinforcement learning for motion prediction. Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 1-9.",
        "Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., & Dai, J. (2022). BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. Proceedings of the European Conference on Screen Vision (ECCV), 1-18.",
        "Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., & Han, S. (2023). BEVFusion: Multi-task multi-sensor fusion with unified bird's-eye view representation. Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 1-8.",
        "Lv, W., Xu, S., Zhao, H., et al. (2024). DETRs beat YOLOs on real-time object detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16300-16310.",
        "Min, C., Zhao, D., Xiao, L., Zhao, J., Xu, X., Zhu, Z., Jin, L., Li, J., Guo, J., Xing, J., Jing, L., Nie, Y., & Dai, B. (2024). DriveWorld: 4D pre-trained scene understanding via world models for autonomous driving. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14000-14010.",
        "Ozguroglu, E., Liu, R., Suris, D., Chen, D., Dave, A., Tokmakov, P., & Vondrick, C. (2024). pix2gestalt: Amodal segmentation by synthesizing wholes. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 12000-12010.",
        "Qiao, Y., Guo, Y., & He, D. (2023). Cattle body detection based on YOLOv5-ASFF for precision livestock farming. Computers and Electronics in Agriculture, 204, 107579.",
        "Ravi, N., Girdhar, V., Samuel, S., et al. (2024). Segment Anything in high-resolution video. arXiv preprint arXiv:2408.00714.",
        "Santamaria, M., Vazquez, J., & Torres, L. (2023). Computer vision and UAVs in precision livestock farming: A systematic review. Sensors, 23(15), 6780.",
        "Shao, F., Li, D., & Wang, L. (2023). Robust multi-animal tracking in aerial videos using target-guided motion models. Computers and Electronics in Agriculture, 210, 107920.",
        "Sun, P., Cao, J., Jiang, Y., Zhang, R., Xie, E., Yuan, Z., Wang, C., & Luo, P. (2020). TransTrack: Multiple-object tracking with transformer. arXiv preprint arXiv:2012.15460.",
        "Xu, B., Wang, W., Falzon, G., et al. (2020). Automated cattle counting using Mask R-CNN in quadcopter vision system. Computers and Electronics in Agriculture, 166, 105000.",
        "Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, D., ... & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. Proceedings of the European Conference on Computer Vision (ECCV), 1-21."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(r)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        
    doc.add_page_break()

    # ==========================================
    # APPENDICES
    # ==========================================
    add_level_1_heading(doc, "APPENDICES")
    
    add_marker(doc, "SECTION: Appendix A: Itemized Budget")
    add_level_3_heading(doc, "Appendix A: Itemized Budget")
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Shading Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Category'
    hdr_cells[1].text = 'Item Description'
    hdr_cells[2].text = 'Cost (UGX)'
    
    budget_items = [
        ('Equipment', 'Deep learning workstation GPU leasing/upgrade', '4,500,000'),
        ('Equipment', 'UAV battery replacement', '1,800,000'),
        ('Equipment', 'High-speed SD cards (256GB, 2 units)', '400,000'),
        ('Stationery', 'Printing paper, notebooks, field logs', '300,000'),
        ('Materials', 'Reflective ground calibration markers', '500,000'),
        ('Travel', '12 data collection field trips to MUARIK (fuel)', '1,200,000'),
        ('Subsistence', 'Field allowance for principal researcher', '1,200,000'),
        ('Research Assistance', '2 Field handlers for safety & animal coordination', '2,400,000'),
        ('Services', 'Secretarial printing, copying, binding', '500,000'),
        ('Dissemination', 'Open-access journal page charges', '3,500,000'),
        ('Dissemination', 'National agricultural conference registration', '1,500,000'),
        ('Overhead', 'Institutional Administrative Fee (15% overhead)', '2,565,000'),
        ('Total', '', '19,665,000')
    ]
    for cat, desc, cost in budget_items:
        row_cells = table.add_row().cells
        row_cells[0].text = cat
        row_cells[1].text = desc
        row_cells[2].text = cost
        
    doc.add_paragraph() # space

    add_marker(doc, "SECTION: Appendix B: Work Plan and Gantt Chart")
    add_level_3_heading(doc, "Appendix B: Work Plan and Gantt Chart")
    
    table_wp = doc.add_table(rows=1, cols=5)
    table_wp.style = 'Light Shading Accent 1'
    hdr_wp = table_wp.rows[0].cells
    hdr_wp[0].text = 'Activity / Phase'
    hdr_wp[1].text = 'Q1 (Sep-Nov)'
    hdr_wp[2].text = 'Q2 (Dec-Feb)'
    hdr_wp[3].text = 'Q3 (Mar-May)'
    hdr_wp[4].text = 'Q4 (Jun-Jul)'
    
    wp_items = [
        ('Literature Review & Proposal Defense', 'X', '', '', ''),
        ('UAV Flight Approvals & Data Collection', 'X', 'X', '', ''),
        ('SAM 2 Semi-Automated Annotation Tool', '', 'X', '', ''),
        ('Foliage Segmentation & Projection Model', '', 'X', 'X', ''),
        ('CTRV & Amodal Seeding Implementation', '', '', 'X', ''),
        ('Attention query routing & Loss tuning', '', '', 'X', 'X'),
        ('Quantitative benchmarking & Analysis', '', '', '', 'X'),
        ('Thesis report writing & Submission', '', '', '', 'X')
    ]
    for row in wp_items:
        row_cells = table_wp.add_row().cells
        for idx, val in enumerate(row):
            row_cells[idx].text = val
            
    doc.add_paragraph() # space

    add_marker(doc, "SECTION: Appendix C: Research Instruments and Explanatory Notes")
    add_level_3_heading(doc, "Appendix C: Research Instruments and Explanatory Notes")
    add_paragraph(doc, "The following research instruments are utilized during execution:\n"
                    "1. UAV Hardware: DJI Mavic Pro drone equipped with a 1/2.3'' CMOS camera recording 4K video at 30 FPS, and telemetry logging (GPS, altitude, pitch, roll, yaw).\n"
                    "2. Computing Hardware: Intel Xeon workstation with 64GB RAM and an NVIDIA RTX 4090 GPU (24GB VRAM) for model training and simulation.\n"
                    "3. Software Stack: PyTorch 2.2, OpenCV 4.9, ONNX Runtime, Python 3.10.\n"
                    "4. Study Site Geolocation: Pasture fields at MUARIK, Wakiso, Uganda. Geolocation coordinates: Latitude 0.4508 N, Longitude 32.6144 E.")

    # Save document
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docx_path = os.path.join(script_dir, "Research_Proposal.docx")
    doc.save(docx_path)
    print(f"Word document saved to: {docx_path}")

if __name__ == "__main__":
    main()
