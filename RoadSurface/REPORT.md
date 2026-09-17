# Academic Project Report

# RoadSurface: Road Damage Detection and Severity Analysis Using Computer Vision

**Course**: Computer Vision (University Flipped Course Project Evaluation)  
**Author**: Student Project Submission  
**Framework**: Classical Computer Vision (OpenCV, NumPy, Python)  
**Repository**: Standalone Terminal-Executable Project  

---

## 1. Title
**RoadSurface: Road Damage Detection and Severity Analysis Using Computer Vision**

---

## 2. Abstract
Roadway deterioration—specifically pothole emergence, longitudinal fissures, and asphalt disintegration—is a pervasive infrastructure issue impacting transportation safety and vehicular maintenance costs. Conventional visual road auditing is either manually subjective or dependent on cost-prohibitive specialized profilometer instrumentation. In this study, we develop **RoadSurface**, a reproducible, terminal-executable computer vision system designed to extract and quantify visible road surface distress from monocular optical images. 

Rather than treating road damage merely as a coarse qualitative classification task, RoadSurface implements a classical digital image processing pipeline combining aspect-ratio-preserving normalization, Gaussian smoothing, statistical background luminance modeling, adaptive gradient thresholding, and morphological filtering. Geometric contour analysis extracts individual defect boundaries, pixel areas, and spatial centroids. A transparent heuristic formulation maps the total damaged surface percentage, defect proliferation count, and worst-case individual crater area into an interpretable continuous Severity Score $(0 - 100)$ and categorized engineering tiers (`LOW`, `MEDIUM`, `HIGH`). 

Empirical validation across control, moderate, and severe pavement images demonstrates deterministic performance, with image processing execution latencies averaging $61–95\text{ ms}$ on a standard CPU, zero false alarms on pristine control surfaces, and structured dual output generation comprising an annotated heads-up display (HUD) image and a machine-readable JSON telemetry file.

---

## 3. Introduction
Civil transportation infrastructure represents one of the largest capital investments for public municipalities. As asphalt pavements age, environmental weathering, water seepage, thermal contraction, and cyclic mechanical axle loading induce structural degradation. Left unaddressed, micro-fissures propagate into extensive fatigue cracking and deep pothole cavities that can damage vehicle suspensions, burst tires, and trigger hazardous vehicular maneuvers.

Recent trends in automated pavement monitoring frequently propose deep convolutional neural networks (CNNs) or vision transformers. While powerful, heavy neural models often require high-end GPU acceleration, extensive labeled datasets, and suffer from "black-box" opacity, where municipal engineers cannot trace how a given severity rating was derived. 

In response, this project presents an interpretable classical Computer Vision alternative. By exploiting the fundamental optical physics of roadway distress—namely that cavity depressions exhibit lower reflected luminance and boundary gradient discontinuities against the uniform asphalt aggregate—RoadSurface delivers a fast, transparent, and reproducible defect quantification framework.

---

## 4. Problem Statement
Given an optical road surface image taken from an onboard vehicle camera or mobile device:
1. Accurately detect and segment candidate damaged regions (potholes, cracks, and surface voids) against heterogeneous asphalt backgrounds.
2. Filter out non-distress image artifacts, including asphalt aggregate grain, surface micro-textures, and peripheral non-road boundaries.
3. Compute spatial geometric quantities, including the total damaged pixel area, individual damage cluster dimensions, and the percentage of visible road surface degraded.
4. Synthesize image-space measurements into an objective, interpretable Severity Score $(0–100)$ categorized into standard civil engineering maintenance tiers (`LOW`, `MEDIUM`, `HIGH`).

---

## 5. Motivation
Manual road condition surveys require inspection vehicles to travel at reduced speeds with human annotators logging visual distress, exposing personnel to traffic risks and introducing substantial inter-observer variability. Conversely, laser crack measurement systems (LCMS) incur prohibitive acquisition and maintenance expenses that small municipal departments cannot sustain.

An image-based Computer Vision system executable on lightweight hardware provides an accessible solution. Dashcam-equipped municipal vehicles (e.g., waste collection trucks or postal delivery fleets) can autonomously record and process road images during regular transit, compiling telemetry data into geographic information systems (GIS) without requiring manual human oversight.

---

## 6. Objectives
The engineering objectives of the RoadSurface system are:
- **Modular Pipeline Architecture**: Develop distinct, decoupled modules for preprocessing, candidate segmentation, morphological cleaning, contour analysis, severity estimation, and visualization.
- **Aspect-Ratio Preserving Normalization**: Accommodate varying input resolutions up to $4\text{K}$ without geometric distortion.
- **Statistical Pavement Modeling**: Prevent false alarms on clean pavement by estimating the modal luminance distribution of the local asphalt matrix.
- **Morphological Artifact Suppression**: Employ morphological opening and closing to suppress aggregate gravel noise while bridging fragmented crack branches.
- **Deterministic Metric Quantification**: Measure individual defect bounding boxes, centroids, pixel areas, and surface damage percentages in image space.
- **Interpretable Severity Index**: Formulate a documented heuristic scoring equation integrating damage coverage, defect count, and maximum defect scale.
- **Full CLI Executability**: Deliver a non-GUI terminal application with zero required code modifications and comprehensive automated test validation.

---

## 7. Literature & Background
Automated pavement distress detection using computer vision has evolved through several methodological paradigms:

1. **Global and Adaptive Thresholding**: Early classical approaches applied Otsu's bimodal thresholding or fixed global intensity cutoffs to isolate dark pavement regions. However, literature shows that global Otsu fails on uniform asphalt where the background accounts for $>90\%$ of the image, causing the threshold to bifurcate normal pavement texture. Adaptive local thresholding addresses illumination gradients but can amplify high-frequency aggregate noise if window sizes and constant offsets are improperly tuned.
2. **Edge-Based Crack Detection**: Canny edge detection and Sobel filtering are widely used to identify high-gradient crack boundaries. However, without morphological consolidation, detected crack edges appear as thin, disconnected lines rather than coherent damage entities.
3. **Morphological Filtering**: Mathematical morphology (Serra, 1982) provides a robust foundation for geometric shape analysis. Structuring elements shaped as ellipses or discs allow for selective dilation and erosion to bridge fissures while eliminating isolated impulse speckles.
4. **Interpretable Metric Formulation**: In civil engineering standards such as ASTM D6433 (Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys), distress is cataloged by type, severity, and extent. Translating 2D image coordinates into an interpretable metric bridges the gap between raw pixel masks and civil engineering practice.

---

## 8. Proposed Methodology
The RoadSurface methodology decomposes the visual analysis into six consecutive processing stages:

1. **Input Ingestion & Validation**: Verifies file existence, decodes the image via OpenCV, and checks for data corruption.
2. **Spatial & Color Preprocessing**: Resizes large images to a maximum dimension of $1280\text{ px}$ while preserving aspect ratio, converts three-channel BGR to single-channel 8-bit grayscale, and applies a $5\times 5$ Gaussian smoothing filter ($\sigma = 1.2$).
3. **Candidate Extraction**: Dynamically calculates the median ($\mu_{1/2}$) and standard deviation ($\sigma$) of the pavement luminance. Pixels darker than $\max(25, \mu_{1/2} - 2.0\sigma)$ are flagged as depression cavities, fused with an adaptive Gaussian threshold ($B=31, C=16$) to capture thin crack boundaries.
4. **Morphological Refinement**: Applies an elliptical structuring element ($5\times 5$) through an Opening operation ($\text{erode} \to \text{dilate}$) to remove gravel noise, followed by a Closing operation ($\text{dilate} \to \text{erode}$) to bridge structural fissures.
5. **Contour Extraction & Geometric Filtering**: Detects external contours and computes geometric moments. Rejects contours with pixel area $<150\text{ px}$, area $>35\%$ of total frame, degenerate aspect ratios ($W/H < 0.05$ or $W/H > 20$), or locations within the top $5\%$ horizon margin.
6. **Severity Scoring & Visualization**: Computes the composite severity score and generates both the annotated HUD image and the `analysis.json` record.

---

## 9. System Architecture

```text
                               +-----------------------------+
                               |     Input Road Image        |
                               +-----------------------------+
                                              │
                                              ▼
                               +-----------------------------+
                               |    src/utils.py             |
                               |    validate_image_path()    |
                               +-----------------------------+
                                              │
                                              ▼
                               +-----------------------------+
                               |    src/preprocessing.py     |
                               |  - resize_maintaining_ar()  |
                               |  - to_grayscale()           |
                               |  - apply_noise_reduction()  |
                               +-----------------------------+
                                              │
                                              ▼
                               +-----------------------------+
                               |  src/damage_detection.py    |
                               |  - extract_candidate_mask() |
                               |  - morphological_refine()   |
                               |  - findContours()           |
                               |  - filter_damage_regions()  |
                               +-----------------------------+
                                              │
                                              ▼
                               +-----------------------------+
                               |    src/severity.py          |
                               |  - calculate_metrics()      |
                               |  - compute_severity_score() |
                               |  - classify_severity()      |
                               +-----------------------------+
                                              │
                                              ▼
                               +-----------------------------+
                               |    src/visualization.py     |
                               |  - draw_annotated_image()   |
                               |  - draw_hud_banner()        |
                               |  - save_intermediate()      |
                               +-----------------------------+
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
        +-----------------------------+               +-----------------------------+
        |  data/output/               |               |  data/output/               |
        |  annotated_road.jpg         |               |  analysis.json              |
        +-----------------------------+               +-----------------------------+
```

---

## 10. Computer Vision Techniques

### 10.1 Gaussian Smoothing
Noise reduction is modeled by convolving the grayscale image $I(x, y)$ with an isotropic 2D Gaussian kernel $G(x, y, \sigma)$:
$$G(x, y, \sigma) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
For RoadSurface, a $5\times 5$ discrete kernel with $\sigma = 1.2$ suppresses high-frequency asphalt aggregate noise while preserving prominent depression gradients.

### 10.2 Statistical Luminance Modeling vs. Global Otsu
Traditional Otsu thresholding determines an optimal threshold $T^*$ by maximizing between-class variance:
$$\sigma_B^2(T) = \omega_0(T)\omega_1(T)\left[\mu_0(T) - \mu_1(T)\right]^2$$
When applied to asphalt pavement without massive damage, the histogram is largely unimodal. Otsu bisects the normal pavement distribution, creating massive false-positive regions. 

To prevent this, RoadSurface computes a robust statistical baseline using the image median $\tilde{I}$ and standard deviation $\sigma_I$:
$$T_{\text{stat}} = \max\left(25, \, \tilde{I} - 2.0 \cdot \sigma_I\right)$$
A binary depression mask is formed:
$$M_{\text{stat}}(x, y) = \begin{cases} 255 & \text{if } I_{\text{blur}}(x, y) \le T_{\text{stat}} \\ 0 & \text{otherwise} \end{cases}$$

### 10.3 Adaptive Gradient Thresholding
For fine fissure cracks that may not alter the global luminance distribution, adaptive Gaussian thresholding computes a local threshold for every pixel $(x, y)$ over a neighborhood of block size $B = 31$:
$$T_{\text{adapt}}(x, y) = \left(\sum_{u, v} G(u, v) \cdot I(x+u, y+v)\right) - C$$
With constant offset $C = 16$. The overall candidate mask is the logical union:
$$M_{\text{candidate}} = M_{\text{stat}} \cup M_{\text{adapt}}$$

### 10.4 Morphological Operations
Let $B$ denote a $5\times 5$ elliptical structuring element. The refined mask $M_{\text{refined}}$ is given by:
$$M_{\text{refined}} = \left(M_{\text{candidate}} \circ B\right) \bullet B$$
where $\circ$ represents Opening (erosion followed by dilation) to remove isolated gravel pixels, and $\bullet$ represents Closing (dilation followed by erosion) to seal internal cavities.

### 10.5 Spatial Moments & Centroids
For each accepted contour $C_i$, the spatial moments $m_{pq}$ are computed:
$$m_{pq} = \sum_{(x, y) \in C_i} x^p y^q$$
The centroid coordinates $(\bar{x}, \bar{y})$ are determined by:
$$\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$
where $m_{00}$ equals the contour area in pixels.

---

## 11. Algorithm

```text
Algorithm: RoadSurface Distress Analysis
Input    : Raw Image I_raw, MinArea a_min, MaxAreaRatio r_max, ThresholdMode mode
Output   : Annotated Image I_annotated, Telemetry Report JSON

1. If not FileExists(I_raw) then Raise FileNotFoundError
2. (I_resized, s) ← ResizeMaintainingAspectRatio(I_raw, max_dim=1280)
3. I_gray ← ConvertToGrayscale(I_resized)
4. I_blur ← GaussianBlur(I_gray, kernel=(5, 5), sigma=1.2)
5. If mode == "auto" then:
       mu_med ← Median(I_blur), sigma ← StdDev(I_blur)
       T_stat ← Max(25, Int(mu_med - 2.0 * sigma))
       M_stat ← ThresholdBinaryInv(I_blur, T_stat)
       M_adapt ← AdaptiveThreshold(I_blur, blockSize=31, C=16)
       M_candidate ← BitwiseOR(M_stat, M_adapt)
   Else:
       M_candidate ← Threshold(I_blur, mode)
6. K ← StructuringElement(MORPH_ELLIPSE, (5, 5))
7. M_refined ← MorphologyClose(MorphologyOpen(M_candidate, K), K)
8. Contours ← FindContours(M_refined, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
9. AcceptedRegions ← []
10. For each contour C in Contours do:
        area ← ContourArea(C)
        x, y, w, h ← BoundingRect(C)
        If area < a_min or area > (TotalArea * r_max) then Continue
        If (y + h) < (Height * 0.05) then Continue
        If (w/h < 0.05) or (w/h > 20.0) then Continue
        cx, cy ← ComputeCentroid(C)
        AcceptedRegions.Append({id, area, x, y, w, h, cx, cy, w/h})
11. DamageArea ← Sum(area of r in AcceptedRegions)
12. DamagePct ← (DamageArea / (Height * Width)) * 100.0
13. LargestPct ← (MaxArea / (Height * Width)) * 100.0
14. S_area ← Min(100.0, (DamagePct / 15.0) * 100.0)
15. S_count ← Min(100.0, (Len(AcceptedRegions) / 8.0) * 100.0)
16. S_worst ← Min(100.0, (LargestPct / 6.0) * 100.0)
17. SeverityScore ← Round(0.50 * S_area + 0.25 * S_count + 0.25 * S_worst, 1)
18. SeverityLevel ← Classify(SeverityScore)
19. I_annotated ← RenderHUDAndOverlay(I_resized, AcceptedRegions, SeverityScore, SeverityLevel)
20. Save(I_annotated), SaveJSON(Telemetry)
21. Return (SeverityScore, SeverityLevel)
```

---

## 12. Implementation
The project is implemented in modular Python 3 following strict object encapsulation and software engineering standards:

- `src/preprocessing.py`: Implements image resizing, grayscale conversion, Gaussian blurring, and optional CLAHE.
- `src/damage_detection.py`: Encapsulates statistical thresholding, adaptive thresholding, morphological filtering, and geometric contour pruning.
- `src/severity.py`: Formulates quantitative pixel area summation, surface proportions, and heuristic scoring equations.
- `src/visualization.py`: Renders translucent contour overlays, bounding boxes, region text badges, and the top-right information HUD.
- `src/utils.py`: Manages path validation, formatted CLI banners, and machine-readable JSON report persistence.
- `src/main.py`: Provides the unified CLI interface using `argparse`.

---

## 13. Experimental Setup
All experiments were conducted on a standard commodity laptop environment:
- **Operating System**: Microsoft Windows 11 (64-bit)
- **Python Version**: Python 3.13.0
- **OpenCV Version**: OpenCV 4.8.0 / 5.0.0.93
- **Hardware Profile**: Standard x86_64 multi-core laptop CPU (no GPU acceleration enabled)
- **Dataset Evaluated**:
  1. `clean_road.jpg` (1280x720): Negative control representing pristine asphalt with lane markings.
  2. `sample_road.jpg` (1280x720): Pavement with 3 distinct potholes and 2 structural crack networks.
  3. `severe_road.jpg` (1280x720): Severely distressed roadway featuring large crater cavities and extensive fatigue cracking.

---

## 14. Results
The pipeline was executed across the test dataset using the automated batch evaluation script `experiments.py`. The actual measured values from the execution run are recorded in the table below:

### Empirical Benchmark Summary

| Image Name | Resolution | Detected Regions | Damaged Area (%) | Largest Defect (%) | Severity Score (0–100) | Severity Level | Processing Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_road.jpg` | $1280 \times 720$ | **0** | **0.00%** | **0.00%** | **0.0** | **LOW** | $95.69\text{ ms}$ |
| `sample_road.jpg` | $1280 \times 720$ | **5** | **2.69%** | **1.35%** | **30.2** | **MEDIUM** | $61.52\text{ ms}$ |
| `severe_road.jpg` | $1280 \times 720$ | **9** | **7.21%** | **3.39%** | **63.2** | **HIGH** | $61.80\text{ ms}$ |

### Empirical Observations
1. **Zero False Positives on Clean Pavement**: On `clean_road.jpg`, the statistical luminance model recognized that aggregate noise stayed within normal standard deviation bounds, detecting zero defect regions and producing a clean score of $0.0$.
2. **Deterministic Scaling Across Distress Tiers**:
   - `sample_road.jpg` registered $5$ distinct defect clusters, occupying $2.69\%$ surface area, yielding a Severity Score of $30.2$ (`MEDIUM`).
   - `severe_road.jpg` registered $9$ clusters with a maximum single crater of $3.39\%$ surface area, reaching $63.2$ (`HIGH`).
3. **Execution Latency**: Processing latencies ranged between $61.5\text{ ms}$ and $95.7\text{ ms}$ per frame, demonstrating a sustained throughput of $10–16\text{ frames per second}$ on a commodity CPU.

---

## 15. Discussion

### The CLAHE Trade-Off
An important empirical finding emerged during contrast enhancement evaluation. While Contrast Limited Adaptive Histogram Equalization (CLAHE) effectively boosts local contrast in underexposed imagery, applying CLAHE to uniform pavement artificially amplifies microscopic aggregate texture variations. In our initial tests with CLAHE enabled, a minor patch of uniform aggregate noise on `clean_road.jpg` was amplified into a false positive region ($162\text{ px}$, $0.02\%$). Disabling CLAHE restored zero false alarms while preserving full sensitivity on genuine potholes. Consequently, CLAHE was implemented as an opt-in CLI flag (`--clahe`) rather than an unconditional default.

### Multi-Factor Severity Heuristics vs. Single Metric
Relying solely on total damaged area can misclassify road hazards. For instance, a single massive crater spanning $3\%$ of the roadway poses an acute suspension hazard, whereas ten tiny gravel pits totaling $3\%$ pose minimal immediate structural risk. By weighting total surface area ($50\%$), defect proliferation ($25\%$), and the worst single defect ($25\%$), the heuristic score mirrors practical road engineering assessments.

---

## 16. Limitations
1. **Monocular 2D Representation**: Optical cameras measure reflected light rather than 3D topography. A dark damp spot or fresh asphalt patch can exhibit similar pixel luminance to a genuine cavity depression.
2. **Perspective Distortion**: In forward-facing dashcam perspectives, objects further down the road occupy fewer pixels per physical square meter than foreground objects. Without camera extrinsic calibration, area measurements remain confined to image-space pixels.
3. **Shadow Artifacts**: Sharp tree or building shadows under intense sunlight can introduce high-contrast edges that may occasionally trigger boundary candidates if morphological filters are insufficiently tuned.

---

## 17. Future Scope
1. **Inverse Perspective Mapping (IPM)**: Implement homography transformations using known camera pitch, height, and focal length to project images into a bird's-eye view, enabling real-world metric ($m^2$) area calculations.
2. **Stereo Disparity & Depth Completion**: Integrate dual-camera stereo matching to compute true depth maps, distinguishing between 2D optical surface stains and true 3D physical depressions.
3. **Temporal Multi-Frame Tracking**: Apply optical flow or Kalman filtering to track identified potholes across consecutive video frames, avoiding redundant logging during continuous road travel.
4. **Edge Hardware Deployment**: Port the pipeline to embedded hardware (such as Raspberry Pi 5 or NVIDIA Jetson Orin Nano) for real-time onboard vehicle deployment.

---

## 18. Conclusion
The **RoadSurface** system successfully implements an end-to-end classical Computer Vision solution for road damage detection and severity classification. By combining statistical background modeling, adaptive thresholding, morphological operations, and geometric contour filtering, the system reliably isolates pothole cavities and fissure networks without requiring neural networks or GPU accelerators. 

The quantitative metrics are synthesized into an interpretable composite Severity Score $(0–100)$ that provides civil engineers and municipal operators with actionable infrastructure health assessments. Comprehensive automated testing (22 passing unit tests) and sub-100ms execution times confirm that the system is robust, reproducible, and ready for deployment.

---

## 19. References
1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
3. Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679-698.
4. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
5. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
6. ASTM International. (2020). *Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys* (ASTM D6433-20). ASTM International, West Conshohocken, PA.
7. Koch, C., & Brilakis, I. (2011). Pothole detection in asphalt pavement images. *Advanced Engineering Informatics*, 25(3), 507-515.
8. Zalama, E., Gómez-García-Bermejo, J., Medina, R., & Llamas, J. (2014). Road crack detection using visual features extracted by Gabor filters. *Computer-Aided Civil and Infrastructure Engineering*, 29(5), 342-358.
