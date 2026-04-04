# Master's Thesis Structure

## Development and Implementation of an Autonomous Mobile Goods Delivery System Based on TurtleBot3 with Manipulator

### Estimated Page Distribution: ~85 pages

---

## Front Matter (~5 pages)

- Title Page
- Declaration of Authorship
- Abstract (English)
- Abstract (German / second language)
- Acknowledgements
- Table of Contents
- List of Figures
- List of Tables
- List of Abbreviations and Symbols

---

## Chapter 1: Introduction (6-7 pages)

### 1.1 Background and Motivation (2 pages)
- Growth of warehouse automation and intralogistics
- Limitations of fixed infrastructure (conveyors, AGVs on rails)
- Need for flexible, autonomous mobile manipulation
- Relevance to Industry 4.0 and smart factory concepts
- Gap between industrial AMRs and accessible research platforms

### 1.2 Problem Statement (1 page)
- Lack of integrated, end-to-end autonomous pick-and-place solutions on low-cost platforms
- Challenges: navigation in cluttered environments, reliable grasping of small objects, task sequencing, human-robot interaction
- Need for a complete pipeline: perception, planning, manipulation, and delivery

### 1.3 Objectives and Scope (1 page)
- Primary objective: develop and evaluate an autonomous mobile delivery system
- Sub-objectives:
  - Implement SLAM-based mapping and A* path planning
  - Develop YOLO-based object detection for cubes and delivery zones
  - Program a 4-DOF manipulator for reliable pick-and-place
  - Design a graphical user interface for mission control
  - Implement autonomous charging dock return
- Scope limitations: indoor environment, predefined object types, simulation-first approach

### 1.4 Research Questions (0.5 pages)
- RQ1: How effectively can a low-cost mobile manipulator perform autonomous multi-object pick-and-place tasks?
- RQ2: What is the reliability of YOLO-based visual servoing for object alignment and grasping?
- RQ3: How does hybrid navigation (SLAM + visual servo) compare to pure odometry-based approaches?
- RQ4: Can the system autonomously return to a charging station after task completion?

### 1.5 Thesis Organization (0.5 pages)
- Brief overview of each chapter

---

## Chapter 2: Literature Review and Theoretical Background (12-14 pages)

### 2.1 Autonomous Mobile Robots in Logistics (2 pages)
- History of AMRs in warehousing (Kiva/Amazon Robotics, Fetch Robotics, MiR)
- Classification: AGV vs AMR vs mobile manipulator
- Current commercial solutions and their limitations
- Research trends in mobile manipulation for logistics

### 2.2 Robot Operating System 2 (ROS 2) (2 pages)
- Architecture of ROS 2 Humble Hawksbill
- DDS communication middleware
- Node lifecycle, topics, services, actions
- ROS 2 control framework (ros2_control, joint trajectory controllers)
- Comparison with ROS 1

### 2.3 Simultaneous Localization and Mapping (SLAM) (2 pages)
- Fundamentals of SLAM problem
- LiDAR-based SLAM approaches (GMapping, Cartographer, Hector SLAM)
- Occupancy grid representation
- Map building from laser scan data
- Challenges: loop closure, dynamic environments

### 2.4 Path Planning and Navigation (2 pages)
- Global planning: A* algorithm, Dijkstra's algorithm, RRT
- A* heuristics and optimality guarantees
- Costmap generation and obstacle inflation
- Local planning and obstacle avoidance
- Pure pursuit and proportional heading control

### 2.5 Object Detection with Deep Learning (2 pages)
- Evolution of object detection: R-CNN, SSD, YOLO family
- YOLOv8 architecture (backbone, neck, head)
- Transfer learning and fine-tuning on custom datasets
- Real-time inference considerations on embedded platforms
- Data augmentation strategies for robotic applications

### 2.6 Visual Servoing (1.5 pages)
- Image-based visual servoing (IBVS) vs position-based (PBVS)
- Proportional control from pixel error
- Eye-in-hand vs eye-to-hand configurations
- Applications in robotic grasping

### 2.7 Robotic Grasping and Manipulation (1.5 pages)
- Grasp planning for parallel-jaw grippers
- Physics simulation challenges (ODE solver limitations)
- Virtual grasp approaches (fixed joint attachment)
- Force/contact-based grasp detection

### 2.8 Related Work and Comparison (1 page)
- Similar projects on TurtleBot3 platform
- Comparison of approaches (table format)
- Identification of gaps addressed by this thesis

---

## Chapter 3: System Architecture and Design (8-10 pages)

### 3.1 System Overview (2 pages)
- High-level architecture diagram (block diagram)
- Two-phase operation concept: Mapping Phase + Mission Phase
- Software stack: Ubuntu 22.04 + ROS 2 Humble + Gazebo Classic 11
- Component interaction diagram (ROS topic/service graph)

### 3.2 Hardware Platform (2 pages)
- TurtleBot3 Waffle Pi specifications
  - Differential drive kinematics
  - LDS-02 LiDAR sensor (360-degree, 5 Hz)
  - Raspberry Pi camera module
  - IMU (gyroscope + accelerometer)
- OpenMANIPULATOR-X specifications
  - 4-DOF serial chain (Dynamixel XM430 servos)
  - Parallel-jaw gripper
  - Workspace analysis and reachable volume
  - Joint limits and payload capacity

### 3.3 Software Architecture (2 pages)
- Modular design philosophy
- ROS 2 node graph and communication topology
- Thread model: ROS spin thread + PyQt5 GUI thread
- Module dependency diagram
- Configuration management (centralized config.py)

### 3.4 Finite State Machine Design (2 pages)
- State diagram for complete mission cycle
- State transition conditions and guards
- Three main phases: Pick Cycle, Drop Cycle, Dock Cycle
- Task ordering and multi-object sequencing (RED -> BLUE -> GREEN)
- Error recovery and fallback strategies

### 3.5 Simulation Environment Design (1.5 pages)
- Gazebo world layout (room dimensions, wall positions)
- Object placement: 3 tables with colored cubes, 3 drop baskets
- Charging dock design and placement
- Physics parameters: friction, contact, gravity
- Camera and sensor simulation fidelity

---

## Chapter 4: Mapping and Navigation (10-12 pages)

### 4.1 LiDAR-Based Occupancy Grid Mapping (3 pages)
- Occupancy grid representation (480x480 grid, 0.025 m resolution)
- Scan-to-grid projection algorithm
- Free/occupied/unknown cell classification
- Map update strategy during exploration
- Map persistence: numpy array, PGM, YAML formats

### 4.2 Autonomous Exploration (2 pages)
- Waypoint-based exploration strategy (17 predefined waypoints)
- Explorer state machine (IDLE -> ROTATING -> DRIVING -> ARRIVED -> SAVING -> DONE)
- Coverage path: left column, center corridor, right column
- Obstacle avoidance during exploration
- Progress tracking and completion criteria

### 4.3 A* Path Planning (3 pages)
- Costmap generation with obstacle inflation (8-pixel radius)
- A* implementation details
  - 8-connected grid neighborhood
  - Euclidean distance heuristic
  - Open/closed set management
  - Path simplification (every 3rd waypoint)
- Path replanning on obstacle detection
- Computational performance analysis

### 4.4 Point-to-Point Navigation (2 pages)
- Navigator state machine (IDLE -> PLANNING -> ROTATING -> DRIVING -> REACHED)
- Heading-first rotation then forward driving
- Proportional heading correction during driving
- Waypoint reach threshold (0.15 m)
- Speed control: cruise (0.10 m/s) vs near-obstacle (0.05 m/s)

### 4.5 L-Path Navigation Strategy (1 page)
- Two-waypoint approach to tables and drop zones
- First waypoint: lateral positioning (same Y as target)
- Second waypoint: longitudinal approach (toward target)
- Advantage: avoids diagonal paths through cluttered areas

### 4.6 Obstacle Avoidance (1 page)
- Front distance computation from LiDAR (20-degree cone)
- Three-zone speed control: stop (< 0.23 m), slow (< 0.40 m), normal
- Emergency stop behavior
- Selective bypass during dock approach

---

## Chapter 5: Object Detection and Visual Perception (10-12 pages)

### 5.1 YOLO Model Selection and Architecture (2 pages)
- YOLOv8 nano architecture overview
- Model size and inference speed considerations
- Suitability for real-time robotic applications
- Comparison with alternative detectors (SSD, Faster R-CNN)

### 5.2 Dataset Creation (3 pages)
- Automated data capture pipeline
  - Gazebo teleportation-based viewpoint generation
  - HSV color-based auto-labeling for cubes and drop zones
  - Viewpoint distribution: close-up, mid-range, far views
- Dataset statistics: 372 images, 7 classes
- Class distribution analysis
- YOLO annotation format (class_id, x_center, y_center, width, height)
- Charging dock dataset extension
  - Orange contact pad for HSV-based detection
  - Bounding box expansion strategy
  - 92 additional dock images from varied angles

### 5.3 Model Training (2 pages)
- Training configuration (epochs, batch size, augmentation)
- Data augmentation pipeline: HSV jitter, flips, scale, translate, erasing
- Training curves: loss, mAP, precision, recall
- Hyperparameter selection rationale
- Training hardware and time

### 5.4 Detection Pipeline Integration (2 pages)
- PerceptionModule class architecture
- Detection data flow: camera callback -> YOLO inference -> detection list
- Class-specific filtering (area thresholds for cubes vs zones vs dock)
- Target finding: highest-confidence detection selection
- Normalized coordinates for resolution independence

### 5.5 Visual Servoing for Object Alignment (2 pages)
- Pixel-error based proportional control
- Alignment center tolerance: 10 pixels (cubes), 15 pixels (dock)
- Angular velocity computation: omega = -Kp * (cx - 0.5)
- Lost target recovery: drift continuation, re-search fallback
- Combined approach: forward drive + proportional steering

### 5.6 Camera Overlay and HUD (1 page)
- Real-time annotation of bounding boxes
- Color-coded class visualization
- Head-up display: current state, task label, LiDAR distance
- Detection confidence display

---

## Chapter 6: Manipulation and Grasping (8-10 pages)

### 6.1 OpenMANIPULATOR-X Kinematics (2 pages)
- Forward kinematics of 4-DOF arm
- Denavit-Hartenberg parameters
- Workspace analysis for pick-and-place poses
- Joint trajectory control via ROS 2 actions

### 6.2 Gripper Design Modifications (2 pages)
- Original STL mesh collision limitations
- Simplified box collision primitives (0.055 x 0.016 x 0.050)
- Friction parameter tuning (mu=100)
- Joint effort limits and position limits (-0.015 to 0.019 m)
- Minimum contact depth (0.003 m)

### 6.3 Pick Sequence (2 pages)
- Pre-pick arm positioning (ARM_PRE_PICK)
- Gripper descent to cube height (ARM_PICK)
- Contact sensor-triggered attachment
  - Gazebo bumper sensors on left/right finger links
  - Auto-attach callback on contact detection
- IFRA LinkAttacher plugin: fixed joint creation
- Lift sequence (ARM_LIFT -> ARM_CARRY)

### 6.4 Place Sequence (2 pages)
- Arm extension over basket (ARM_DROP_EXTEND -> ARM_DROP_OVER)
- Link detachment via IFRA plugin
- Cube teleportation to basket center (physics reliability)
- Gripper open and arm retraction (ARM_DROP_RETREAT)
- Post-place backup maneuver

### 6.5 Table Alignment for Reliable Grasping (1.5 pages)
- Three-phase alignment at table:
  1. LiDAR perpendicularity check (line fitting, slope threshold 0.02)
  2. YOLO re-centering gate (cube must be centered before pick)
  3. Final creep to optimal pick distance
- Odom-based lateral drift correction using known table Y-coordinates

### 6.6 Grasp Reliability and Physics Limitations (0.5 pages)
- ODE solver limitations for small object grasping
- Why pure physics grip fails in Gazebo
- Virtual attachment as pragmatic solution
- Implications for real hardware deployment

---

## Chapter 7: Autonomous Charging Dock Return (5-6 pages)

### 7.1 Motivation and Design (1 page)
- Energy management in autonomous mobile robots
- Autonomous docking as a requirement for continuous operation
- Design goals: YOLO-based detection, visual servo approach

### 7.2 Charging Dock Model (1 page)
- SDF model description: platform, back wall, LED indicator, guide rails, orange contact pad
- Collision-free platform design (visual-only base for wheel traversal)
- Distinctive visual markers for YOLO detection

### 7.3 YOLO-Based Dock Detection (1 page)
- Training data generation for "charging_dock" class
- HSV-based auto-labeling using orange contact pad
- Bounding box expansion to cover full dock structure
- Detection integration with existing 6-class model (7 classes total)

### 7.4 Dock Approach State Machine (1.5 pages)
- State flow: RETURN_HOME -> SEARCH_DOCK -> ALIGN_DOCK -> APPROACH_DOCK -> DOCK_CREEP -> PARKED
- SEARCH_DOCK: face south, scan for dock detection
- ALIGN_DOCK: center dock in camera (15px tolerance)
- APPROACH_DOCK: visual servo with LiDAR stop distance (0.30 m)
- DOCK_CREEP: timed forward creep (4 s at 0.04 m/s)
- Obstacle avoidance bypass during final approach

### 7.5 GUI Integration (0.5 pages)
- Status messages: "Searching for Charging Dock", "Approaching", "Docking...", "Charging -- Mission Complete!"
- Mission button state: "CHARGING -- MISSION COMPLETE!"

---

## Chapter 8: User Interface Design (4-5 pages)

### 8.1 GUI Architecture (1 page)
- PyQt5 framework selection rationale
- Thread-safe design: ROS daemon thread + Qt main thread
- 30 FPS update timer for smooth visualization
- Three-column layout philosophy

### 8.2 Camera and Map Visualization (1.5 pages)
- Live camera feed with YOLO overlay
- Navigation map rendering: occupancy grid, inflated obstacles, planned path
- Robot position and heading indicator
- Exploration progress bar

### 8.3 Mission Control Panel (1 page)
- Phase and state display with color coding
- Color indicators for task progress (RED/BLUE/GREEN)
- START MISSION and STOP ALL controls
- Scrolling log area for event history

### 8.4 State Visualization (0.5 pages)
- 24-state color mapping
- Friendly human-readable state names
- Dock-specific status messages
- Detection information display

---

## Chapter 9: Experimental Evaluation (10-12 pages)

### 9.1 Experimental Setup (1.5 pages)
- Simulation environment: Gazebo Classic 11 on Ubuntu 22.04
- Hardware specifications of development machine
- World configuration: room dimensions, object placement
- Robot configuration: sensor parameters, controller settings

### 9.2 Mapping Evaluation (2 pages)
- Map quality assessment: occupancy grid accuracy
- Coverage completeness after exploration
- Map building time
- Comparison: generated map vs ground truth world layout
- Figures: occupancy grid at various exploration stages

### 9.3 Navigation Performance (2 pages)
- Path planning success rate across multiple trials
- Navigation time: A* planned path vs L-path odometry
- Obstacle avoidance effectiveness
- Path optimality: planned path length vs straight-line distance
- Table: navigation times for each cube/zone target

### 9.4 Object Detection Accuracy (2 pages)
- YOLO model performance metrics: mAP@0.5, precision, recall
- Per-class detection accuracy (cubes, zones, dock)
- Detection performance vs distance (close, medium, far)
- Confusion matrix analysis
- False positive/negative analysis
- Figures: training loss curves, precision-recall curves

### 9.5 Pick-and-Place Success Rate (2 pages)
- Overall mission success rate (N trials)
- Per-cube success rate (RED, BLUE, GREEN)
- Failure mode analysis:
  - Alignment failures
  - Grasp failures
  - Navigation failures
  - Drop failures
- Timing analysis: average time per pick-place cycle
- Table: detailed results per trial

### 9.6 Charging Dock Return Evaluation (1 page)
- Dock detection success rate
- Docking alignment accuracy
- Final parking position analysis
- Time from last drop to docked state

### 9.7 End-to-End Mission Evaluation (1.5 pages)
- Complete mission timing (3 cubes + dock return)
- System reliability over multiple consecutive runs
- Resource utilization (CPU, memory during operation)
- Comparison with thesis objectives

---

## Chapter 10: Discussion (5-6 pages)

### 10.1 Key Findings (2 pages)
- Summary of quantitative results
- Answers to research questions (RQ1-RQ4)
- Comparison with related work
- Strengths of the implemented approach

### 10.2 Limitations (2 pages)
- Simulation vs real-world gap
  - Physics fidelity of Gazebo ODE solver
  - Virtual grasp vs physical contact
  - Idealized sensor models
- Environmental constraints
  - Static environment assumption
  - Predefined object types and positions
  - Known room layout
- Technical limitations
  - Single-object grasp only
  - No dynamic obstacle avoidance during manipulation
  - Fixed arm poses (no adaptive grasping)
  - YOLO confidence threshold sensitivity

### 10.3 Lessons Learned (1 page)
- Importance of modular architecture
- Value of visual servoing over pure odometry
- Gazebo physics workarounds for grasping
- Iterative tuning of control parameters

### 10.4 Real-World Deployment Considerations (1 page)
- Sensor calibration requirements
- Environmental lighting variability
- Floor surface and wheel traction
- Network latency for distributed ROS 2
- Safety considerations (ISO standards)

---

## Chapter 11: Conclusion and Future Work (3-4 pages)

### 11.1 Summary of Contributions (1.5 pages)
- Complete autonomous mobile manipulation pipeline
- Integration of SLAM, A* planning, YOLO detection, and visual servoing
- Automated YOLO training data generation from simulation
- Autonomous charging dock return with visual detection
- Modular, extensible software architecture on ROS 2
- Comprehensive GUI for mission monitoring and control

### 11.2 Future Work (2 pages)
- **Hardware validation:** Deploy on physical TurtleBot3 Waffle Pi
- **Dynamic environments:** Moving obstacles, human coexistence
- **Advanced grasping:** 6-DOF arm, force-torque sensing, adaptive grasp planning
- **Multi-robot coordination:** Fleet management, task allocation
- **Enhanced perception:** 3D object detection (depth camera), instance segmentation
- **Reinforcement learning:** Learned manipulation policies
- **Natural language commands:** Voice-based task specification
- **Battery management:** Real charging integration, energy-aware planning
- **Extended object set:** Variable shapes, sizes, and weights
- **ROS 2 Navigation Stack (Nav2):** Migration from custom navigator to Nav2

---

## References (~3-4 pages)

Suggested categories of references:
- ROS 2 documentation and tutorials (Open Robotics)
- YOLO papers (Redmon et al., Ultralytics YOLOv8)
- SLAM literature (Grisetti et al., Hess et al.)
- A* and path planning (Hart et al., LaValle)
- TurtleBot3 documentation (ROBOTIS)
- OpenMANIPULATOR-X documentation (ROBOTIS)
- Gazebo simulation (Open Robotics)
- Visual servoing (Chaumette & Hutchinson)
- Warehouse automation surveys
- Mobile manipulation reviews
- PyQt5 documentation
- IFRA LinkAttacher plugin documentation

---

## Appendices (~5-6 pages)

### Appendix A: Configuration Parameters
- Complete listing of config.py constants
- Arm pose definitions (table format)
- Navigation parameters
- YOLO training hyperparameters

### Appendix B: World File Layout
- Room dimensions and wall coordinates
- Object positions (tables, cubes, baskets, dock)
- Coordinate system diagram

### Appendix C: State Machine Transition Table
- Complete state transition table (from-state, condition, to-state)
- 24 states with all transitions

### Appendix D: ROS 2 Topic and Service List
- All ROS 2 topics with message types
- All services and actions used
- Node communication graph

### Appendix E: Installation and Setup Guide
- System requirements
- Package dependencies
- Build instructions
- Launch commands

### Appendix F: YOLO Dataset Samples
- Sample annotated images per class
- Class distribution histogram
- Training/validation split details

---

## Suggested Figures and Tables

### Key Figures:
1. System architecture block diagram
2. ROS 2 node and topic graph
3. TurtleBot3 + OpenMANIPULATOR-X photo/render
4. Gazebo world overview (top-down and perspective)
5. State machine diagram (full FSM)
6. Occupancy grid at various exploration stages
7. A* path planning example (costmap + planned path)
8. L-path navigation strategy illustration
9. YOLO detection examples (cubes, zones, dock)
10. Training loss and mAP curves
11. Pick sequence: arm poses step-by-step
12. Place sequence: arm poses step-by-step
13. Gripper collision modification diagram
14. Table alignment (LiDAR perpendicularity)
15. Visual servoing control loop diagram
16. Charging dock model (SDF render)
17. Dock approach sequence (camera view)
18. GUI screenshot (full interface)
19. GUI screenshot (during mission)
20. Navigation map visualization
21. Confusion matrix for YOLO detection
22. Mission timeline (Gantt-style per cube)

### Key Tables:
1. TurtleBot3 Waffle Pi specifications
2. OpenMANIPULATOR-X joint parameters
3. YOLO training configuration
4. YOLO class list with detection colors
5. Arm pose definitions (joint angles)
6. State machine states and descriptions
7. Navigation parameters
8. Object positions in world
9. Experimental results: navigation times
10. Experimental results: detection accuracy per class
11. Experimental results: pick-place success rates
12. Comparison with related work

---

## Page Count Estimate

| Section | Pages |
|---------|-------|
| Front Matter | 5 |
| Ch 1: Introduction | 7 |
| Ch 2: Literature Review | 13 |
| Ch 3: System Architecture | 10 |
| Ch 4: Mapping and Navigation | 11 |
| Ch 5: Object Detection | 11 |
| Ch 6: Manipulation | 9 |
| Ch 7: Charging Dock | 6 |
| Ch 8: User Interface | 5 |
| Ch 9: Experimental Evaluation | 11 |
| Ch 10: Discussion | 6 |
| Ch 11: Conclusion | 4 |
| References | 4 |
| Appendices | 6 |
| **Total** | **~108 pages** |

> Note: With figures and tables included inline, the actual page count
> will reach approximately 85-90 pages of core content (excluding
> front matter and appendices).
