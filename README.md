# # Senior-Capstone-Project


# Literature Review

**Literature Review: Malware Analysis Using Static and Dynamic Techniques with Machine Learning Models**

**1. Introduction**


The exponential increase in the quantity and complexity of malicious software has made malware analysis a crucial component of cybersecurity. For the detection of complex and obfuscated malware, traditional signature-based techniques are no longer enough. As a result, machine learning (ML) models—including sophisticated deep learning models like Long Short-Term Memory (LSTM) networks and more traditional techniques like Logistic Regression (LR)—have become more and more popular in malware detection. Usually, either static or dynamic features that are taken from executables are subjected to these models.

2. Static Malware Analysis with Machine Learning
2.1 Overview
Static analysis inspects an executable without executing it. It typically involves extracting features such as:

- Opcode sequences

- API calls

- PE (Portable Executable) header information

- Strings and metadata

In other words, analysts examine the code structure, file headers, and embedded data to find potentially harmful traits and behaviors without actually running the malware on a system. 
These features are suitable for traditional ML algorithms due to their structured and well-defined format.

2.2 Logistic Regression in Static Analysis
Logistic Regression, a linear classifier, has been used in malware detection due to its simplicity and interpretability. Several studies have shown its effectiveness when applied to high-quality static features.

Raff et al. (2018) demonstrated that using n-grams of byte sequences combined with logistic regression yields competitive accuracy for binary classification.

Saxe and Berlin (2015) used features from the PE headers and metadata and applied logistic regression alongside other models. While LR lagged behind complex models in accuracy, it proved valuable in fast inference settings.

Advantages:

Fast training and inference

Robust to small datasets

Easy to interpret

Limitations:

Inability to capture complex patterns (e.g., obfuscation or polymorphism)

Linear decision boundaries may miss non-linear malware behavior

3. Dynamic Malware Analysis with Machine Learning
3.1 Overview
Dynamic analysis involves executing the malware in a controlled environment (sandbox) and monitoring behavior such as:

System calls

File system and registry changes

Network activity

Behavioral data is often temporal and more resilient to obfuscation, making it suitable for time-series models.

3.2 LSTM in Dynamic Analysis
LSTM networks, a type of recurrent neural network (RNN), excel at modeling sequential data and are well-suited for dynamic analysis.

Advay Balakrishnan and Guilliermo Goldsztein (2022) used LSTM models to analyze sequences of API calls. Their model has the  capability  to  easily  capture  the  nature  of  time  series  processing  due  to  their  sequential  structure,  whereas  generic neural networks cannot find an underlying relationship between data that is separated by timesteps. The pri-mary goal of the paper is to describe an approach to analyzing malware attacks on networks while accounting for each timestep of data and gaining more flexibility in the size of the detection request using the features of the Long Short-Term Memory (LSTM) architecture.

Mehta, R., Jurečková, O., Stamp, M. (2025). Malware Classification Using a Hybrid Hidden Markov Model-Convolutional Neural Network. In: Stamp, M., Jureček, M. (eds) Machine Learning, Deep Learning and AI for Cybersecurity. Springer, Cham. https://doi.org/10.1007/978-3-031-83157-7_4 presented a hybrid approach that combined static and dynamic features with LSTM layers to detect sophisticated malware variants.

Advantages:

Captures temporal dependencies in behavior logs

Handles variable-length input sequences

Better detection of advanced evasion techniques

Limitations:

Requires large labeled datasets for training

Computationally intensive

Risk of overfitting on imbalanced datasets

**4. Comparative Analysis and Hybrid Models**


Several studies suggest combining static and dynamic features improves detection accuracy:

Anusha Damodaran, Fabio Di Troia,  Visaggio Aaron Corrado,
Thomas H. Austin, Mark Stamp(2017) compared static, dynamic, and hybrid approaches, finding that hybrid models consistently achieved higher precision and recall. The very complexity of such detection techniques often makes it difficult to discern the actual benefit of any one particular aspect of a technique. The primary goal of this research was to test the tradeoffs between
static, dynamic, and hybrid analysis, while eliminating as many confounding variables
as possible.

Yao Saint Yen; Zhe Wei Chen; Ying Ren Guo; Meng Chang Chen (2019) integrated static and dynamic features into a deep learning pipeline, showing improved generalization across malware families.

Hybrid approaches can leverage Logistic Regression for quick filtering and LSTM for deeper behavioral analysis, combining the strengths of both.

5. Challenges and Research Gaps
Dataset Availability: Public malware datasets (e.g., VirusShare, EMBER) often lack comprehensive dynamic features.

Feature Engineering: Automated feature extraction and selection remain critical bottlenecks, especially for static analysis.

Model Interpretability: While LSTM offers high accuracy, its black-box nature poses challenges in trust and adoption.

Adversarial Evasion: ML models are susceptible to evasion through adversarial examples and mimicry attacks.

**6. Conclusion**


A viable avenue for malware investigation is machine learning, namely Logistic Regression and LSTM networks. While LSTM is excellent at catching intricate, dynamic activity, static analysis using LR is still helpful for quick, comprehensible detection. A comprehensive strategy is one that makes use of both static and dynamic analysis. Explainability, adversarial robustness, and the incorporation of real-time detection systems ought to be the main topics of future studies.

8. References/Resources
Ferhat Ozgur Catak & Ahmet Faruk Yazi (2024). A Benchmark API Call Datset for Windows PE. API
dataset reference for datasets.

Advay Balakrishnan and Guilliermo Goldsztein (2022). used LSTM models to analyze sequences of API calls.

Kaggle (2015). Static analysis dataset used for project. https://www.kaggle.com/datasets/blackarcher/malware-dataset

Kaggle (2016). Deep learning for classification of malware system call sequences. Australasian Joint Conference on Artificial Intelligence.

Huang, W., & Stokes, J. W. (2016). MtNet: A multi-task neural network for dynamic malware classification. International Conference on Detection of Intrusions and Malware.

Yao Saint Yen; Zhe Wei Chen; Ying Ren Guo; Meng Chang Chen (2019) integrated static and dynamic features into a deep learning pipeline, showing improved generalization across malware families.

Hardy, W., et al. (2016). DL4MD: A deep learning framework for intelligent malware detection. Proceedings of the 2016 International Conference on Data Science.

# User Story
🧾 User Story Example:
Title: Research Machine learning techniques to help analyze malware in datasets

As a cybersecurity analyst,
I want to analyze malware in dynamic and static datasets by using different machine learning techniques. This will help to create a method that  helps us better evaluate and understand the effectiveness of analyzing malware.

✅ Acceptance Criteria:

- The system should accept API call traces as input (in text format).

- The system should classify the input as malicious or benign with at least 90% accuracy on the validation set.

- The output should include a confidence score.

- The model must be trained on a labeled dataset of malware and benign samples.

🧠 Additional Notes :

- The APIhashes are extracted from dynamic and static analysis.

- The detection model uses deep learning (e.g., LSTM or CNN) trained on sequences of API calls.

- The APIhashes of both static and dynamic malware and benign are merged together into a csv file.


# Requirements


**1. Functional Requirements**

1.1 Data Collection

- The system must be able to collect and store both static features (e.g., PE headers, strings, imported functions) and dynamic features (e.g., system calls, registry access, file/network activity) from executable files.

- Support for both benign and malicious sample datasets.

- (Optional) Ability to interface with sandboxes (e.g., Cuckoo Sandbox) for dynamic behavior extraction.

**1.2 Feature Extraction and Preprocessing**

- Parse and extract apihashes from csv files(.csv files).
- Normalize and encode features for ML model compatibility.
- Merge dynamic and static apihashes into one csv file

**1.3 Model Training**

- Train a Logistic Regression model using static features for binary classification (malware vs benign).

- Train an LSTM model using dynamic behavioral sequences for malware classification.

- Split dataset into training, validation, and test sets.

**1.4 Model Evaluation**

- Evaluate both models using standard metrics (accuracy, precision, recall, F1-score, AUC).

- Make comparative analysis of the model using their calculated metrics(accuracy, sensitivity, specificity, precision and F1-score)

- Compare performance across multiple malware families and types.

- Perform cross-validation to ensure robustness.

**1.5 Prediction & Inference**

- Accept new executable files for analysis.

- Run both static and dynamic analysis pipelines.

- Predict whether the file is malicious or benign using trained models.

- Display prediction confidence and rationale (e.g., top contributing features for LR).

**1.6 Reporting and Visualization**

Generate reports with:

- Classification result

- Detected features

- Model confidence

Visualizations such as:

- Confusion matrix

- ROC curve

- System call sequences for dynamic samples

**2. Non-Functional Requirements**

2.1 Security
Sandbox environment must be isolated to prevent malware from escaping.

Datasets must be stored securely with restricted access.

2.2 Performance
Preprocessing and feature extraction should not take more than X seconds per sample (define based on hardware).

LSTM inference latency must be acceptable for near real-time analysis (< 1s per sample preferred).

2.3 Scalability

The system should support batch processing of thousands of malware samples.

Scalable infrastructure to handle increased dataset sizes and model retraining.

2.4 Usability

A user-friendly interface (CLI or web-based dashboard) for uploading samples and viewing results.

2.5 Maintainability
Modular code structure to allow updating feature extractors or models independently.

Logging for each step (data collection, preprocessing, training, inference).

**3. Dataset Requirements**

3.1 Static Datasets

- Malware and benign PE files (e.g., from EMBER dataset, VirusShare, or custom collected).

Features to extract:

- PE Header info

- Section entropy

- Imported libraries and API calls

- Strings

3.2 Dynamic Datasets

- System call traces or behavior logs from executing malware in sandbox environments.

- Tools: Cuckoo Sandbox, CAPE, or a custom VM-based environment.

- Data format: time-series sequences of actions/events.

4. Software Requirements

- Component	Requirement
- Programming Language	Python 3.8+
- ML Libraries	scikit-learn (for LR), TensorFlow or PyTorch (for LSTM)
- Sandbox Tool	Cuckoo Sandbox (for dynamic analysis)
- PE Analysis	pefile, lief, or custom parsers
- Data Handling	pandas, NumPy
- Visualization	matplotlib, seaborn, Plotly

**5. Hardware Requirements**

- Component	Minimum Specification	Recommended Specification
- CPU	Quad-core 2.4GHz	8-core 3.0GHz+
- RAM	8 GB	16–32 GB
- GPU (for LSTM)	Optional	NVIDIA GPU with CUDA support
- Storage	256 GB SSD	1 TB SSD
- Virtualization	Support for VM/Sandbox environments	Dedicated machine or cloud-based sandbox


# Design

**1. Design Overview**

This system is designed to analyze software samples (executables) using both static and dynamic features to classify them as malware or benign. It integrates a Logistic Regression model for static features and an LSTM model for analyzing time-series dynamic behaviors such as system calls.

Key Components:

- Data Ingestion Module: Upload or input samples.

- Static Feature Extractor: Extracts PE headers, strings, and API calls.

- Dynamic Feature Extractor: Executes sample in sandbox, collects behavior logs.

- Preprocessor: Normalizes, encodes, and prepares data for modeling.

- ML Engine: Contains and trains Logistic Regression and LSTM models.

- Prediction Engine: Inference using trained models.

- Reporting & Visualization: Displays results and model insights.

- Storage Module: Manages dataset and model storage.

**Architecture/Model**

Presentation Layer
- Web Dashboard / CLI
- Upload, Visualization, Reports

Application Layer                             
 - Feature Extraction Modules                         
 - ML Engine (Training/Prediction Logic)

 ML Models Layer                                         
 - Logistic Regression Model (Static Features)           
 - LSTM Model (Dynamic Features)

Data Layer                                              
 - Raw Sample Storage (Files, Logs)                      
 - Feature Storage (Preprocessed datasets)               
 - Model Storage (Serialized model files)                
 - Report Storage (JSON/HTML reports)  


**Use case Diagram**

**Actors**

Analyst: Uploads malware samples, views results.

**Use Cases**

- Upload Sample

- Extract Features (Static/Dynamic)

- Train Models

- Predict Result

- View Classification Report






