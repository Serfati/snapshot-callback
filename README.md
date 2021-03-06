<img src="https://in.bgu.ac.il/marketing/graphics/BGU.sig3-he-en-white.png" height="48px" align="right" /> 

<img src="https://cdn-images-1.medium.com/max/2000/1*T5WWecP_EaQWk1yDX15h_w.png" height="200px"/> 

  
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Serfati/snapshot-callback) ![](https://img.shields.io/apm/l/atomic-design-ui.svg?)


# Description  
This repository contains an implementation in Keras of the paper Snapshot Ensembles: Train 1, get M for free.

The authors use a modified version of cyclical learning rate to force the model to fall into local minima at the end of each cycle. Each local minima makes different mistakes. Thus the ensemble of every local minima helps to reach a better generalization.

### Training and Testing Models
Let us try the following models:

- Baseline
- Snapshot
- ELRS - ours

Scoring function will be F1, since it is more costly to have false negatives than false positives

#### **Nested K-Fold Cross Validation**

<img src="https://hackingmaterials.lbl.gov/automatminer/_images/cv_nested.png" height="200px"/> 

## ⚠️ Prerequisites  
  
- [Python 3.7](https://www.python.org/download/releases/3.6/)  
- [Git 2.26](https://git-scm.com/downloads/)  
- [PyCharm IDEA](https://www.jetbrains.com/pycharm/) (recommend)  

## 📦 How To Install  
  
You can modify or contribute to this project by following the steps below:  
  
**1. Clone the repository**  
  
- Open terminal ( <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd> )  
  
- [Clone](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) to a location on your machine.  
 ```bash  
 # Clone the repository 
 $> git clone https://github.com/serfati/snapshot-callback.git  

 # Navigate to the directory 
 $> cd snapshot-callback
  ``` 

**2. Install Dependencies**  
  
 ```bash  
 # install with pip/conda 
 $> pip install -r requirments.txt
 ```  

**3. launch of the project**  
  
 ```bash  
 # Run nootebook 
 $> jupyter aml-snapshot.ipynb
 ```  

- **Or open with Colab**
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Serfati/snapshot-callback)

---  

> author Serfati
  
## ⚖️ License  
  
This program is free software: you can redistribute it and/or modify it under the terms of the **MIT LICENSE** as published by the Free Software Foundation.  
  
**[⬆ back to top](#description)**
