# Assignment 1 - distributed in Github Repo e4040-2021Fall-assign1
The assignment is distributed as several jupyter notebooks and a number of directories and subdirectories in utils.

# Students need to follow the instructions below, and they also need to edit the README.md such that key information is shown in it - right after this line
TODO students to add text summarizing the key (high level) modifications that they did to solve the assignment

# Detailed instructions how to submit this assignment/homework:
1. The assignment will be distributed as a github classroom assignment - as a special repository accessed through a link
2. A students copy of the assignment gets created automatically with a special name - students have to rename the repo per instructions below
3. The solution(s) to the assignment have to be submitted inside that repository as a set of "solved" Jupyter Notebooks, and several modified python files which reside in directories/subdirectories
4. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud


## (Re)naming of the student repository (TODO students) 
INSTRUCTIONS for naming the student's solution repository for assignments with one student:
* This step will require changing the repository name
* Students MUST use the following name for the repository with their solutions: e4040-2021Fall-assign??-UNI (the first part "e4040-2021Fall-assign??" will probably be inherited from the assignment, so only UNI needs to be added) 
* Initially, the system will give the repo a name which ends with a  student's Github userid. The student MUST change that name and replace it with the name requested in the point above
* Good Example: e4040-2021Fall-assign??-zz9999;   Bad example: e4040-2021Fall-assign??-e4040-2021Fall-assign??-zz9999.
* This change can be done from the "Settings" tab which is located on the repo page.

INSTRUCTIONS for naming the students' solution repository for assignments with more students, such as the final project. Students need to use a 4-letter groupID): 
* Template: e4040-2021Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2021Fall-Project-MEME-zz9999-aa9999-aa0000.


# Organization of this directory

```
./
├── Assignment1_intro.ipynb
├── README.md
├── figures
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
├── requirements.txt
├── save_models
│   └── best_model.pkl
├── task1-basic_classifiers.ipynb
├── task2-mlp_numpy.ipynb
├── task3-mlp_tensorflow.ipynb
├── task4-questions.ipynb
└── utils
    ├── __pycache__
    │   ├── display_funcs.cpython-36.pyc
    │   ├── layer_funcs.cpython-36.pyc
    │   ├── layer_utils.cpython-36.pyc
    │   └── train_funcs.cpython-36.pyc
    ├── classifiers
    │   ├── __pycache__
    │   │   ├── basic_classifiers.cpython-36.pyc
    │   │   ├── logistic_regression.cpython-36.pyc
    │   │   ├── mlp.cpython-36.pyc
    │   │   ├── softmax.cpython-36.pyc
    │   │   └── twolayernet.cpython-36.pyc
    │   ├── basic_classifiers.py
    │   ├── logistic_regression.py
    │   ├── mlp.py
    │   ├── softmax.py
    │   └── twolayernet.py
    ├── display_funcs.py
    ├── features
    │   ├── __pycache__
    │   │   ├── pca.cpython-36.pyc
    │   │   └── tsne.cpython-36.pyc
    │   ├── pca.py
    │   └── tsne.py
    ├── layer_funcs.py
    ├── layer_utils.py
    └── train_funcs.py

8 directories, 33 files
```