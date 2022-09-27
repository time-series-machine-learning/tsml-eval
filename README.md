# estimator-evaluation
Evaluation tools for classification and regression

Installation
------------
pip install -r requirements.txt


Export new requirements
-----------------------
pip freeze > requirements.txt 

Sktime install from branch
--------------------------
- You can use this package with a sktime install that isn't deployed to pypip
- First uninstall current sktime: 'pip uninstall sktime'
- Then install from git using 'pip install git+https://github.com/alan-turing-institute/sktime.git@estimator-evaluation'
- The values after the @ at the end is the branch i.e. another branch would be
'pip install git+https://github.com/alan-turing-institute/sktime.git@another-branch'