# tsml-estimator-evaluation

Evaluation tools for classification, regression and clustering algorithms built on the sktime package.

Installation
------------
pip install -r requirements.txt

Export new requirements
-----------------------
pip freeze > requirements.txt

sktime install from main or branch
--------------------------
You can use this package with a sktime install that isn't deployed to pypip.

- First uninstall current sktime: 'pip uninstall sktime'
- Then install from git using 'pip install git+https://github.com/sktime/sktime.git@main'
- The values after the @ at the end is the branch i.e. another branch would be 'pip install git+https://github.com/sktime/sktime.git@another-branch'
