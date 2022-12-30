# tsml-eval

Evaluation tools and research algorithm implementations for classification, regression
and clustering algorithms built on the scikit learn and sktime interfaces.

Dependency installation
-----------------------

pip install -r requirements.txt

Export package versions to new requirements
-------------------------------------------

pip freeze > requirements.txt


sktime install from main, branch or fork
----------------------------------------

You can use this package with a sktime install that isn't deployed to pypip.

- First uninstall current sktime: 'pip uninstall sktime'
- Then install from git using i.e.
  - pip install git+https://github.com/sktime/sktime.git@main
  - pip install git+https://github.com/sktime/sktime.git@uea_debugging
  - pip install git+https://github.com/MatthewMiddlehurst/sktime-mm.git@main
