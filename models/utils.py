model_imports = {
    "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
    "Linear Regression": "from sklearn.linear_model import LinearRegression",
    "Decision Tree Classifier": "from sklearn.tree import DecisionTreeClassifier",
    "Decision Tree Regressor": "from sklearn.tree import DecisionTreeRegressor",
    "Random Forest Classifier": "from sklearn.ensemble import RandomForestClassifier",
    "Random Forest Regressor": "from sklearn.ensemble import RandomForestRegressor",
    "Gradient Boosting Classifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "Gradient Boosting Regressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "AdaBoost Classifier": "from sklearn.ensemble import AdaBoostClassifier",
    "Support Vector Regression": "from sklearn.svm import SVR",
}

model_urls = {
    "Linear Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    "Logistic Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "Decision Tree Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "Decision Tree Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    "Random Forest Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "Random Forest Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    "Gradient Boosting Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "Gradient Boosting Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    "Support Vector Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    "AdaBoost Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
}


model_infos = {
    "Linear Regression": """
        - Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). 
        - More specifically, that y can be calculated from a linear combination of the input variables (x).
        - Different techniques can be used to prepare or train the linear regression equation from data, the most common of which is called Ordinary Least Squares.
    """,
    "Logistic Regression": """
        - A logistic regression is only suited to **linearly separable** problems
        - It's computationally fast and interpretable by design
        - It can handle non-linear datasets with appropriate feature engineering
    """,
    "Decision Tree Classifier": """
        - Decision trees are simple to understand and intrepret
        - They are prone to overfitting when they are deep (high variance)
    """,
    "Decision Tree Regressor": """
        - With a particular data point, it is run completely through the entirely tree by answering True/False questions till it reaches the leaf node. 
        - The final prediction is the average of the value of the dependent variable in that particular leaf node. 
        - Through multiple iterations, the Tree is able to predict a proper value for the data point.
    """,
    "Random Forest Classifier": """
        - They have lower risk of overfitting in comparison with decision trees
        - They are robust to outliers
        - They are computationally intensive on large datasets 
        - They are not easily interpretable
    """,
    "Random Forest Regressor": """
        - Random forest is a bagging technique. The trees in random forests are run in parallel
        - They operate by constructing a multitude of decision trees at training time mean prediction (regression) of the individual trees.
        - A random forest is a meta-estimator (i.e. it combines the result of multiple predictions) which aggregates many decision trees.
    """,
    "Gradient Boosting Classifier": """
        - Gradient boosting combines decision trees in an additive fashion from the start
        - Gradient boosting builds one tree at a time sequentially
        - Carefully tuned, gradient boosting can result in better performance than random forests
    """,
    "Gradient Boosting Regressor": """
        - Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
        - When a decision tree is the weak learner, the resulting algorithm is called gradient boosted trees, which usually outperforms random forest
    """,
    "SVC": """
        - SVMs or SVCs are effective when the number of features is larger than the number of samples
        - They provide different type of kernel functions
        - They require careful normalization   
    """,
    "AdaBoost Classifier": """
        - AdaBoost is similar to Random Forest in that they both tally up the predictions made by each decision trees within the forest to decide on the final classification. 
        - There are however, some subtle differences. For instance, in AdaBoost, the decision trees have a depth of 1 (i.e. 2 leaves). 
        - In addition, the predictions made by each decision tree have varying impact on the final prediction made by the model.
    """,
    "Support Vector Regression": """
        - SVR gives us the flexibility to define how much error is acceptable in our model.
        - It will find an appropriate line (or hyperplane in higher dimensions) to fit the data.
    """,
}
