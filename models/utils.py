model_imports = {
    "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
    "Linear Regression": "from sklearn.linear_model import LinearRegression",
    "Decision Tree Classifier": "from sklearn.tree import DecisionTreeClassifier",
    "Decision Tree Regressor": "from sklearn.tree import DecisionTreeRegressor",
    "Random Forest Classifier": "from sklearn.ensemble import RandomForestClassifier",
    "Gradient Boosting Classifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "AdaBoost Classifier": "from sklearn.ensemble import AdaBoostClassifier",
    "SVC": "from sklearn.svm import SVC",
}

model_urls = {
    "Logistic Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "Decision Tree Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "Decision Tree Regressor": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    "Random Forest Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "Gradient Boosting Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "SVC": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
    "AdaBoost Classifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
}


model_infos = {
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
    "Gradient Boosting Classifier": """
        - Gradient boosting combines decision trees in an additive fashion from the start
        - Gradient boosting builds one tree at a time sequentially
        - Carefully tuned, gradient boosting can result in better performance than random forests
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
}
