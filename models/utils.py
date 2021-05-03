model_imports = {
    "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
    "Linear Regression": "from sklearn.linear_model import LinearRegression",
    "Decision Tree Classifier": "from sklearn.tree import DecisionTreeClassifier",
    "Random Forest Classifier": "from sklearn.ensemble import RandomForestClassifier",
    "Gradient Boosting Classifier": "from sklearn.ensemble import GradientBoostingClassifier",
    "AdaBoost Classifier": "from sklearn.ensemble import AdaBoostClassifier",
    "SVC": "from sklearn.svm import SVC",
}

model_urls = {
    "Logistic Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "Decision Tree": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
    "Random Forest": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "Gradient Boosting": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
    "SVC": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
}


model_infos = {
    "Logistic Regression": """
        - A logistic regression is only suited to **linearly separable** problems
        - It's computationally fast and interpretable by design
        - It can handle non-linear datasets with appropriate feature engineering
    """,
    "Decision Tree": """
        - Decision trees are simple to understand and intrepret
        - They are prone to overfitting when they are deep (high variance)
    """,
    "Random Forest": """
        - They have lower risk of overfitting in comparison with decision trees
        - They are robust to outliers
        - They are computationally intensive on large datasets 
        - They are not easily interpretable
    """,
    "Gradient Boosting": """
        - Gradient boosting combines decision trees in an additive fashion from the start
        - Gradient boosting builds one tree at a time sequentially
        - Carefully tuned, gradient boosting can result in better performance than random forests
    """,
    "SVC": """
        - SVMs or SVCs are effective when the number of features is larger than the number of samples
        - They provide different type of kernel functions
        - They require careful normalization   
    """,
}
