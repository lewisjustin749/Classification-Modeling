# Classification-Modeling
Classification modeling, data preprocessing, PCA for reducing dimensions, and grid search for hyper parameter optimization on HR company data set.


**Introduction**

In this assignment, you have to perform so some classification tasks on a given data set. Also as part of the classification tasks, you need to perform a number of data preprocessing, PCA for reducting dimensions, and grid search for hypar parameter optimization. For each section, you must have to put the question (with question number) and an appropriate header text as a text cell. And after the header, you can use multiple cells for coding and explaining, and plotting.

**Dataset**

hrdata2.csv 

**Context**

This dataset is a modified version of the dataset you have used for A1. You have done most of the EDA in A1, This dataset is much cleaner with no missing values. Even if you feel to remove any record, don't do it as part of the assignment. But feel free to explore in your own notebook]

A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates. Information related to demographics, education, experience are in hands from candidates signup and enrollment.

This dataset designed to understand the factors that lead a person to leave current job for HR research too. By model(s) that uses the current credentials, demographics, experience data you will predict the probability of a candidate to look for a new job or will work for the company, as well as interpreting affected factors on employee decision.

Although the main objective of this kind of data is the build predictive model to classify a candidate to find the probability whether the candidate will work for them or not, your main goal in this assignment will be to do exploratory data analysis (We have learned various concepts of EDA in the class already in the class). During this process you will learn more about this data, missing values, outliers, correlations, distributions, filling out missing values or removing records, combining features, removing unnecessary features, etc. You will also find other issues such as whether the data set is imbalanced and if yes, how to balance it, etc.

In order to answer the questions, you might need to use your python and EDA knowledge from data camp courses, lecture notes, and for some python syntax and libraries, you might need to google or use python documentation and examples.

**List of Features**

* enrollee_id : Unique ID for the candidate
* city: City code
* city_ development _index : Developement index of the city (scaled)
* gender: Gender of candidate
* relevent_experience: Relevant experience of candidate
* enrolled_university: Type of University course enrolled if any
* education_level: Education level of candidate
* major_discipline :Education major discipline of candidate
* experience: Candidate total experience in years
* company_size: No of employees in current employer's company
* company_type : Type of current employer
* lastnewjob: Difference in years between previous job and current job
* training_hours: training hours completed
* target: 0 – Not looking for job change, 1 – Looking for a job change

**Tasks:**

**1.Load Data and perform basic EDA  (7 pts) [1 + 1 + 1 + 1 + 3]**
* import libraries: pandas, numpy, matplotlib (set %matplotlib inline), matplotlib’s pyplot, seaborn, missingno, scipy’s stats, sklearn (1 pt)
* import the data to a dataframe and show the count of rows and columns (1 pt)
* Show the top 5 and last 5 rows (1 pt)
* Show how many columns have null values
* Plot the count of target and discuss its imbalances and probable issues and solutions

**2.Feature Selection and Pre-processing  (32 pts)**
* Preprocessing City: (4 pts = 1 + 1 + 1 + 1 )
  * Plot #of records per city so that the highest city counts are shown in descending order
  * How many rows belong to the count-wise top 4 cities in total and how many for the remaining? (The plot you have generated in 2.i.i should help you to identify those cities)
  * Replace the city name with city_others if the city name is not within the top 4 city names. (This link might help you: https://stackoverflow.com/questions/31128477/how-to-set-values-based-on-a-list-in-pandas-python (Links to an external site.) (Also, converting the list to a set and then doing a set difference might help you as well)
  * Show some sample data that the records have changed appropriately
  
* Education Level: (4 pts = 1 + 2 + 1)
  * Show the unique values of education level.
  * Replace the value of Education level column like ordinal values, "Graduate" -> 0, Masters->1, and Phd -> 2 (writing a function with the condition and return values  and using apply with the data frame can help you to achieve this.)
  * Show some sample data that the records have changed appropriately
  
* company_size column: (4 pts = 1 + 2 + 1)
  * Show the unique values of the company_size column
  * Change the values of the company_size column from 0 to 7 where e0 is <10 and 7 is 10000+. The order of the numbers should be based on the values of the column-like an ordinary variable. (writing a function with the condition and return values and using apply with the data frame can help you to achieve this.)
  * Show the updated unique values
  
* Last_new_job: (4 pts = 1 + 2 + 1)
  * Show the unique values of the last_new_job column
  * Convert the values of this column to never->0, 1->1,....>4 -->5
  * Show the updated values

* Other columns: (8 pts = 2 + 4 +1 +1)
  * Show the unique values of company_type, major_descipline, enrolled_university, relevant_experience, gender, and updated city column
  * As one-hot encoding is a bit strict, use panda's get_dummies function to create binary columns for the values of the following columns:
    * company_tye
    * major_descipline
    * enrolled_university
    * relevant_eperience
    * gender
    * updated city column
  * Show the top 5 and last 5 rows to show that the table has changed [You must set this first before showing the data frame as many columns will be hidden due to the   large number of columns: pd.set_option('display.max_columns', None)
  * Also, show the shape of the table
* Drop the enrollee_id and any duplicate columns (if you have multiple city column one with actual and one with updated, then remove the actual one) (2 pts)

* Feature Scaling: (5 pts = 2.5 + 2.5)
  * Use sklearn.preprocessing's MinMaxScaler to perform min max scaling to all the columns (see documentation on how to use it)
  * Show sample records that show some the scaled records
* Move the target column to the last column of the data frame and show that it has changed

**3.X/Y and Training/Test Split with stratified sampling and SMOTE (total 15 pts = 2 + 1.5 + 4 + 1.5 + 4 +  2)**
* Copy all the features into X and the target to Y
* Show the ratio of 1 and 0 in Y
* Use sklearn's train_test_split to split the data set into training and test sets. There should be 30% records in the test set. The random_stat should be 0. As we want to have the same ratio of 0 and 1 in the test set, use the stratify parameter to the Y.   
* Show the ratio of 1 and 0 in y_train and then y_test
* Rebalance:
  * Use imblearn's SMOTE to balance the x_train (Help link: https://imbalanced-learn.org/stable/over_sampling.html (Links to an external site.)  (section 2.1.2)
* Show the ratio of 0 and 1 in Y_train after rebalancing. (do you have 50% of each class now?)

**4.PCA and Logistic Regression (20 pts = 7 + 4 +2 +2 + 2 + 3)**
* As we have many features now, we would like to do principal component analysis (you have learned it in datacamp). As part of it, create pipeline to find how many dimensions give you the best logistic regression model. You can follow this link: https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/ (Links to an external site.) (consider using the code right before the plot). But you need to use our balanced training set in this experiment. Also, number of features should be based on how many maximum features do we have so far. This question should produce a plot and based on that you need to decide how many features would you like to use.
* Based on the number of features chosen in the above step, use the test set to evaluate the model for accuracy (the code right after the plot can give you an idea about it.). Use sklearn.metrics import accuracy_score for accuracy (the google colab link in the classification module should help with it)
* Show the confusion matrix and interpret the numbers in the confusion matrix (the google colab link in the classification module should help with it)
* Show precision, recall, and f1 score ((the google colab link in the classification module should help with it)). Note that all of these scores should be calculated based on the test set and predicted result for the test set
* Plot ROC curve and find AUC (the same google colab link should help you)
* Plot precision-recall curve for different thresholds and discuss the plot

**5.Softmaxt regression: 3 pts**
* How softmax regression is related to logistic regression? What library can you use for softmax regression?

**6.KNN (Always use rebalanced training set for training, if it is not specified which training set to use) (25 pts = 4 +4 +3 +2 +4 +4 +4)**
* Use sklearn's KNN classifier to train (with k=  10) and predict the model based on the unbalanced training set (the training set before rebalancing) and test it and show the confusion matrix and classification report
* Use sklearn's KNN classifier to train (with k=  10) and predict the model based on the rebalanced training set and test it and show the confusion matrix and classification report
* Use grid search to tune the following hyperparameters of KNN: number of neighbors (between 1 and 20), weights  (uniform or distance), and metrics (Euclidean, Manhattan, or Minkowski)istance) to use for KNN. While creating an instance of GridSearchCV, use multiple evaluation metrics such as AUC and accuracy based on the example available at Link to sklearn (Links to an external site.). Also some helpful links and codes: https://github.com/oguzhankir/Hyperparameter_Tuning/tree/main/Knn_tuning (Links to an external site.)  and
https://www.youtube.com/watch?v=TvB_3jVIHhg (Links to an external site.)
 
* The above grid search process can take a couple of minutes. After completing the process, print the best_params_
* Based on the result from grid search, use the parameters to train a model, test it with test set, and then print the confusion matrix and classification report. Also, show the AUC of ROC.
* Use PCA and based on that train model, test it and then print the confusion matrix and classification report. Also, show the AUC of ROC.
* A short discussion on the 4 models and their differences.

**7.Naive Bayes (15 pts = 7.5 +7.5)**
* Train a model with GaussianNB, test it and then print the confusion matrix and classification report. Also, plot ROC curve and show the AUC of ROC, and the count of the number of misclassification.
* Train a model with CategoricalNB, test it and then print the confusion matrix and classification report. Also, plot ROC curve, and show the AUC of ROC and the count of the number of misclassification.

**8.Support Vector Machine  (20 pts = 10 + 10)**
* Build a support vector machine model using SVC. Use grid search to tune some parameters and then based on that show the best parameters found
* Test the model and print the confusion matrix and classification report. Also, plot ROC curve and show the AUC of ROC, and the count of the number of misclassification.

**9.Decision Tree (25 pts = 11 +10 + 4)**
* Build a decision tree model using sklearns DecisionTreeClassifier. Use the unbalanced training set, entropy as the criterion. Try with different max_depth (or use grid search). After building model, test it and print the confusion matrix and classification report. Also, plot ROC curve and show the AUC of ROC, and the count of the number of misclassification. Show the decision tree. (you can simply import tree from sklearn and call tree.plot_tree with your model and the call plt.show. At the beginning of this process, use plt.figure to change the figsize)
* Perform the same tasks as 9.1 with the balanced training set
* Discuss any difference and also discuss part of the tree of 9.2

**10.Random Forest (15 pts = 8 + 1 + 6)**
* Use grid search to tune the max_depth, min_samples_leaf, and n_estimators  (helpful link:https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/) it may take about 5 minutes
* Print the best estimator
* Train the model. After building the model, test it and print the confusion matrix and classification report. Also, plot ROC curve and show the AUC of ROC, and the count of the number of misclassification.

**11.Boosting Algorithms (10 pts= 5 + 5)**
* Train an AdaBoostClassifier model with some manual/grid search-based parameters and then test it and then print the confusion matrix and classification report. Also, plot ROC curve and show the AUC of ROC, and the count of the number of misclassification.
* Do the same for Gradient BoostingClassifier

Helpful links: 
https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/#:~:text=Types%20of%20Boosting%20Algorithms&text=AdaBoost%20(Adaptive%20Boosting),XGBoost  (Links to an external site.) 
Another link: https://www.machinelearningplus.com/machine-learning/an-introduction-to-gradient-boosting-decision-trees/  (Links to an external site.) 

**12.Finally, briefly discuss your finding such as which model could be most suitable for this given scenario and what could be your future work based on this experiment. (10 pts)**
