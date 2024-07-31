### Heart Disease Prediction Model using multiple Classification Models

**Steps:**
Data is Cleaned and Visualized
After EDA we got to know:
1. Min age for heart disease starts from 28
2. Heart Disease problem occurs more in ages between 53-55
3. Heart Diseases are more commmon in Males
4. The Resting blood pressure values tend to increase by age as the mean and median values are more stable during young age but shows variability by increase in age
5. Most Males and Females have Asymptotic Chest Pain
6. We have calculated collinearilty using corelation matrix and values lesser than 0.1 are considered as having no direct effect on the target variable (Heart Disease) and values greater than 0.8 are also dropped to avoid multicollinearity between features

Corelation matrix was used to identify the best features with the target variable (Heart Disease) and the features with threshold < 0.1 and > 0.8 were dropped to reduce redundancy and improve model performance. 
One hot encoding was performed on the data columns.

### Models Used: 
1. Logistic regression 
2. KNN
3. NB
4. SVM
5. Decision Tree 
6. Random Forest
7. XGBoost
8. GardientBoost
9. AdaBoost
10. LightGBM

A pipeline of these models was created and the best model with hyparamater tuning approach Grid Search came out to be Gradient Boosting and Naive Bayes.

Pickle file was created for Gradient Boosting Classifier model and the results were dumped into the pkl file which was then used in app.py file created to deploy the model on Streamlit web app.
Once the app.py file is ready an app was created on streamlit cloud app and github repo was aligned with it and then a domain was set and the app was finally deployed with the url: https://sufiheartdiseaseprediction.streamlit.app/.
Optional: Docker was also used at some point as I was planning to deploy it on GCP but had no credits for it.
Heroku could also have been used to deploy the model but I used streamlit cloud.
