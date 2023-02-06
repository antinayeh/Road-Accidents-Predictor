# Road Accident Predictor

### Team Members 
* Joyce Wang
* Antina Yeh
* Norris Chen

### Tools 
* Python
* Pandas
* Scikit-learn
* Seaborn
* Matplotlib
* NumPy
* Plotly

## Summary
Explored a variety of machine learning models that identifies factors contributing to road accidents and predicts the likelihood of future accidents based on those factors. Project performs data processing, feature engineering, and exploratory data analysis on U.S. accident dataset. Models explored include logistic regression, KNN, and decision tree models. The final decision tree model has a training accuracy of 0.66 and f1 score of 0.65.

## Motivation
The United States has one of the highest rates of road accidents in the world, with over 40,000 fatalities and millions of injuries each year. These accidents have a devastating impact on individuals, families, and communities, and they also have a significant economic cost, with billions of dollars in medical expenses and lost productivity.

One way to address this problem is to develop a road accidents model that can help identify the factors that contribute to accidents and predict the likelihood of future accidents in different areas. This model could be used by policymakers and transportation agencies to target interventions and invest in safety measures that can prevent accidents and save lives.

The motivation for designing such a model is twofold. First, it would help to better understand the causes of road accidents and identify opportunities for intervention. By analyzing data on factors such as weather, road conditions, and road obstacles, the model could provide insights into the underlying causes of accidents and help identify interventions that can reduce their frequency.

Second, the model could be used to predict the likelihood of accidents in different areas, which could help to prioritize safety investments and allocate resources more effectively. By providing a quantitative assessment of the risk of accidents in different areas, the model could help policymakers and transportation agencies make more informed decisions about where to invest in safety measures such as improved signage, enhanced road maintenance, and driver education programs.

Overall, the motivation for designing a road accidents model is to help prevent accidents, save lives, and reduce the economic cost of accidents in the United States. By providing insights into the causes of accidents and predicting their likelihood in different areas, such a model could play a valuable role in improving road safety and protecting the public.

## Project Components  
* Data processing
* Feature engineering
* Exploratory Data Analysis
* Geography-based analysis
* Machine learning modeling


## Conclusion
### EDA
From the exploratory data analysis, we were able to better understand our dataset and gain some insights about traffic accidents in the United States.

First, we saw that California and Florida had the most number of accidents out of all the states and that December had the most number of accidents out of all the months. We also saw how time of the day and day of the week relates to accident counts, with the rush hour times having drastically more accidents. These findings reiterate the fact that higher traffic flows naturally leads to more traffic accidents, and while there may be other factors like visibility, weather, or road features that could impact traffic accident frequency, ultimately, the biggest influencing factor is the amount of traffic on the road.

Looking at other factors like daylight, weather, road features and how they relate to the count and severity of traffic accidents, we saw that while weather by itself doesn’t seem to have a strong impact on the number of traffic accidents, certain weather such as snow when combined with poor visibility at night could potentially increase the chance of a traffic accident. Additionally, while weather might not have a strong impact on increasing or decreasing accident counts, it appears to have a slight relationship with the severity of accidents seeing that severity 1 and 2 accidents follow a similar weather pattern distribution whereas severity 3 and 4 have a more similar distribution. We also found that the majority of the accidents with a severity level 1 happened near an obstacle, which suggests that certain road features may influence the severity level of accidents. However, the same cannot be said about the relationship between obstacles and weather or time of day, since we couldn't find any conclusive relationship in our EDA.

Overall, while there are some minimal insights from our EDA portion, it seems as if many of the features are intercorrelated with each other, and that there isn’t one specific feature that has a great impact on accident count or severity level.

Taking a further look at the correlation heatmap further demonstrates that there isn’t one specific feature that relates to severity level of the accidents. The features that show a strong correlation in our heatmap are features that are inherently related to one another such as longitude and latitude, traffic signal and crossing, and height and pressure. Thus, while we were able to find some relationships between our features, we weren’t able to find any clear relationship between how a feature may impact the severity of accidents.

### Modeling
From the modeling data, we could see that there are stark improvements in the modeling that we did. From the beginning, we had a dummy classifier that had an accuracy score of 0.25, which makes sense because the labels in the target feature is evenly divided up by 4. By doing two iterations with different number of data points and different parameters, we came to the conclusion below:

* Logistic Regression: The first result of logistic regression returned an accuracy score of 0.44 and a f1 score of 0.43. This is a bit better than just randomly guessing, but its still severely underfitted. To improve our score, we used more data points in the second iteration and also 6 new categorical features that added to the complexity of the model. We can see an improvement in the second iteration where the accuracy score is 0.48 and the f1 score is also 0.48.

* KNN: The first result of KNN returned an accuracy score of 0.52 and an f1 score of 0.51. However, it also had a training accuracy of 0.62, which means the model is overfitting. However, we cannot deny that this model outperform logistic regression by a bit. To fix the overfitting in the model, our second iteration called for the n_neighbors to be higher. Increasing n_neighbors will decrease the noise that comes from small nearest neighbor values. Also including the categorical data means the model will be more complex. We found that the second iteration did indeed improve the model to a 0.57 accuracy and 0.56 f1 score with the training accuracy dropping to 0.6.

* Decision Trees: The initial result of decsion trees was a 0.63 accuracy score and a 0.63 f1 score. However, this came with a training accuracy of 0.69, which proved to be a little overfitted. To combat this overfitting, we tried to use random forest, but unfortunately got a result that was even more overfitted (training: 0.93 test: 0.68). For our second iteration of Decision Trees, we have new categorical data that helped smooth out the overfittedness of our original model, and end up with the best score of 0.66 testing accuracy and 0.65 f1 score.

In conclusion, the decision tree model presented in our project best models the US Road Accidents Dataframe, with a training accuracy of 0.66 and f1 score of 0.65.

### Future Direction
There are quite a few directions this project can go from here, however, some of the biggest improvements this project can make are listed below:

* Forward Feed Neural Network: A neural network is definitely possible with this dataset, especially with a dataset this big. A neural net would help with the overall underfitting of the models in this project. However, we do have to watch out for the computing power and the time it takes to run a neural network. Otherwise, a neural network would be a great addition to this project.

* Gathering more data: Another way to solve the overall underfitted models in our project (to increase complexity) is to include more features in our dataset to run the machine learning models on. We could potentially find other datasets such as car density in each city as one of the features or the even the type of car that was in the accident. All these new features can add to the complexity of the model and help it get a better accuracy and f1 score

* Better Hardware/Optimized Code: A problem we repeatedly encountered in this project was the fact that colab would keep crashing on us because there isn't enough RAM. We made some adjustments in the project like sampling only a small subset of data to be used in machine learning. However, with more RAM or a more optimized code, we would not have to worry about that.

* A prediction tool: We could potentially even turn this project into a detection tool for predicting accidents in certain cities near certain features. This would require a lot more domain knowledge and more up to date data from around the US.
