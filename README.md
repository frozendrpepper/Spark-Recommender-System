# Collaborative Filtering Recommender System using Apache Spark

Based on a sample YELP dataset with columns of userID, businessID and user rating, build 4 different recommender systems
1) ALS (model based) recommender system - (Matrix model approach that is available in the Spark library)
2) User based collaborative filtering recommender system
3) Item based collaborative filtering recommender system
4) Hybrid recommender system (Uses Item based collaborative filtering with extra information provided in busfeatures.json)


## Note about working environment

* The code was developed on Anaconda Package based on Python version 3.6.9 and pyspark version 2.4.4.


## What is Collaborative Filtering?

### Pearson Correlation
![alt text](https://cdn1.byjus.com/wp-content/uploads/2019/02/Correlation-Coefficient-Formula.png)  
(Source: Google Image Search (https://cdn1.byjus.com/wp-content/uploads/2019/02/Correlation-Coefficient-Formula.png))

### Matrix Representing User/Item Rating
![alt text](https://github.com/frozendrpepper/Spark-Recommender-System/blob/master/user_item_matrix.png?raw=true)  
(Source: UCI Math77b Lecture 12 slide 7 (https://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2002%20-%20Collaborative%20recommendation.pdf))

The basic idea is that to find an unknown rating, either find similar users or similar items and compute the Pearson Correlation which then can be used to find the predicted rating. In the above example, to predict Alice's rating on Item 5, we could either user other users (rows) who have high positive Pearson Correlation values with Alice
to compute the rating or find other items (columns) that have high positive Pearson Correlation values with Item 5 to compute the rating.

## File Explanation

* yelp_train.csv -> A small sample of the YELP sample dataset that was used for analysis. The original dataset contains approximate 500,000 rows of data point.
* busfeatures.json -> Extra information that is used to build the hybrid system.
* task2.py -> Contains all 4 implementations of the recommender systems. To execute the program, a user needs to provide train dataset,
              additional bus_features data containing extra information for hybrid system, test dataset, which case (1~4) and output file to write the result to
              Ex) python3 task2.py train.csv bus_features.json test.csv 1 output.csv


## Result
Each model's performance is measured by the RMSE (root mean squared error). Lower RMSE implies more accurate result. A separate validation dataset, not shown in the current repository, is used to analyze the performance of each model

1) ALS - 
2) User - 
3) Item -
4) Hybrid - 


## Useful References

Resources for Face Recognition Deep Neural Network
* [UT Dallas CS6375 Lecture of Collaborative Filtering](https://personal.utdallas.edu/~nrr150130/cs6375/2015fa/lects/Lecture_23_CF.pdf)
* [UCI Math77b Lecture on Collaborative Filtering](https://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2002%20-%20Collaborative%20recommendation.pdf)
