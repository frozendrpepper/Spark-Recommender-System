# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:57:44 2019

@author: ckang
"""

from __future__ import division
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import time
import json
import sys
from math import sqrt

'''Take command line arguments'''
train_file_name = sys.argv[1]
bus_feature = sys.argv[2]
test_file_name = sys.argv[3]
case = int(sys.argv[4])
output_file_name = sys.argv[5]

def writeFilev1(result, output_file_name, user_map, business_map):
    with open(output_file_name, 'w') as theFile:
        theFile.write("user_id, business_id, prediction")
        theFile.write("\n")
        for i in range(len(result)):
            user = user_map[result[i][0]]
            business = business_map[result[i][1]]
            theFile.write(user)
            theFile.write(",")
            theFile.write(business)
            theFile.write(",")
            theFile.write(str(result[i][2]))
            theFile.write("\n")

def writeFilev2(result, output_file_name):
    with open(output_file_name, 'w') as theFile:
        theFile.write("user_id, business_id, prediction")
        theFile.write("\n")
        for i in range(len(result)):
            user = result[i][0][0]
            business = result[i][0][1]
            theFile.write(user)
            theFile.write(",")
            theFile.write(business)
            theFile.write(",")
            theFile.write(str(result[i][1]))
            theFile.write("\n")

'''https://stackoverflow.com/questions/26679272/how-to-use-argv-with-spyder'''
##############################################################################
# Case 1 Model Based (ALS) CF
##############################################################################
if case == 1:
    '''Model Based CF using ALS'''
    start = time.time()
    conf = SparkConf().setMaster("local[*]").setAppName("ALS")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")
    
    training = sc.textFile(train_file_name)
    test = sc.textFile(test_file_name)
    
    '''Remove Header'''
    train_header = training.first()
    training_filtered = training.filter(lambda row: row != train_header)
    test_header = test.first()
    test_filtered = test.filter(lambda row: row != test_header)
    
    '''Find mapping for user and business id'''
    training_map = training_filtered.map(lambda x: x.split(','))
    test_map = test_filtered.map(lambda x: x.split(','))
    
    user_count = 0
    business_count = 0
    user_map = {}
    business_map = {}
    for user, business, rating in training_map.toLocalIterator():
        if user not in user_map:
            user_map[user] = user_count
            user_map[user_count] = user
            user_count += 1
        if business not in business_map:
            business_map[business] = business_count
            business_map[business_count] = business
            business_count += 1
    
    for user, business, rating in test_map.toLocalIterator():
        if user not in user_map:
            user_map[user] = user_count
            user_map[user_count] = user
            user_count += 1
        if business not in business_map:
            business_map[business] = business_count
            business_map[business_count] = business
            business_count += 1
    
    '''Convert training and test dataset in terms of int and apply Rating'''
    training_map_hash = training_map.map(lambda x: Rating(user_map[x[0]], business_map[x[1]], float(x[2])))
    test_map_hash = test_map.map(lambda x: Rating(user_map[x[0]], business_map[x[1]], float(x[2])))
    
    '''Train ALS model'''
    rank = 4
    numIterations = 5
    lamb = 0.3
    model = ALS.train(training_map_hash, rank, numIterations, lamb)
    
    '''Evalulate the model on test dataset'''
    test_data = test_map_hash.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(test_data).map(lambda x: (x[0], x[1], x[2]))
    #ratesAndPreds = test_map_hash.map(lambda x: ((x[0], x[1]), x[2])).join(predictions)
    #MSE = ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean()
    #RMSE = sqrt(MSE)
    #print("Root Mean Squared Error = " + str(RMSE))
    result = predictions.collect()
    
    writeFilev1(result, output_file_name, user_map, business_map)
    end = time.time()
    interval = end - start
##############################################################################
# Case 2 User Based CF
##############################################################################
elif case == 2:
    '''User based CF using Spark and Python'''
    def mapDict(vals):
        return_dict = {}
        for val in vals:
            return_dict[val[0]] = val[1]
        return return_dict

    def findAvg(vals):
        '''Find average value for each row'''
        count = 0
        tot = 0
        for item, rating in vals.items():
            tot += rating
            count += 1
        average = float(tot) / count
        
        for item, rating in vals.items():
            vals[item] = rating - average
        vals['row_avg'] = average
        return vals
    
    start = time.time()
    conf = SparkConf().setMaster("local[*]").setAppName("ITEM_CF")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")
    
    train = sc.textFile(train_file_name)
    test = sc.textFile(test_file_name)
    
    train_header = train.first()
    train_filtered = train.filter(lambda row: row != train_header)
    test_header = test.first()
    test_filtered = test.filter(lambda row: row != test_header)
    
    train_rdd = train_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[0], (line_split[1], float(line_split[2]))))
    test_rdd = test_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[0], line_split[1], float(line_split[2])))
    
    '''At this end of this section the data is in form (user_id, {item1: count1, item2:count2, item3:count3 etc... row_avg: average})'''
    training_group = train_rdd.groupByKey().mapValues(list)
    training_group_dict = training_group.mapValues(mapDict)
    training_group_dict_avg = training_group_dict.mapValues(findAvg)
    
    '''Add 1 as a key to all data so we can compile. At this end of this section
    we have (1, matrix_representation_of_training_data)
    In other words, 
    (1, [(user_id1, {item1:count1, item2:count2})
         (user_id2, {item1:count1, item2:count2})
         (user_id3, {item1:count1, item2:count2})]) '''
    training_group_dict_one = training_group_dict_avg.map(lambda x: (1, x))
    training_one_reduce = training_group_dict_one.groupByKey().mapValues(list).map(lambda x: x[1])
    
    '''The problem I encountered here is that join operation is massively expensive
    and is causing the program to crash. Instead, collect the training data result
    and use it as an input to filter out information we need for further computation'''
    training_data_compile = training_one_reduce.collect()
    training_data_compile = training_data_compile[0]
    training_compile_dict = {}
    for i in range(len(training_data_compile)):
        training_compile_dict[training_data_compile[i][0]] = training_data_compile[i][1]
    
    '''Get item to user dictionary so we can use it to find rows containing certain item faster'''
    item_to_user_pre = train_rdd.map(lambda x: (x[1][0], x[0])).groupByKey().mapValues(list)
    item_to_user_compile = item_to_user_pre.collect()
    item_to_user_dict = {}
    for i in range(len(item_to_user_compile)):
        item_to_user_dict[item_to_user_compile[i][0]] = set(item_to_user_compile[i][1])
    
    test_data = test_rdd.collect()
    RMSE_tmp = 0
    tmp_result = []
    pearson_threshold = 0.3
    random_pred = 0
    upper_limit = 150
    for test in test_data:
        '''Get all the rows corresponding to cur user and item of test dataset'''
        cur_user, cur_item = test[0], test[1]
        filtered_train = {}
        if cur_item not in item_to_user_dict or cur_user not in training_compile_dict:
            '''If it's an unseen business id, assign some random prediction'''
            prediction = 2.5
            random_pred += 1
        else:
            filtered_train[cur_user] = training_compile_dict[cur_user]
            cur_user_info = filtered_train[cur_user]
            '''Get a list of user_id who contains the current item'''
            row_set = item_to_user_dict[cur_item]
            for row in row_set:
                if len(training_compile_dict[row]) < upper_limit:
                    filtered_train[row] = training_compile_dict[row]
                
            '''Compute Pearson for each row'''
            predict_num = 0
            predict_den = 0
            for user, item_list in filtered_train.items():
                if user != cur_user:
                    num = 0
                    den1 = 0
                    den2 = 0
                    for item, rating in item_list.items():
                        if item in cur_user_info and item != cur_item and item != 'row_avg':
                            num += rating * cur_user_info[item]
                            den1 += rating**2
                            den2 += (cur_user_info[item])**2
                    denom = sqrt(den1) * sqrt(den2)
                    if num == 0 or denom == 0:
                        pearson = 0
                    else:
                        pearson = float(num) / denom
                    if pearson > pearson_threshold:
                        predict_num += (filtered_train[user][cur_item]) * pearson
                        predict_den += abs(pearson)
            if predict_num == 0 or predict_den == 0:
                prediction = cur_user_info['row_avg']
            else:
                prediction = cur_user_info['row_avg'] + float(predict_num) / predict_den
                prediction = (prediction + cur_user_info['row_avg']) / 2.0 
        '''Save the results which consists of user_id, business_id, ground truth and predicted'''
        tmp_result.append((test, prediction))
        '''Compile results for final MSE computation'''
        RMSE_tmp += (test[2] - prediction)**2
    RMSE = sqrt(RMSE_tmp / len(test_data))
    #print("RMSE: ", RMSE)
    writeFilev2(tmp_result, output_file_name)
    end = time.time()
    duration = end - start
##############################################################################
# Case 3 Item Based CF
##############################################################################
elif case == 3:
    '''Item Based CF using Spark and Python'''
    def mapDict(vals):
        return_dict = {}
        for val in vals:
            return_dict[val[0]] = val[1]
        return return_dict
    
    def findAvg(vals):
        '''Find average value for each row'''
        count = 0
        tot = 0
        for item, rating in vals.items():
            tot += rating
            count += 1
        average = float(tot) / count
        
        for item, rating in vals.items():
            vals[item] = rating - average
        vals['row_avg'] = average
        return vals
    
    start = time.time()
    conf = SparkConf().setMaster("local[*]").setAppName("ITEM_CF")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")
    
    '''Toy dataset used to test the algorithm'''
    train = sc.textFile(train_file_name)
    test = sc.textFile(test_file_name)
    
    train_header = train.first()
    train_filtered = train.filter(lambda row: row != train_header)
    test_header = test.first()
    test_filtered = test.filter(lambda row: row != test_header)
    
    train_rdd = train_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[1], (line_split[0], float(line_split[2]))))
    test_rdd = test_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[1], line_split[0], float(line_split[2])))
    
    '''At this end of this section the data is in form (item_id, {user1: count1, user2:count2, user3:count3 etc... row_avg: average})'''
    training_group = train_rdd.groupByKey().mapValues(list)
    training_group_dict = training_group.mapValues(mapDict)
    
    '''Subtract average of the row from each count i.e) (item_id, {user1:count1 - average, user2:count2 - average etc... row_avg: average})'''
    training_group_dict_avg = training_group_dict.mapValues(findAvg)
    
    '''Add 1 as a key to all data so we can compile. At this end of this section
    we have (1, matrix_representation_of_training_data)
    In other words, 
    (1, [(item_id1, {user1:count1 - row_avg, user2:count2 - row_avg})
         (item_id2, {user2:count1 - row_avg, user4:count2 - row_avg})
         (item_id3, {user1:count1 - row_avg, user3:count2 - row_avg})]) '''
    training_group_dict_one = training_group_dict_avg.map(lambda x: (1, x))
    training_one_reduce = training_group_dict_one.groupByKey().mapValues(list).map(lambda x: x[1])
    
    '''The problem I encountered here is that join operation is massively expensive
    and is causing the program to crash. Instead, collect the training data result
    and use it as an input to filter out information we need for further computation'''
    training_data_compile = training_one_reduce.collect()
    training_data_compile = training_data_compile[0]
    training_compile_dict = {}
    for i in range(len(training_data_compile)):
        training_compile_dict[training_data_compile[i][0]] = training_data_compile[i][1]
    
    '''Get user to item dictionary so we can use it to find rows containing certain item faster'''
    item_to_user_pre = train_rdd.map(lambda x: (x[1][0], x[0])).groupByKey().mapValues(list)
    item_to_user_compile = item_to_user_pre.collect()
    item_to_user_dict = {}
    for i in range(len(item_to_user_compile)):
        item_to_user_dict[item_to_user_compile[i][0]] = set(item_to_user_compile[i][1])
    
    test_data = test_rdd.collect()
    RMSE_tmp = 0
    tmp_result = []
    pearson_threshold = 0.3
    random_pred = 0
    upper_limit = 150
    lower_limit = 15
    for test in test_data:
        '''Get all the rows corresponding to cur user and item of test dataset'''
        cur_item, cur_user = test[0], test[1]
        filtered_train = {}
        if cur_item not in training_compile_dict or cur_user not in item_to_user_dict:
            '''If it's an unseen business id, assign some random prediction'''
            prediction = 2.5
            random_pred += 1
        else:
            '''We want to attach row corresponding to current item'''
            filtered_train[cur_item] = training_compile_dict[cur_item]
            cur_item_info = filtered_train[cur_item]
            '''Get a list of user_id who contains the current item'''
            row_set = item_to_user_dict[cur_user]
            for row in row_set:
                if len(training_compile_dict[row]) > lower_limit and len(training_compile_dict[row]) < upper_limit:
                    filtered_train[row] = training_compile_dict[row]
                
            '''Compute Pearson for each row and add to the final result if Pearson
            passes the threshold value'''
            predict_num = 0
            predict_den = 0
            for item, user_list in filtered_train.items():
                if item != cur_item:
                    num = 0
                    den1 = 0
                    den2 = 0
                    for user, rating in user_list.items():
                        if user in cur_item_info and user != cur_user and user != 'row_avg':
                            num += rating * cur_item_info[user]
                            den1 += rating**2
                            den2 += (cur_item_info[user])**2
                    denom = sqrt(den1) * sqrt(den2)
                    if num == 0 or denom == 0:
                        pearson = 0
                    else:
                        pearson = float(num) / denom
                    if pearson > pearson_threshold:
                        predict_num += (filtered_train[item][cur_user] + filtered_train[item]['row_avg']) * pearson
                        predict_den += abs(pearson)
            if predict_num == 0 or predict_den == 0:
                prediction = cur_item_info['row_avg']
            else:
                prediction = float(predict_num) / predict_den
                prediction = (prediction + cur_item_info['row_avg']) / 2.0 
        '''Save the results which consists of user_id, business_id, ground truth and predicted'''
        tmp_result.append(((test[1], test[0], test[2]), prediction))
        '''Compile results for final MSE computation'''
        RMSE_tmp += (test[2] - prediction)**2
    RMSE = sqrt(RMSE_tmp / len(test_data))
    #print("RMSE: ", RMSE)
    writeFilev2(tmp_result, output_file_name)
    end = time.time()
    duration = end - start
##############################################################################
# Part 4 Hybrid RC
##############################################################################
elif case == 4:
    '''Hybrid Recommendation using item based CF and business json'''
    def parseJson(line):
        json_data = json.loads(line)
        return (json_data["business_id"], json_data["stars"], json_data["review_count"])
        
    def mapDict(vals):
        return_dict = {}
        for val in vals:
            return_dict[val[0]] = val[1]
        return return_dict
    
    def findAvg(vals):
        '''Find average value for each row'''
        count = 0
        tot = 0
        for item, rating in vals.items():
            tot += rating
            count += 1
        average = float(tot) / count
        
        for item, rating in vals.items():
            vals[item] = rating - average
        vals['row_avg'] = average
        return vals
    
    start = time.time()
    conf = SparkConf().setMaster("local[*]").setAppName("ITEM_CF")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("OFF")
    
    '''Toy dataset used to test the algorithm'''
    train = sc.textFile(train_file_name)
    test = sc.textFile(test_file_name)
    bus_json = sc.textFile(bus_feature)
    
    train_header = train.first()
    train_filtered = train.filter(lambda row: row != train_header)
    test_header = test.first()
    test_filtered = test.filter(lambda row: row != test_header)
    
    '''Cleanse all data so they can be further processed'''
    train_rdd = train_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[1], (line_split[0], float(line_split[2]))))
    test_rdd = test_filtered.map(lambda x: x.split(',')).map(lambda line_split: (line_split[1], line_split[0], float(line_split[2])))
    bus_rdd = bus_json.map(parseJson)
    
    '''At this end of this section the data is in form (item_id, {user1: count1, user2:count2, user3:count3 etc... row_avg: average})'''
    training_group = train_rdd.groupByKey().mapValues(list)
    training_group_dict = training_group.mapValues(mapDict)
    
    '''Subtract average of the row from each count i.e) (item_id, {user1:count1 - average, user2:count2 - average etc... row_avg: average})'''
    training_group_dict_avg = training_group_dict.mapValues(findAvg)
    
    '''Add 1 as a key to all data so we can compile. At this end of this section
    we have (1, matrix_representation_of_training_data)
    In other words, 
    (1, [(item_id1, {user1:count1 - row_avg, user2:count2 - row_avg})
         (item_id2, {user2:count1 - row_avg, user4:count2 - row_avg})
         (item_id3, {user1:count1 - row_avg, user3:count2 - row_avg})]) '''
    training_group_dict_one = training_group_dict_avg.map(lambda x: (1, x))
    training_one_reduce = training_group_dict_one.groupByKey().mapValues(list).map(lambda x: x[1])
    
    '''The problem I encountered here is that join operation is massively expensive
    and is causing the program to crash. Instead, collect the training data result
    and use it as an input to filter out information we need for further computation'''
    training_data_compile = training_one_reduce.collect()
    training_data_compile = training_data_compile[0]
    training_compile_dict = {}
    for i in range(len(training_data_compile)):
        training_compile_dict[training_data_compile[i][0]] = training_data_compile[i][1]
    
    '''Get user to item dictionary so we can use it to find rows containing certain item faster'''
    item_to_user_pre = train_rdd.map(lambda x: (x[1][0], x[0])).groupByKey().mapValues(list)
    item_to_user_compile = item_to_user_pre.collect()
    item_to_user_dict = {}
    for i in range(len(item_to_user_compile)):
        item_to_user_dict[item_to_user_compile[i][0]] = set(item_to_user_compile[i][1])
    
    '''Collect business json information that is gonna be part of the hybrid logic'''
    business_json = bus_rdd.collect()
    business_dict = {}
    for i in range(len(business_json)):
        business_dict[business_json[i][0]] = (business_json[i][1], business_json[i][2])
    
    test_data = test_rdd.collect()
    RMSE_tmp = 0
    tmp_result = []
    pearson_threshold = 0.3
    random_pred1 = 0
    upper_limit = 150
    lower_limit = 15
    for test in test_data:
        '''Get all the rows corresponding to cur user and item of test dataset'''
        cur_item, cur_user = test[0], test[1]
        filtered_train = {}
        if cur_item not in training_compile_dict and cur_item in business_dict:
            '''In case business Id does not exist in the training dataset, use 
            bus_json to extract the rating. This is the hybrid component'''
            random_pred1 += 1
            prediction = business_dict[cur_item][0]
        elif cur_item not in training_compile_dict and cur_item in business_dict:
            prediction = 2.5
        elif cur_user not in item_to_user_dict:
            '''If it's a new user, assign some random rating'''
            prediction = 2.5
        else:
            '''We want to attach row corresponding to current item'''
            filtered_train[cur_item] = training_compile_dict[cur_item]
            cur_item_info = filtered_train[cur_item]
            '''Get a list of user_id who contains the current item'''
            row_set = item_to_user_dict[cur_user]
            for row in row_set:
                if len(training_compile_dict[row]) > lower_limit and len(training_compile_dict[row]) < upper_limit:
                    filtered_train[row] = training_compile_dict[row]
                
            '''Compute Pearson for each row and add to the final result if Pearson
            passes the threshold value'''
            predict_num = 0
            predict_den = 0
            for item, user_list in filtered_train.items():
                if item != cur_item:
                    num = 0
                    den1 = 0
                    den2 = 0
                    for user, rating in user_list.items():
                        if user in cur_item_info and user != cur_user and user != 'row_avg':
                            num += rating * cur_item_info[user]
                            den1 += rating**2
                            den2 += (cur_item_info[user])**2
                    denom = sqrt(den1) * sqrt(den2)
                    if num == 0 or denom == 0:
                        pearson = 0
                    else:
                        pearson = float(num) / denom
                    if pearson > pearson_threshold:
                        predict_num += (filtered_train[item][cur_user] + filtered_train[item]['row_avg']) * pearson
                        predict_den += abs(pearson)
            if predict_num == 0 or predict_den == 0:
                prediction = (cur_item_info['row_avg'] + business_dict[cur_item][0]) / 2.0
            else:
                prediction = float(predict_num) / predict_den
                prediction = (prediction + cur_item_info['row_avg'] + business_dict[cur_item][0]) / 3.0 
        '''Save the results which consists of user_id, business_id, ground truth and predicted'''
        tmp_result.append(((test[1], test[0], test[2]), prediction))
        '''Compile results for final MSE computation'''
        RMSE_tmp += (test[2] - prediction)**2
    RMSE = sqrt(RMSE_tmp / len(test_data))
    #print("RMSE: ", RMSE)
    writeFilev2(tmp_result, output_file_name)
    end = time.time()
    duration = end - start