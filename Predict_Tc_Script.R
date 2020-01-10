
##################################################################################################

#               ML2 Project R script (Predicting critical temperature of semiconductor)                                  

##################################################################################################


#load the  data set
library(readr)
dataset_sc <- read_csv("ml2/Super_Conductor_Project/train.csv")
View(dataset_sc)

# Look at the data
str(train)
#rm(train)
rm(sc_test)
rm(sc_train)

# Set seed 
set.seed(1)

#Split your data into three parts: 60% training data, 20% validation data and 20% test data. 

split_sc <- sample(1:3, size = nrow(dataset_sc), prob = c(0.6, 0.20, 0.20), replace = TRUE)

# Create a train, validation and tests from the original data frame 
sc_train <- dataset_sc[split_sc == 1, ]  # subset dataset_sc to training indices only
sc_valid <- dataset_sc[split_sc == 2, ]  # subset dataset_sc to validation indices only
sc_test <- dataset_sc[split_sc == 3, ]   # subset dataset_sc to test indices only

split_sc <- sample(1:2, size = nrow(dataset_sc), prob = c(0.8, 0.20), replace = TRUE)

# Create a train, validation and tests from the original data frame 
sc_train <- dataset_sc[split_sc == 1, ]  # subset dataset_sc to training indices only
#sc_valid <- dataset_sc[split_sc == 2, ]  # subset dataset_sc to validation indices only
sc_test <- dataset_sc[split_sc == 2, ]   # subset dataset_sc to test indices only

##################################################################################################

#                     ML method 1 :Random Forest Tree Regressor

##################################################################################################

####scale the data
sc_train.scale <- cbind( scale(sc_train[,1:81]), sc_train[ ,-c(1:81)] ) 
sc_test.scale <- cbind( scale(sc_test[,1:81]), sc_test[ ,-c(1:81)] ) 


#Random forest is avaliable in randomForest package
install.packages("randomForest")
library(randomForest)

#The default random forest which is availble in randomForest Package performs 500 trees and
#features/3 =26 randomly selected predictor variables at each split. 
#Averaging across all 500 trees provides an OOB MSE = ()

# default RF model training
m1<- randomForest(
  formula = critical_temp ~ .,
  data    = sc_train.scale
)

#plot the model
plot(m1, main = "RandomForest(default)")

#predictions on the test dataset by using m1 model
m1_prediction<-predict(m1,sc_test.scale)
head(sc_test.scale$critical_temp)
head(m1_prediction)

#plot the actual values and predicted values
plot(m1_prediction,sc_test.scale$critical_temp,col=c("black","red"))

#Plotting the model will illustrate the error rate as we average across more trees and shows that 
#our error rate stabalizes with around 100 trees but continues to decrease slowly until 
#around 300 or so trees.

# number of the tree with lowest MSE
which.min(m1$mse)
## [1] 499

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])
## [1] 9.2007

#As the training dataset has around 81 features. But we use 30 features for prediction. so for this
#our fitted model will give us the most important features for the prediction
#randomforest has importance variable
#the higher importance value the more important feature

#important vaiables in dataset 
varImpPlot(m1, main = "Important variables(rf_default)")

#getting important variable names
important_variables=order(-m1$importance)
important_varaible_names=rownames(m1$importance)[important_variables][1:30]
important_varaible_names<-c(important_varaible_names)

#creating the dataset using only important vaiable names for train and test
sc_train.scale<-cbind(sc_train.scale[important_varaible_names], sc_train.scale["critical_temp"] ) 
sc_test.scale<-cbind( sc_test.scale[important_varaible_names], sc_test.scale["critical_temp"] ) 

#fitting the model using the updated dataset
rf <- randomForest(
  formula = critical_temp ~ .,
  data    = sc_train.scale
)
#plot the model
plot(rf, main = "(rf)RandomForest(default)")

# number of the tree with lowest MSE
which.min(rf$mse)
## [1] 491

# RMSE of this optimal random forest
sqrt(rf$mse[which.min(rf$mse)])
## [1] 9.365003

#predictions on the test dataset by using m1 model
rf_prediction<-predict(rf,sc_test.scale)
head(sc_test.scale$critical_temp)
head(rf_prediction)
sqrt(mean((sc_test.scale$critical_temp -rf_prediction)^2))


#fitting a model using test and validati data

# create training and validation data 
valid_split <- initial_split(sc_train.scale, .8)

# training data
sc_train.scale_v2 <- analysis(valid_split)

# validation data
sc_valid.scale <- assessment(valid_split)
x_test <- sc_valid.scale[setdiff(names(sc_valid.scale), "critical_temp")]
y_test <- sc_valid.scale$critical_temp

rf_oob_comp <- randomForest(
  formula = critical_temp ~ .,
  data    = sc_train.scale_v2,
  xtest   = x_test,
  ytest   = y_test
)

plot(sqrt(rf$mse), main = "Root mean square error", xlab = "Number of Trees", ylab = "RMSE")


# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

install.packages("tibble")
library(tibble)


# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")




################## Tuning the model ##########

# mtry: the number of variables to randomly sample as candidates at each split.

# names of features
features <- setdiff(names(sc_train.scale), "critical_temp")

rf_tune <- tuneRF(
  x          = sc_train.scale[features],
  y          = sc_train.scale$critical_temp,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.004,
  trace      = FALSE      # to not show real-time progress 
)
rf_tune[,"mtry"][which.min(rf_tune[,"OOBError"])]



###############Full grid search with ranger###################

#Grid-searching is the process of scanning the data to configure optimal parameters for a 
#given model. Depending on the type of model utilized, certain parameters are necessary. 
#Grid-searching does NOT only apply to one model type. Grid-searching can be applied across
#machine learning to calculate the best parameters to use for any given model. It is important 
#to note that Grid-searching can be extremely computationally expensive and may take your machine
#quite a long time to run. Grid-Search will build a model on each parameter combination possible.
#It iterates through every parameter combination and stores a model for each combination




#To perform a larger grid search across several hyperparameters we’ll need to create
#a grid and loop through each hyperparameter combination and evaluate the model. 
#Unfortunately, this is where randomForest becomes quite inefficient since it does not 
#scale well. Instead, we can use ranger which is a C++ implementation of Brieman’s random 
#forest algorithm and, as the following illustrates, is over 6 times faster than randomForest.

# randomForest speed
system.time(
  ames_randomForest <- randomForest(
    formula = critical_temp ~ ., 
    data    = sc_train, 
    ntree   = 500,
    mtry    = floor(length(features) / 3)
  )
)
##   user      system elapsed 
##   254.300   2.782 257.201 

# ranger speed
install.packages("ranger")
library(ranger)
system.time(
  ames_ranger <- ranger(
    formula   = critical_temp ~ ., 
    data      = sc_train, 
    num.trees = 500,
    mtry      = floor(length(features) / 3)
  )
)

##  user      system  elapsed 
##  174.751   0.892  24.172 
rm(hyper_grid)
# hyperparameter grid search
hyper_grid <- expand.grid(
  mtry = seq(7, ncol(sc_train.scale) * 0.8, 2), 
  node_size = seq(3, 8, 2),
  sampe_size <- c(0.55, 0.632,0.7, 0.8),
  OOB_RMSE   = 0
)


hyper_grid <- expand.grid(
  mtry = seq(7, ncol(sc_train.scale) * 0.8, 2), 
  nodesize = seq(3, 8, 2), 
  sampsize = nrow(sc_train.scale) * c(0.7, 0.8),
  OOB_RMSE   = 0)
# total number of combinations
nrow(hyper_grid)
## [1] 360

library(ranger)

for(i in 1:nrow(hyper_grid)) {
  
  # train model
  rf_ranger <- ranger(
    formula         = critical_temp ~ ., 
    data            = sc_train.scale, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    importance      = 'impurity'
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(rf_ranger$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

which.min(hyper_grid$OOB_RMSE)
##[1] 88
hyper_grid$mtry[88]
## [1] 19
hyper_grid$node_size[88]
## [1] 3
hyper_grid$sampe_size[88]
## [1] 0.8

#fitting the random forest using above hyperparameters

rf_opt <- randomForest(critical_temp ~ ., data=sc_train.scale, mtry=7,
                         importance=TRUE, nodesize = 3, )#ntree

plot(rf_opt)
sqrt(mean(rf_opt$mse))
sqrt(rf_opt$mse[which.min(rf_opt$mse)])
m1
rf_ranger

#making predictions on the test data set using rf_optimal model
rf_opt_predictions <- predict(rf_opt,sc_test.scale)
rf_opt_predictions.resid<-rf_opt_predictions-sc_test.scale$critical_temp
(mean(rf_opt_predictions.resid^2))
head(rf_opt_predictions)
head(sc_test.scale$critical_temp)

##[1] 9.57616

#plot((1:m_opt$ntree),sqrt(m_opt$mse))
plot(m_opt_predictions,sc_test.scale$critical_temp,main = "actual vs predicted values",
     col=c("red","black"),xlab = "predicted values",ylab = "actual values", )
legend("bottomright",legend = c("predicted", "actual"),col = c("red", "black"),bty = "n",
       text.col = c("red","black"))
##################################################################################################

#                     ML method 2 :Neural Networks

##################################################################################################

#prerequisites
library(keras)
library(tensorflow)
library(reticulate)

#add your own python path
reticulate::use_python('/Users/dasyamchandu/opt/anaconda3/envs/tensorflow_env/bin/python')
repl_python()
#the above command will take to the python terminal
#inside python terminal type import tensorflow
#exit the terminal

#creating the instance of keras model
rm(nn)

# creting the model and adding the model with desired no.of layers

nn <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu",
              input_shape = 30) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1)

nn %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error")
)

nn
y_nn<-pull(sc_train.scale, critical_temp)

history <- nn %>% fit(
  as.matrix(sc_train.scale[features]), 
  y_nn, 
  epochs = 500, 
  #batch_size = 32, 
  validation_split = 0.2
)


#plot
library(ggplot2)
plot(history, main="neural network evalutaion")
plot(history, metrics = "mean_absolute_error", smooth = FALSE)

#prediction by using test dataset
y_test_nn<-pull(sc_test.scale, critical_temp)
#nn_evaluation %<-% (nn %>% evaluate(as.matrix(sc_test.scale[features]), y_test_nn, verbose = 0))

nn_test_predictions <- nn %>% predict(as.matrix(sc_test.scale[features]))
#optDataSet <- data.frame(matrix(ncol = 30, nrow = 1))
#comparing first 6 actual and predicted values
head(sc_test.scale$critical_temp)
head(nn_test_predictions[ , 1])

#means square of neural networks
nn.resid<-nn_test_predictions[,1]-sc_test.scale$critical_temp
(mean(nn.resid^2))


#plot actual and predicted values
plot(sc_test.scale$critical_temp,nn_test_predictions[,1],col=c("red","black"))

head(m_opt_predictions)
m_opt_predictions.resid<-m_opt_predictions-sc_test.scale$critical_temp
#mean square error of random forest
(mean(m_opt_predictions.resid^2))




###############################################################################################


library(CHNOSZ)
predict_tc_rf = function(your_material, verbose = F)
{
  tmp = makeup(your_material)
  if ( any( names(tmp) == "Fe") ) tmp_iron = 1 else tmp_iron = 0
  if ( any( names(tmp) == "O") & any( names(tmp) == "Cu") ) tmp_cuprate = 1 else tmp_cuprate = 0
  mat = as.data.frame( t(c(as.numeric(extract( x = your_material, ed = subset_element_data)) , 
                           tmp_iron, tmp_cuprate))  )
  colnames(mat) = colnames(train)[-ncol(train)]
  #######################################
  mat <- mat[features]
  #print(mat)
  #mat <- scale(mat[,1:30])
  #print(typeof(as.matrix(c(mat))))
  #######################################
  prediction = predict(rf_opt, mat) #$prediction
  if (verbose) 
  {
    checker = numeric(ncol(unique_m[1:86]))
    names(checker) = colnames(unique_m)[1:86]
    checker[match(names(tmp) , colnames(unique_m))] = tmp
    matched_location = c()
    for (i in 1:nrow(unique_m))
    {
      if ( all(checker == unique_m[i,1:86]) ) matched_location = c(matched_location,i)
    }
    if (length(matched_location) > 0)
    {
      info = unique_m[matched_location,c("critical_temp", "material")]
    } else
    {
      info = "No match(es) found."
    }  
  }
  if (verbose) 
  {
    out = list(prediction = prediction, info = info)
  } else {
    out = prediction
  }
  return(out)
}

###############################################################################################



predict_tc_nn = function(your_material, verbose = F)
{
  tmp = makeup(your_material)
  if ( any( names(tmp) == "Fe") ) tmp_iron = 1 else tmp_iron = 0
  if ( any( names(tmp) == "O") & any( names(tmp) == "Cu") ) tmp_cuprate = 1 else tmp_cuprate = 0
  mat = as.data.frame( t(c(as.numeric(extract( x = your_material, ed = subset_element_data)) , 
                           tmp_iron, tmp_cuprate))  )
  colnames(mat) = colnames(train)[-ncol(train)]
  #######################################
  mat <- mat[features]
  #print(mat)
  
  #mat <- scale(mat, center = col_means_train, scale = col_stddevs_train)
  mat <- scale(mat, center = col_means_train, scale = col_stddevs_train)
  #print(mat)
  #######################################
  prediction = nn %>% predict(as.matrix(mat)) #$prediction
  if (verbose) 
  {
    checker = numeric(ncol(unique_m[1:86]))
    names(checker) = colnames(unique_m)[1:86]
    checker[match(names(tmp) , colnames(unique_m))] = tmp
    matched_location = c()
    for (i in 1:nrow(unique_m))
    {
      if ( all(checker == unique_m[i,1:86]) ) matched_location = c(matched_location,i)
    }
    if (length(matched_location) > 0)
    {
      info = unique_m[matched_location,c("critical_temp", "material")]
    } else
    {
      info = "No match(es) found."
    }  
  }
  if (verbose) 
  {
    out = list(prediction = prediction, info = info)
  } else {
    out = prediction
  }
  return(out)
}

predict_tc_rf("Sr0.1La1.9Cu1O4")
predict_tc_nn("Sr0.1La1.9Cu1O4")




dataset_scale <- scale(sc_train)
dataset_scale> subset(dataset_scale, select=c(z,u))
dataset_scale<- subset(dataset_scale, select=c("range_ThermalConductivity" ,      "wtd_std_ThermalConductivity",    
                               "range_atomic_radius"         ,    "wtd_entropy_Valence",            
                               "std_ThermalConductivity"      ,   "wtd_mean_Valence",               
                               "wtd_gmean_ThermalConductivity" ,  "wtd_entropy_atomic_mass",        
                               "wtd_gmean_Valence"              , "wtd_std_ElectronAffinity",       
                               "wtd_mean_ThermalConductivity"   , "wtd_gmean_ElectronAffinity",     
                               "wtd_range_ElectronAffinity"      ,"std_atomic_mass",                
                               "wtd_entropy_ThermalConductivity" ,"range_fie",                      
                               "wtd_std_Valence"    ,             "entropy_Valence",                
                               "wtd_range_fie"       ,            "std_Density",                    
                               "wtd_entropy_fie"      ,           "std_ElectronAffinity",           
                               "wtd_range_ThermalConductivity",   "gmean_ElectronAffinity",         
                               "gmean_Density"                 ,  "wtd_entropy_Density",            
                               "gmean_Valence"                  , "wtd_std_atomic_mass",            
                               "wtd_range_Valence"            ,   "wtd_entropy_atomic_radius" ))



dataset_scale <- scale(dataset_sc)
col_means_train <- attr(dataset_scale, "scaled:center") 
col_stddevs_train <- attr(dataset_scale, "scaled:scale")

col_means_train <- col_means_train[features]
col_stddevs_train <- col_stddevs_train[features]
#sc_test_v2 <- sc_test[features]
#sc_test_v2 <- scale(sc_test_v2, center = col_means_train, scale = col_stddevs_train)


# Save multiple objects
#save(col_means_train,col_stddevs_train,std,element_data,extract,get_features,subset_element_data,
     #rf_opt,sc_train.scale,sc_test.scale,important_varaible_names,important_variables,features,
     #history,file = "sc_tc.RData")

