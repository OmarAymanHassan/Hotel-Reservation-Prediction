from scipy.stats import randint , uniform
from box import ConfigBox
light_params = ConfigBox({"n_estimators" : randint(100,500) , "max_depth":randint(0,40) , "learning_rate" : uniform(0.01,0.2),"num_leaves":randint(20,100) , "boosting_type" : ["gbdt" , "dart" , "goss"]}
)

random_search_params = ConfigBox({"n_iter" : 4 , "cv":2 , "n_jobs":-1 , "verbose": 2 ,"random_state" :42 ,"scoring":"accuracy"})



