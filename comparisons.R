library(RcppCNPy)
library(stringr)
library(tidyverse)


f = stringr::str_interp

load_model = function(sample_size, model_str, metric = 'roc') {
  data_dir = f('file_50_epochs_final/output_npy/${sample_size}')
  f('${data_dir}/${model_str}_${metric}.npy') %>% 
  npyLoad
}

compare = function(sample_size, model1_str, model2_str, metric='roc') {
  model1 = load_model(sample_size, model1_str, metric)
  model2 = load_model(sample_size, model2_str, metric)
  
  n = length(model1)
  count = sum(model1 >= model2)
  
  list(ratio = count/n, 
       count = count,
       n = n)
}

combos = data.frame(model1 = c("cnn", "lstm")) %>% 
  merge(data.frame(model2 = c("lstm", "han")), by = character()) %>% 
  merge(data.frame(sample_size = seq(from = 500, to = 2000, by = 500))) %>% 
  filter(as.character(model1) != as.character(model2)) %>% 
  rowwise() %>% 
  mutate(ratio1_gt_2 = compare(sample_size, model1, model2)$ratio, 
         count1_gt_2 = compare(sample_size, model1, model2)$count, 
         n = compare(sample_size, model1, model2)$n)
