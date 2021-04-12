library(tidyverse)
library(RcppCNPy)

f = stringr::str_interp

metric_f = function(model, metric, j)
  f('file_50_epochs_final/output_npy/${j}/${model}_${metric}.npy')

load_roc = function(model, j) model %>% 
  metric_f('roc', j) %>% 
  {RcppCNPy::npyLoad(.)}


calc_p = function(higher_model_arr, lower_model_arr) {
  diff_arr = higher_model_arr - lower_model_arr
  
  bias_correct = function(x)  1 + x
  (sum(diff_arr < 0) %>% bias_correct)/
    (diff_arr %>% length %>% bias_correct)
}


get_combos = function() {
  model_combos = c('cnn', 'lstm', 'han') %>% 
    {combinat::combn(., 2)} %>% 
    t %>% 
    data.frame %>% 
    purrr::set_names(c('model1', 'model2'))
  
  seq(from = 500, to = 2000, by = 500) %>% 
    data.frame %>% 
    purrr::set_names('j') %>% 
    full_join(model_combos, ., by = character()) %>% 
    arrange(j)
}


get_combos_ps = function() {
  combo_df = get_combos()
  
  calc_p_wrap = function(model1, model2, j) {
    model1_arr = load_roc(model1, j)
    model2_arr = load_roc(model2, j)
    
    calc_p(model1_arr, model2_arr)
  }
  
  length_row = function(model, j) model %>% 
    load_roc(j) %>% 
    length
  
  
  combo_df %>% 
    rename(samples_per_group = j) %>% 
    mutate(total_samples = samples_per_group * 2) %>% 
    rowwise() %>% 
    mutate(n = length_row(model1, samples_per_group),
           p = calc_p_wrap(model1, model2, samples_per_group))
}