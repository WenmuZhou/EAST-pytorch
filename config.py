# data config
trainroot = '/data2/dataset/ICD15/train'
testroot = '/data2/dataset/ICD15/test'
output_dir = 'output/east_icd15'
data_shape = 512

# train config
gpu_id = '2'
workers = 12
start_epoch = 0
epochs = 600

lr = 0.0001
lr_decay_step = 10000
lr_gamma = 0.94


train_batch_size_per_gpu  = 14

init_type = 'xavier'
display_interval = 10
show_images_interval = 50
pretrained = True
restart_training = True
checkpoint = ''
seed = 2







