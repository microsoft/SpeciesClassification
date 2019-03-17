import os
import shutil
import subprocess
import torch

def main():
    # Output directory for all the models after successful training
    output_dir = 'result/run1'
    # Paramas shared across all runs, e.g. the image directory
    shared_params = ['--data_root', '/data/animals2/species_extended/',
                     '--train_file', 'trainval_animalsExtended2017.json',
                     '--val_file', 'minival_animalsExtended2017.json',
                     '--label_smoothing', '0.15',
                     '--use_onevsall_loss']
    # Name tags for the different models that we will train
    tags = []
    # The run specific parameters, should correspond to the order in TAGS
    params = []

    ### Preparing the training configurations
    # For each model training, we define a tag and the parameters
    # The examples below are for configured for a single GPU
    # For multiple GPUs,, you might want to divide learning rate
    # and batch size by the number of GPUs and enable --sync_bn for 
    # identical results
    tags.append('resnext_448_init')
    params.append(['--model_type', 'resnext101', 
                '--image_size', '448', 
                '--epochs', '1', 
                '--epoch_decay', '4', 
                '--lr_decay', '0.1', 
                '--lr', '0.01',
                '--warm_up_iterations', '0',
                '--train_logits_only', 
                '--batch_size', '16'])

    tags.append('resnext_448')
    params.append(['--model_type', 'resnext101',
                '--image_size', '448',
                '--epochs', '100',
                '--epoch_decay', '30',
                '--lr_decay', '0.1',
                '--lr', '0.01',
                '--batch_size', '16',
                '--warm_up_iterations', '3200',
                '--resume', get_best_model_path(output_dir, 'resnext_448_init')])

    ### Inception V4 224px training
    tags.append('inc4_299_init')
    params.append(['--model_type', 'inceptionv4',
                '--image_size', '299',
                '--epochs', '1',
                '--epoch_decay', '4',
                '--lr_decay', '0.94',
                '--lr', '0.00225',
                '--batch_size', '16'])

    tags.append('inc4_299')
    params.append(['--model_type', 'inceptionv4', 
                '--image_size', '299', 
                '--epochs', '25', 
                '--epoch_decay', '4', 
                '--lr_decay', '0.94', 
                '--lr', '0.0045',
                '--batch_size', '32',
                '--resume', get_best_model_path(output_dir, 'inc4_299_init')])

    ### Inception V4 448px training
    # we could add the parameter --start_epoch 0 to reset the learning rate if needed
    # With the configuration below, we will traing from epoch 25 to 50 with 448px input
    # If you add --start_epoch 0, you probably want to set --epochs 25 to keep the training
    # duration the same
    tags.append('inc4_448')
    params.append(['--model_type', 'inceptionv4', 
                '--image_size', '448', 
                '--epochs', '50', 
                '--epoch_decay', '4', 
                '--lr_decay', '0.94', 
                '--lr', '0.0045',
                '--batch_size', '32',
                '--resume', get_best_model_path(output_dir, 'inc4_299')])

    ### Inception V4 560px training
    tags.append('inc4_560')
    params.append(['--model_type', 'inceptionv4', 
                '--image_size', '560', 
                '--epochs', '200', 
                '--epoch_decay', '4', 
                '--lr_decay', '0.94', 
                '--lr', '0.0045', 
                '--batch_size', '32',
                '--resume', get_best_model_path(output_dir, 'inc4_448')])

    ### Example of fine-tuning of Inception V4 560px on validation data
    # we probably don't need this here as we will fine-tune the whole ensemble
    # This code is here just for reference
    #tags.append('inc4_560_valft')
    #params.append(['--model_type', 'inceptionv4', 
    #            '--image_size', '560', 
    #            '--epochs', '250', 
    #            '--epoch_decay', '4', 
    #            '--lr_decay', '0.94', 
    #            '--lr', '0.0045', 
    #            '--batch_size', '32',
    #            '--resume', get_best_model_path(output_dir, 'inc4_560'),
    #            '--train_file', 'val_wo_minival2017.json'])

    # Train the ensemble
    ### Inception V4 560px + ResNeXt 448px training
    # We want to learning rate to smoothly continue where the Inception V4 training finished,
    # so I chose the starting epoch to be where the Inception V4 training finished and
    # left the initial learning rate untouched
    # We probably do not need to worry about ResNeXt here, as it seems less sensitive to
    # learning hyperparameters in general
    tags.append('inc4_560_resnext_448_ft')
    params.append(['--model_type', 'inceptionv4_resnext101', 
                '--image_size', '560', '448', 
                '--epochs', '250',
                '--start_epoch', '200',
                '--epoch_decay', '4', 
                '--lr_decay', '0.94', 
                '--lr', '0.0045', 
                '--batch_size', '16',
                '--resume', get_best_model_path(output_dir, 'inc4_560'), get_best_model_path(output_dir, 'resnext_448')])

    # Checking if everything is set up properly
    assert len(tags) == len(params)

    ### The actual training
    for tag, param in zip(tags, params):
        print('Starting training of', tag)
        result_dir = get_result_dir(output_dir, tag)
        model_best = get_best_model_path(output_dir, tag)
        if os.path.isfile(model_best):
            print('Found existing trained model at {}, skipping the training of {}'.format(model_best, tag))
        else:
            # Check for checkpoint
            checkpoint_file = 'checkpoint.pth.tar'
            if os.path.isfile(checkpoint_file):
                resume_param = ['--resume', checkpoint_file]
            else:
                resume_param = []
            subprocess.run(['python', 
                             '-m', 'torch.distributed.launch', 
                             '--nproc_per_node={}'.format(torch.cuda.device_count()),
                             'train.py']
                           + param + shared_params + resume_param, check=True)
            assert os.path.isfile('model_best.pth.tar'), 'ERROR: The training did not produce model_best.pth.tar, ' + \
                                                         'You might need to adjust learning parameters.'
            print('Seems training finished, moving trained models and log directory to', result_dir)
            os.makedirs(result_dir, exist_ok=True)
            shutil.move('model_best.pth.tar', result_dir)
            shutil.move('checkpoint.pth.tar', result_dir)
            shutil.move('log', result_dir)


def get_result_dir(output_dir, tag):
    ''' Returns the directory, where we will store all models and logs after successful training '''
    return os.path.join(output_dir, tag)

def get_best_model_path(output_dir, tag):
    ''' Returns the path, where we will store the best model after successful training '''
    return os.path.join(get_result_dir(output_dir, tag), 'model_best.pth.tar')

if __name__ == '__main__':
    main()
