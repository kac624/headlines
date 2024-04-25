from itertools import product
from datetime import datetime
import pandas as pd
import subprocess
import json
import os
import time

# Setup
start = time.time()

# Config
with open('scripts/_config.json', 'r') as file:
    config = json.load(file)

PREPROCESS_FOR_TUNING = config['PREPROCESS_FOR_TUNING']
PREPROCESS_ONLY_ONCE = config['PREPROCESS_ONLY_ONCE']

# Read grid of hyperparameter values
with open('scripts/_hp_grid.json', 'r') as json_file:
    hp_grid = json.load(json_file)

# Create list of all hyperparameter combinations
hp_combos = list(product(*hp_grid.values()))

# Set up variables
results = pd.DataFrame()
log_stamp = datetime.now().strftime("%m.%d_%H.%M")

# Loop through each combination, run scripts, and Append results
for counter, values in enumerate(hp_combos):

    # Write hyperparameters to config
    config = dict(zip(hp_grid.keys(), values))
    config.update({
        'PREPROCESS_FOR_TUNING': PREPROCESS_FOR_TUNING,
        'PREPROCESS_ONLY_ONCE': PREPROCESS_ONLY_ONCE
    })
    with open('scripts/_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(
        f'\nRUN NUMBER {counter+1} OF {len(hp_combos)} - TIME ELAPSED - {(time.time()-start) / 60:.2f} minutes\n'
        f'\n----Config----'
    )
    for key, value in config.items():
        print(f'{key}: {value}')

    # Run preprocessing
    if PREPROCESS_FOR_TUNING:
        print('\n----Preprocess----')
        output = subprocess.run(['python', 'scripts/preprocess.py'], capture_output=True, text=True)
        print(output.stdout)
        if PREPROCESS_ONLY_ONCE:
            PREPROCESS_FOR_TUNING = False
    
    # Run training
    print('----Train----')
    output = subprocess.run(['python', 'scripts/train_llm.py'], capture_output=True, text=True)
    print(output.stdout)

    # Append results
    result = pd.read_csv('logs/summary_temp.csv', index_col=0)
    results = pd.concat([results, result], axis=1)

    # Save results and delete temp file
    results.to_csv(f'logs/tuning_results_{log_stamp}.csv')
    os.remove('logs/summary_temp.csv')

print(f'\n\nHYPERPARAMETER TUNING SCRIPT COMPLETE - Time Elapsed - {(time.time()-start) / 60:.2f} minutes\n')

best_params = {
    'CONSOLIDATE_LABELS': True,
    'CONSOLIDATE_OTHER': True,
    'DATA_AUGMENTATION': {
        'PERCENTAGE': 0.2, 'BACKTRANSLATION': True, 'DELETION': True, 'REPLACEMENT': True
    },
    'TRAIN_SPLIT_PROP': [0.7], 
    'PRETRAINED_LM': ['bert-base-uncased'],
    'LOGGING': [True],
    'MAX_LENGTH': [128],
    'BATCH_SIZE': [32],
    'NUM_EPOCHS': [10],
    'LEARNING_RATE': [2e-5],
    'DROPOUT': [0.3],
    'BALANCE_FACTOR': [0.3],
    'PATIENCE': [3]
}