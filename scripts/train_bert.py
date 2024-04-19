import pandas as pd
import numpy as np
import json
import time

# torch
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler
from torch.optim import AdamW

# transformers
import transformers
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
transformers.logging.set_verbosity_error()

# evaluation
from sklearn.metrics import classification_report

# logging
import os
from datetime import datetime



"""SETUP"""

start = time.time()
log_stamp = datetime.now().strftime("%m.%d_%H.%M")

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)



"""CONFIG"""

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('scripts/_config.json', 'r') as file:
    config = json.load(file)

PRETRAINED_LM = config['PRETRAINED_LM']
LOGGING = config['LOGGING']
MAX_LENGTH = config['MAX_LENGTH']
BATCH_SIZE = config['BATCH_SIZE']
NUM_EPOCHS = config['NUM_EPOCHS']
LEARNING_RATE = config['LEARNING_RATE']
DROPOUT = config['DROPOUT']
BALANCE_FACTOR = config['BALANCE_FACTOR']
PATIENCE = config['PATIENCE']

RANDOM_SEED = config['RANDOM_SEED']
torch.manual_seed(RANDOM_SEED)



"""LOAD DATA"""

input_ids_train = torch.load('data/train_input_ids.pt')
att_masks_train = torch.load('data/train_att_masks.pt')
y_train = torch.load('data/y_train.pt')

input_ids_valid = torch.load('data/valid_input_ids.pt')
att_masks_valid = torch.load('data/valid_att_masks.pt')
y_valid = torch.load('data/y_valid.pt')

with open('data/code_labels_dict.json', 'r') as f:
    code_labels_dict = json.load(f)

label_names = code_labels_dict.values()



"""DATA LOADERS"""

train_dataset = TensorDataset(input_ids_train, att_masks_train, y_train)
class_sample_count = torch.tensor([(y_train == class_).sum() for class_ in torch.unique(y_train, sorted=True)])
weight = (1. / class_sample_count.float()) ** BALANCE_FACTOR
samples_weight = torch.tensor([weight[t] for t in y_train])
train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset = TensorDataset(input_ids_valid, att_masks_valid, y_valid)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)



"""INSTANTIATE MODEL"""

output_size = len(code_labels_dict.keys())
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_LM, num_labels=output_size, output_attentions=False, output_hidden_states=False
)
model = model.cuda()

# Adjust dropout
model.bert.embeddings.dropout.p = DROPOUT
model.dropout.p = DROPOUT
for layer in model.bert.encoder.layer:
    layer.attention.self.dropout.p = DROPOUT
    layer.output.dropout.p = DROPOUT

# Optimizer
optimizer = AdamW(
    model.parameters(), lr=LEARNING_RATE,
    eps=1e-8, weight_decay=0.01
)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0,
    num_training_steps=len(train_dataloader)*NUM_EPOCHS
)



"""TRAINING"""

# Set up variables
train_loss_per_epoch = []
val_loss_per_epoch = []
best_val_loss = float('inf')
triggers = 0

print('Training Loop Started...')

# Training Loop
for epoch in range(NUM_EPOCHS):

    # Set to train mode
    model.train()
    # Set up variables
    train_loss = 0
    train_pred = []
    train_actual = []
    # Loop through batches in dataloader
    for step_num, batch_data in enumerate(train_dataloader):
        # Load batch to GPU and predict
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
        # Calculate loss and log
        loss = output.loss
        train_loss += loss.item()
        train_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))
        train_actual.append(labels.cpu().detach().numpy())
        # Calculate gradients and backpropagate
        model.zero_grad()
        loss.backward()
        # Clip gradients and step
        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    # Calculate average loss for epoch
    train_loss_per_epoch.append(train_loss / (step_num + 1))
    train_pred = np.concatenate(train_pred)
    train_actual = np.concatenate(train_actual)


    # Set to eval mode
    model.eval()
    # Set up variables
    valid_loss = 0
    valid_pred = []
    with torch.no_grad():
        # Loop through batches in dataloader
        for step_num_e, batch_data in enumerate(valid_dataloader):
            # Load batch to GPU and predict
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            # Calculate loss and log
            loss = output.loss
            valid_loss += loss.item()
            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(), axis=-1))
    # Calculate average loss for epoch
    val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    valid_pred = np.concatenate(valid_pred)

    # Check for improvement
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), 'models/bert_model.pt')
        triggers = 0
    else:
        triggers += 1

    # Early Stopping
    if triggers == PATIENCE:
        print(
            f'--Early Stopping at Epoch: {epoch + 1} - Time Elapsed - {(time.time()-start) / 60:.2f} minutes',
            f'\nTrain Loss: {train_loss_per_epoch[-1]:.4f}',
            f'\nValidation Loss: {val_loss_per_epoch[-1]:.4f}'
        )
        early_stop = True
        break
    else:
        early_stop = False

    # Print Epoch Results
    print(
        f'--Epoch: {epoch + 1} Complete - Time Elapsed - {(time.time()-start) / 60:.2f} minutes',
        f'\nTrain Loss: {train_loss_per_epoch[-1]:.4f}',
        f'\nValidation Loss: {val_loss_per_epoch[-1]:.4f}',
    )

"""EVALUATE"""

# Classification Reports 
train_class_report = classification_report(train_pred, train_actual, output_dict=True, target_names=label_names, zero_division=0)
valid_class_report = classification_report(valid_pred, y_valid, output_dict=True, target_names=label_names, zero_division=0)

# Create summary df
summary = pd.DataFrame(config, index=[log_stamp]).transpose()
# Loop through classification reports to add to summary
for report, name in zip([train_class_report, valid_class_report],['train', 'valid']):
    temp = pd.DataFrame({
        f'{name}_accuracy': report['accuracy'],
        f'{name}_precision': report['macro avg']['precision'],
        f'{name}_recall': report['macro avg']['recall'],
        f'{name}_f1': report['macro avg']['f1-score']
    }, index=[log_stamp]).transpose()
    summary = pd.concat([summary, temp], axis=0)
    print(f'--{name} results')
    print(temp)
# Add epochs and time
summary = pd.concat([summary, pd.DataFrame(
    {'epochs': epoch+1, 'time_min': round(((time.time()-start) / 60), 2)}, index = [log_stamp]
).transpose()], axis = 0)



"""LOG RESULTS"""

# Save model if not already saved
if not early_stop:
    torch.save(model.state_dict(), 'models/bert_model.pt')

if LOGGING:
    # Capture losses, predictions, and classification reports in dfs
    losses_per_epoch = pd.DataFrame({'train_loss': train_loss_per_epoch, 'val_loss': val_loss_per_epoch})

    train_output = pd.DataFrame({'train_pred': train_pred, 'y_train': train_actual})
    val_output = pd.DataFrame({'val_pred': valid_pred, 'y_valid': y_valid})

    train_class_report = pd.DataFrame(train_class_report).transpose()
    valid_class_report = pd.DataFrame(valid_class_report).transpose()

    # Log to excel                    
    filename = f'logs/train_results_{log_stamp}.xlsx'

    with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary')
        train_class_report.to_excel(writer, sheet_name='train_class_report')
        valid_class_report.to_excel(writer, sheet_name='valid_class_report')
        losses_per_epoch.to_excel(writer, sheet_name='losses_per_epoch')
        train_output.to_excel(writer, sheet_name='train_results')
        val_output.to_excel(writer, sheet_name='val_results')
else:
    summary.to_csv(f'logs/summary_temp.csv')

print(f'\nTraining Script Complete - Time Elapsed - {(time.time()-start) / 60:.2f} minutes\n')