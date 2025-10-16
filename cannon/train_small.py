import torch 
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from torch.utils.data import DataLoader
from torch import nn

from torch import optim

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(d_model=128, nhead=8, num_layers=4, 
          learnedPE=True, lr = 2.5e-4, 
          weight_decay = 0.01,
          batch_size = 64,
          epochs = 100
          ):
    
    
    
    
    all_data = torch.load("../data/preprocessed_small_spectra.pt", weights_only = False)
    all_sample_ids = sorted({entry['sample_id'] for entry in all_data})

    train_sample_ids = all_sample_ids
    train_data = [entry for entry in all_data if entry['sample_id'] in train_sample_ids]
    #test_data  = [entry for entry in all_data if entry['sample_id'] in train_sample_ids]

    mean_std_dict = torch.load('../data/normalization_stats.pt', weights_only = False)
    transform = NormalizeSpectralData(mean_std_dict)
    train_dataset = FastSupernovaDataset(samples=train_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    #### models ####
    model = FluxTransformerDecoder(d_model=d_model, nhead=nhead, 
                                   num_layers=num_layers, learnedPE=learnedPE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)
    
    criterion = nn.MSELoss() 
    
    model, losses = train_model(model, train_loader, 
                criterion, optimizer, scheduler, 
                epochs = epochs, 
                device = device
                )
    
    torch.save(model, f"../model_ckpt/NN_dim{d_model}_nhead{nhead}_numlayers{num_layers}_learnedPE{learnedPE}_lr{lr}_weightdecay{weight_decay}_batchsize{batch_size}_epochs{epochs}.pth")
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"../experimental_results/training_loss_NN_dim{d_model}_nhead{nhead}_numlayers{num_layers}_learnedPE{learnedPE}_lr{lr}_weightdecay{weight_decay}_batchsize{batch_size}_epochs{epochs}.png", dpi=300)
    plt.show()



import fire 

if __name__ == "__main__":
    fire.Fire(train)