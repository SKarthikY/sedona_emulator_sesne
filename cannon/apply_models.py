import torch
from sedoNNa.model import FluxTransformerDecoder
from sedoNNa.train import train_model
from sedoNNa.dataloader import NormalizeSpectralData, FastSupernovaDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


all_data = torch.load("/n/netscratch/avillar_lab/Everyone/karthik/NeuralNetworks/COperation/PredictingRawFluxes/preprocessed_spectra.pt", weights_only=False)
all_sample_ids = sorted({entry['sample_id'] for entry in all_data})
train_sample_ids, test_sample_ids = train_test_split(
    all_sample_ids, train_size=0.8, random_state=42
)
train_sample_ids = set(train_sample_ids)
train_data = [entry for entry in all_data if entry['sample_id'] in train_sample_ids]
mean_std_dict = torch.load('../data/normalization_stats.pt', weights_only=False)
transform = NormalizeSpectralData(mean_std_dict)
train_dataset = FastSupernovaDataset(samples=train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,  drop_last=True)
ckpt = "NN_dim128_nhead8_numlayers4_learnedPETrue_lr0.00025_weightdecay0.01_batchsize256_epochs100"
model = torch.load(f"../model_ckpt/{ckpt}.pth", map_location = torch.device("cpu"), weights_only=False)
model.eval()
# After model.eval()
device = "cpu"
criterion = nn.MSELoss()
from tqdm import tqdm
with torch.no_grad():
    for i, batch in tqdm(enumerate(train_loader)):
        # Input prep as before
        descriptor = batch['descriptor'].float().to(device)
        time = batch['time'].unsqueeze(1).float().to(device)
        wav = batch['wav'].cpu().numpy()
        fluxes = batch['flux'].float().to(device)
        sample_id = batch['sample_id']
        input_data = torch.cat((descriptor, time), dim=1)
        output = model(input_data)
        loss = criterion(fluxes, output)
        # Denormalize Chebyshev coefficients for FIRST sample in batch (for demo)
        norm_stats = mean_std_dict
        fluxes_mean = norm_stats['fluxes_mean'].float().to(device)
        fluxes_std = norm_stats['fluxes_std'].float().to(device)
        output = output.to(torch.float64)
        fluxes = fluxes.to(torch.float64)
        spec_pred = (output * fluxes_std + fluxes_mean).cpu().numpy() #10.**(output * fluxes_std + fluxes_mean).cpu().numpy()
        spec_true = (fluxes * fluxes_std + fluxes_mean).cpu().numpy() #10.**(fluxes * fluxes_std + fluxes_mean).cpu().numpy()
        #print("spec_pred shape: "+str(spec_pred.shape))
        #print("spec_true shape: "+str(spec_true.shape))
        time_norm = batch['time']         # [batch]            # normalized time (tensor)
        desc_norm = batch['descriptor']   # [batch, 9]         # normalized descriptor (tensor)
        # Get mean/std as numpy arrays or tensors of compatible type
        time_mean = float(mean_std_dict['time_mean'])
        time_std = float(mean_std_dict['time_std'])
        descriptor_mean = mean_std_dict['descriptor_mean'].cpu().numpy()  # [9]
        descriptor_std = mean_std_dict['descriptor_std'].cpu().numpy()    # [9]
        # Unnormalize time and descriptors for the FIRST sample in the batch:
        #    (if you want all samples, use a for loop over the batch)
        time_unnorm = time_norm[0].cpu().numpy() * time_std + time_mean
        time_d = str(round(time_unnorm/86400, 3))
        desc_unnorm = desc_norm[0].cpu().numpy() * descriptor_std + descriptor_mean
        #print(f"Batch {i} -- un-normalized time (first in batch):", time_unnorm)
        #print(f"Batch {i} -- un-normalized descriptor (first in batch):\n", desc_unnorm)
        # --- Plot as before ---
        import matplotlib.pyplot as plt
        plt.scatter(wav[0], spec_true[0], label='Ground Truth', color = 'blue')
        plt.plot(wav[0], spec_pred[0], label='Predicted\n('+f"{loss.item():04.3f}"+")", linewidth=4, color = 'green')
        plt.xlabel(r"Wavelength ($\AA$)", fontsize=35)
        plt.ylabel("Flux", fontsize=35)
        plt.title('Time_' + str(time_unnorm/86400)+"|Sample: "+str(sample_id[0]), fontsize=35)
        plt.legend(fontsize=25)
        plt.savefig(f"../experimental_results/{ckpt}_"+time_d+"_"+str(sample_id[0])+".pdf", bbox_inches='tight')
        plt.close()