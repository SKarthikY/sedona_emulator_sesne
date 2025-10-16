import torch


def train_model(model, dataloader, 
                criterion, optimizer, scheduler, 
                epochs = 10, 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ):
    model.train()
    training_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            # Only print occasionally for progress
            descriptor = batch['descriptor'].float().to(device)
            time = batch['time'].unsqueeze(1).float().to(device)
            fluxes = batch['flux'].float().to(device)
            #wav = batch['wav'].float().to(device) 
            input_data = torch.cat((descriptor, time), dim=1)
            optimizer.zero_grad()
            #output = torch.clamp(model(input_data), -3, 3)
            output = model(input_data)
            
            '''
            mag_loss = torch.log10(criterion( torch.sum(10.**(spec_pred*fluxes_std + fluxes_mean)), 
                                              torch.sum(10.**(spec_true*fluxes_std + fluxes_mean)) ))'''
            

            loss = criterion(fluxes, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        training_losses.append(epoch_loss)
        
    model.eval()
    return model