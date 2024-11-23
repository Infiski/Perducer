import torch
import torch.optim as optim
import torch.nn as nn
from BTier import B_Tier
from MegaDecoder import MEGADecoder



def train_perducer_encoder_decoder(train_loader, validation_loader, i_dim, h_dim, b_dim, o_dim, num_epochs=100, lr=0.00001):
    # Initialize the encoder and decoder
    encoder = B_Tier(i_dim, h_dim, b_dim)
    decoder = MEGADecoder(b_size=train_loader.dataset[0][0].size(1), b_dim=b_dim, o_dim=o_dim)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs, targets

            hidden = None  # Initialize hidden state if necessary

            optimizer.zero_grad()

            # Forward pass
            outputs, hidden = encoder(inputs, hidden)
            predictions = decoder(outputs)

            # Ensure the predictions and targets match expected dimensions for CrossEntropyLoss
            loss = criterion(predictions.view(-1, o_dim), targets.view(-1, o_dim))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                # Validation loop
        encoder.eval()
        decoder.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for batch in validation_loader:
                inputs, targets = batch

                hidden = None  # Initialize hidden state if necessary

                # Forward pass
                outputs, hidden = encoder(inputs, hidden)
                predictions = decoder(outputs)

                # Compute validation loss
                loss = criterion(predictions.view(-1, o_dim), targets.view(-1, o_dim))
                validation_loss += loss.item()

        validation_loss /= len(validation_loader)
        print(f'Validation Loss: {validation_loss:.4f}')

    # Save the model
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'model.pth')

    print("Training complete and model saved.")

def validate_model(encoder, decoder, val_loader, criterion):
    encoder.eval()
    decoder.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch

            outputs, _ = encoder(inputs)
            predictions = decoder(outputs)

            val_loss += criterion(predictions, targets).item()

            _, predicted = torch.max(predictions.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')