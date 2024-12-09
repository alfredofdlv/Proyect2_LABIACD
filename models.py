import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # LSTM Forward Pass
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]    # Tomar la última salida (batch_size, hidden_size)
        out = self.fc(out)     # Pasar por la capa fully connected (batch_size, output_size)
        return out




class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # RNN Forward Pass
        out, _ = self.rnn(x)   # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]    # Tomar la última salida (batch_size, hidden_size)
        out = self.fc(out)     # Pasar por la capa fully connected (batch_size, output_size)
        return out



def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    # Configurar pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Ciclo de entrenamiento
    for epoch in range(num_epochs):
        model.train()  # Modo de entrenamiento
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluar en conjunto de prueba
        model.eval()  # Modo de evaluación
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        # Imprimir resultados
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
    return model


def evaluate_model(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            predictions.append(outputs.numpy())
            actuals.append(y_batch.numpy())
    return predictions, actuals
  