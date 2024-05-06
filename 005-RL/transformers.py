import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class TransformerModel(nn.Module):
    def __init__(self, feature_size):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.fc = nn.Linear(feature_size, 3)  # Output size is 3 for buy, sell, do nothing

    def forward(self, src):
        x = self.transformer(src, src)
        x = self.fc(x.mean(dim=1))  # Aggregate over the sequence length dimension
        return x

def prepare_data(data):
    train_mean = data.loc[:, ["Open", "High", "Low", "Close"]].mean()
    train_std = data.loc[:, ["Open", "High", "Low", "Close"]].std()
    data = (data.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std

    # Prepare lags for Transformer input
    lags = 5
    for lag in range(1, lags + 1):
        for col in ["Open", "High", "Low", "Close"]:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    data.dropna(inplace=True)
    return data

def train_model(model, train_features, train_targets):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_features = torch.tensor(train_features, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
    train_targets = torch.tensor(train_targets, dtype=torch.long)

    for epoch in range(10):  # Small number of epochs for demonstration
        optimizer.zero_grad()
        output = model(train_features)
        loss = criterion(output, train_targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

def backtest_signals(data, signals):
    cash = 1000
    position = 0
    for price, signal in zip(data['Close'], signals):
        if signal == 1 and cash > 0:  # Buy
            position = cash / price
            cash = 0
        elif signal == 2 and position > 0:  # Sell
            cash = position * price
            position = 0
    return cash + position * data['Close'].iloc[-1]  # Final portfolio value

def main():
    data = pd.read_csv('path_to_your_5m_data.csv')
    data = prepare_data(data)
    
    features = data.drop(columns=['signal']).values
    targets = data['signal'].values

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    model = TransformerModel(train_features.shape[1])
    train_model(model, train_features, train_targets)
    
    model.eval()
    test_preds = model(torch.tensor(test_features, dtype=torch.float32).unsqueeze(1)).argmax(dim=1).numpy()
    print('Testing complete.')

if __name__ == "__main__":
    main()
