class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def train(data):
    x = data.x.float().to(device)
    train_pos_edge_index = data.edge_index.to(device)
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(data, pos_edge_index, neg_edge_index):
    x = data.x.float().to(device)
    current_pos_edge_index = data.edge_index.to(device)
    model.eval()
    with torch.no_grad():
        z = model.encode(x, current_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)