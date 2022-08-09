class LinkPredModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, layer = GCNConv, dropout=0.25, loss = torch.nn.BCEWithLogitsLoss):
        
        super(LinkPredModel, self).__init__()
        self.conv1 = layer(input_dim, hidden_dim)
        self.conv2 = layer(hidden_dim, num_classes)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, batch):
        x , edge_index, edge_label_index = batch.x.float(), batch.edge_index, batch.edge_label_index
        
        ## Note
        ## 1. Feed the node feature into the first conv layer
        ## 2. Add a leaky-ReLU after the first conv layer
        ## 3. Add dropout after the ReLU (with probability self.dropout)
        ## 4. Repeat for the next layers
        ## 5. Select the embeddings of the source nodes and destination nodes
        ## by using the edge_label_index and compute the similarity of each pair
        ## by dot product

        h = self.conv1(x, batch.edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout)
        
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_sim = h_src * h_dst #dot product
        pred = torch.sum(h_sim, dim=-1)

        return pred
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

def train(model, train_data, val_data, test_data, device,\
          optimizer, num_epochs=200, verbose=True):
    
    avgpr_val_max = 0
    best_model = copy.deepcopy(model)
    train_data = train_data.to(device)
    best_epoch = -1
    
    avgpr_trains = []
    avgpr_vals = []
    avgpr_tests = []
    
    #roc_trains = []
    #roc_vals = []
    #roc_tests = []
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            
        #pred = best_model(train_data)

        pred = model(train_data)
        loss = model.loss(pred, train_data.edge_label.type_as(pred))

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n ROC Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_train, f1_score_train, roc_score_train = test(model, train_data, device)
        avgpr_score_val, f1_score_val, roc_score_val = test(model, val_data, device)
        avgpr_score_test, f1_score_test, roc_score_test = test(model, test_data, device)
        #score_test = test(model, dataloaders['test'], args)
        
        #f1_trains.append(f1_score_train)
        #f1_vals.append(f1_score_val)
        #f1_tests.append(f1_score_test)
        
        avgpr_trains.append(avgpr_score_train)
        avgpr_vals.append(avgpr_score_val)
        avgpr_tests.append(avgpr_score_test)
        
        if verbose:
            print(log.format(epoch, avgpr_score_train, avgpr_score_val, avgpr_score_test, roc_score_train, roc_score_val, roc_score_test, f1_score_train, f1_score_val, f1_score_test, loss.item()))
            
        if avgpr_val_max < avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
        
    return best_model, avgpr_trains, avgpr_vals, avgpr_tests

def test(model, test_data, device):
    model.eval()
    
    f1_model_score=0

    test_data = test_data.to(device)

    h = model(test_data)
    
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    pred = [1 if p > 0.5 else 0 for p in pred_cont]

    label = test_data.edge_label.cpu().detach().numpy()
      
    roc_score = roc_auc_score(label, pred_cont)
    avgpr_score = average_precision_score(label, pred_cont)
    f1_model_score = f1_score(label,pred)
 
    return avgpr_score, f1_model_score, roc_score