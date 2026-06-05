from time import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from pathlib import Path


# Layer x positions in the recoil tracker
layer_x_recoil = np.array([9.5, 15.5, 24.5, 30.5, 39.5, 45.5, 54.5, 60.5, 93.5, 95.5, 183.5, 185.5])

# Layer rotations in the recoil tracker
layer_rot_recoil = np.array([0,100,0,-100,0,100,0,-100,0,0,0,0]) #mrad

# Module centers (y_center, z_center) for back layers of recoil tracker 
module_centers = {
    8:  np.array([(-48., -40.), (-48.,  40.), ( 48., -40.), ( 48.,  40.)]),
    9:  np.array([(-96., -40.), (-96.,  40.), (  0., -40.), (  0.,  40.), ( 96., -40.), ( 96.,  40.)]),
    10: np.array([(-48., -40.), (-48.,  40.), ( 48., -40.), ( 48.,  40.)]),
    11: np.array([(-96., -40.), (-96.,  40.), (  0., -40.), (  0.,  40.), ( 96., -40.), ( 96.,  40.)]),
}

# Layer x positions in the tagger tracker
layer_x_tagger = np.array([-615.5, -609.5, -515.5, -509.5, -415.5, -409.5,
                    -315.5, -309.5, -215.5, -209.5, -115.5, -109.5, -15.5, -9.5])

# Layer offsets and rotations in the tagger tracker
layer_offset_tagger = np.array([10.4775, 10.4775, 7.3125, 7.3125, 4.7305, 4.7305, 2.7035, 2.7035, 1.2405, 1.2405, 0.3405, 0.3405, 0.003, 0.003]) # mm
layer_rot_tagger = np.array([0,100,0,-100,0,100,0,-100,0,100,0,-100,0,100]) #mrad


def transform_to_local_tagger(y_arr, z_arr, layer_arr):
    theta = layer_rot_tagger[layer_arr] * 1e-3  # mrad -> rad (signed)
    offset = layer_offset_tagger[layer_arr]
    u = -(y_arr + offset) * np.cos(theta) + z_arr * np.sin(theta)
    v =  (y_arr + offset) * np.sin(theta) + z_arr * np.cos(theta)
    return u, v

def transform_to_global_tagger(u_arr, v_arr, layer_arr):
    theta = layer_rot_tagger[layer_arr] * 1e-3  # mrad -> rad (signed)
    offset = layer_offset_tagger[layer_arr]
    y = -np.cos(theta) * u_arr + np.sin(theta) * v_arr - offset
    z =  np.sin(theta) * u_arr + np.cos(theta) * v_arr
    return y, z


def transform_to_local_recoil(y_arr, z_arr, layer_arr):
    u  = np.zeros(len(y_arr))
    v  = np.zeros(len(y_arr))
    y_c_out = np.zeros(len(y_arr))
    z_c_out = np.zeros(len(y_arr))

    front = layer_arr < 8
    back  = layer_arr >= 8

    if front.any():
        theta = layer_rot_recoil[layer_arr[front]] * 1e-3
        u[front] = -(y_arr[front]) * np.cos(theta) + z_arr[front] * np.sin(theta)
        v[front] =  (y_arr[front]) * np.sin(theta) + z_arr[front] * np.cos(theta)

    for L in [8, 9, 10, 11]:
        lmask = back & (layer_arr == L)
        if not lmask.any():
            continue
        centers = module_centers[L]
        dy = y_arr[lmask, None] - centers[:, 0]
        dz = z_arr[lmask, None] - centers[:, 1]
        idx = np.argmin(dy**2 + dz**2, axis=1)
        y_c = centers[idx, 0]
        z_c = centers[idx, 1]
        u[lmask]  = -(y_arr[lmask] - y_c)
        v[lmask]  =   z_arr[lmask] - z_c
        y_c_out[lmask] = y_c
        z_c_out[lmask] = z_c

    return u, v, y_c_out, z_c_out

def transform_to_global_recoil(u_arr, v_arr, layer_arr, y_c_arr, z_c_arr):
    y = np.zeros(len(u_arr))
    z = np.zeros(len(u_arr))

    front = layer_arr < 8
    back  = layer_arr >= 8

    if front.any():
        theta = layer_rot_recoil[layer_arr[front]] * 1e-3
        y[front] = -np.cos(theta) * u_arr[front] + np.sin(theta) * v_arr[front]
        z[front] =  np.sin(theta) * u_arr[front] + np.cos(theta) * v_arr[front]

    if back.any():
        y[back] = y_c_arr[back] - u_arr[back]
        z[back] = v_arr[back]   + z_c_arr[back]

    return y, z


# Adds strip ambiguities by setting the local v-ccordinate to 0
def add_strip_ambiguities(df, tracker):

    if tracker == 'Recoil':
        df["layer"] = df["Recoil_Digi_x"].apply(lambda x_arr: np.argmin(np.abs(x_arr[:, None] - layer_x_recoil[None, :]), axis=1))

        local_trans = df.apply(
            lambda row: transform_to_local_recoil(row['Recoil_Digi_y'], row['Recoil_Digi_x'], row['layer']), axis=1)

        df['u'] = local_trans.apply(lambda r: r[0])
        df['v'] = local_trans.apply(lambda r: r[1])
        df['y_c']    = local_trans.apply(lambda r: r[2])
        df['z_c']    = local_trans.apply(lambda r: r[3])

        # set local v-coordinates to 0
        df["v"]  = df["v"].apply(lambda x_arr: np.zeros_like(x_arr))

        global_trans = df.apply(
            lambda row: transform_to_global_recoil(row['u'], row['v'], row['layer'], row['y_c'], row['z_c']), axis=1)
        df['Recoil_Digi_y'] = global_trans.apply(lambda r: r[0])
        df['Recoil_Digi_z'] = global_trans.apply(lambda r: r[1])

        return df
        
    elif tracker == 'Tagger':
        df["layer"] = df["Tagger_Digi_x"].apply(lambda x_arr: np.argmin(np.abs(x_arr[:, None] - layer_x_tagger[None, :]), axis=1))


        df["u"] = df.apply(lambda row: transform_to_local_tagger(row["Tagger_Digi_y"], row["Tagger_Digi_z"], row["layer"])[0], axis=1)
        df["v"] = df.apply(lambda row: transform_to_local_tagger(row["Tagger_Digi_y"], row["Tagger_Digi_z"], row["layer"])[1], axis=1)
        # set local v-coordinates to 0
        df["v"]  = df["v"].apply(lambda x_arr: np.zeros_like(x_arr)) 

        df["Tagger_Digi_y"] = df.apply(lambda row: transform_to_global_tagger(row["u"], row["v"], row["layer"])[0], axis=1)
        df["Tagger_Digi_z"] = df.apply(lambda row: transform_to_global_tagger(row["u"], row["v"], row["layer"])[1], axis=1)

        return df

    else:
        raise ValueError(f"Invalid tracker name: {tracker}. Must be 'Tagger' or 'Recoil'.")
    

class MyInMemoryDataset(InMemoryDataset):

    # In EC mode the dataset includes a feature tensor (x), edge_index where edges connect space points in adjacent layers, and labels (y) indicating whether the edge connects true space points from the same track.
    # In OC mode the data includes a feature tensor (x) and labels (y) indicating the true track ID for each space point.

    def __init__(self, df, tracker = 'Tagger'):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.tracker = tracker

    def len(self):
        return len(self.df)

    def get(self, idx):

        # Build the feature tensor
        # Shape [num_nodes, 3]
        x = torch.Tensor([self.df.Digi_x[idx], self.df.Digi_y[idx], self.df.Digi_z[idx]]).T


        # Make Edge Index Tensor
        edge_index = torch.stack([ torch.Tensor(self.df.edge_index_0[idx]),torch.Tensor(self.df.edge_index_1[idx])]).to(torch.long)
        # Create corresponding Edge Feature Tensor
        edge_feat = torch.stack([torch.Tensor(self.df.edge_feat_0[idx]),torch.Tensor(self.df.edge_feat_1[idx]),torch.Tensor(self.df.edge_feat_2[idx])]).T
        # Edge Labels
        y = torch.Tensor(self.df.edge_label[idx])

        if self.tracker == 'Tagger':
            return Data(x=x, edge_index=edge_index, edge_attr=edge_feat, truthID = torch.Tensor(self.df.iloc[idx].Digi_trackID), y=y, truthP=torch.Tensor(self.df.iloc[idx].Digi_P), n_primaries =torch.tensor(self.df.iloc[idx].SimParticles_n_primaries)  )
        elif self.tracker == 'Recoil':
            return Data(x=x, edge_index=edge_index, edge_attr=edge_feat, truthID = torch.Tensor(self.df.iloc[idx].Digi_trackID), SignalID = torch.tensor(self.df.iloc[idx].SignalID), truthP = torch.tensor(self.df.iloc[idx].TruthP), y=y)
        else:
            raise ValueError(f"Invalid tracker name: {self.tracker}. Must be 'Tagger' or 'Recoil'.")


        

def plot_pyg_graph_3d(data, plot_truth=False, plot_pred=False, node_size=30, lw=0.8, label_size=12, tick_size=10, labelpad=10):
    # Node positions (N,3)
    pos = data.x[:, :3].detach().cpu()
    edge_index = data.edge_index.detach().cpu()


    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # nodes
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=node_size)

    # edges
    for e in range(edge_index.size(1)):
        i, j = edge_index[:, e]
        xs = [pos[i,0], pos[j,0]]
        ys = [pos[i,1], pos[j,1]]
        zs = [pos[i,2], pos[j,2]]


        ax.plot(xs, ys, zs, color="k", linewidth=lw, alpha=0.35)

    ax.set_xlabel("x [mm]", fontsize=label_size)
    ax.set_ylabel("y [mm]", fontsize=label_size)
    ax.set_zlabel("z [mm]", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    plt.tight_layout()
    plt.show()

    if plot_truth:
        # Node positions (N,3)
        pos = data.x[:, :3].detach().cpu()
        edge_index = data.edge_index[:,data.y.bool()].detach().cpu()


        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # nodes
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=node_size)

        # edges
        for e in range(edge_index.size(1)):
            i, j = edge_index[:, e]
            xs = [pos[i,0], pos[j,0]]
            ys = [pos[i,1], pos[j,1]]
            zs = [pos[i,2], pos[j,2]]


            ax.plot(xs, ys, zs, color="k", linewidth=lw, alpha=0.35)

        ax.set_xlabel("x [mm]", fontsize=label_size, labelpad=labelpad)
        ax.set_ylabel("y [mm]", fontsize=label_size, labelpad=labelpad)
        ax.set_zlabel("z [mm]", fontsize=label_size, labelpad=labelpad)
        ax.tick_params(labelsize=tick_size)
        plt.tight_layout()
        #plt.savefig("/Users/mghrear/Desktop/fig1.pdf")
        plt.show()

    if plot_pred:
        # Node positions (N,3)
        pos = data.x[:, :3].detach().cpu()
        edge_index = data.edge_index[:,data.pred.bool()].detach().cpu()


        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        # nodes
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=node_size)

        # edges
        for e in range(edge_index.size(1)):
            i, j = edge_index[:, e]
            xs = [pos[i,0], pos[j,0]]
            ys = [pos[i,1], pos[j,1]]
            zs = [pos[i,2], pos[j,2]]


            ax.plot(xs, ys, zs, color="k", linewidth=lw, alpha=0.35)

        ax.set_xlabel("x [mm]", fontsize=label_size, labelpad=labelpad)
        ax.set_ylabel("y [mm]", fontsize=label_size, labelpad=labelpad)
        ax.set_zlabel("z [mm]", fontsize=label_size, labelpad=labelpad)
        ax.tick_params(labelsize=tick_size)
        plt.tight_layout()
        plt.show()


# Processes pandas dataframe to get graph information
# n_primaries is the number of primaries simulated, this is only needed for label building 
def GetGraphInfo(row, tracker = "Tagger"):

    # Vertex position feature vectors in x, y, z
    x = np.stack([np.asarray(row[f'{tracker}_Digi_x']),
                np.asarray(row[f'{tracker}_Digi_y']),
                np.asarray(row[f'{tracker}_Digi_z'])], axis=1)

    # Vertex position in beam direction
    v_pos = np.asarray(row[f'{tracker}_Digi_x'])   # shape (N,)

    # Make edges connect space point in subsequent layers only
    D = np.abs(v_pos[:, None] - v_pos[None, :])   # distance along beam direction between all vertices shape (N, N)

    if tracker == "Tagger":
        T = (D < 110.0) & (D > 5.0)     # For inclusive events in the tagging tracker we only connect space points in subsequent layers
    elif tracker == "Recoil":
        T = (D < 1100.0)                # For signal events in the recoil tracker we use fully connected graphs

    # indices where T is True
    i_idx, j_idx = np.nonzero(T)                  # each shape (n_edges,)
    edge_index = np.stack((i_idx, j_idx), axis=0) # shape (2, n_edges)

    # Build edge features as vector distance between connected nodes
    src = edge_index[0]
    dst = edge_index[1]
    edge_feat = x[dst] - x[src]  # [E, F]

    # Find edges that connect space points from the same track
    vID = np.asarray(row[f'{tracker}_Digi_trackID'])
    TID = np.abs(vID[:, None] - vID[None, :]) == 0.0
    T_truth = TID * T 

    if tracker == "Tagger":
        # Remove edges connecting non-primaries
        n_primaries = row.SimParticles_n_primaries
        NopT = vID > n_primaries
        T_truth[:,NopT] = False
        T_truth[NopT,:] = False
    elif tracker == "Recoil":
        # Remove edges connecting non-signal vertices
        NopT = vID != row['SignalID']
        T_truth[:,NopT] = False
        T_truth[NopT,:] = False

    iT, jT = np.nonzero(T_truth)
    edge_index_truth = np.stack((iT, jT), axis=0)    # shape (2, n_edges)

    edge_label = (edge_index.T[:, None, :] == edge_index_truth.T[None, :, :]).all(axis=2).any(axis=1).astype(np.float32) # create a Truth label for all edges 

    return edge_index[0], edge_index[1], edge_feat.T[0], edge_feat.T[1], edge_feat.T[2], edge_label



def train(model, device, train_loader, optimizer, epoch):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        if Nparams == 4:
            output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
        else:
            output = model(data.x, data.edge_index, data.edge_attr)

        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx, len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        losses.append(loss.item())

    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)



def validate(model, device, val_loader):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)
    
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            if Nparams == 4:
                output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
            else:
                output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y, output.squeeze()
            loss = F.binary_cross_entropy(output, y, reduction='mean').item()
            losses.append(loss)

    print("...val loss=", np.mean(losses))

    return np.mean(losses)


def test(model, device, test_loader, thld=0.5):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    model.eval()
    losses, accs, TPRs, TNRs = [], [], [], []
    labels, preds = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            if Nparams == 4:
                output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
            else:
                output = model(data.x, data.edge_index, data.edge_attr)
                
            TP = torch.sum((data.y==1).squeeze() & 
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() & 
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (output<thld).squeeze()).item()            
            acc = (TP+TN)/(TP+TN+FP+FN)
            TPR = TP / (TP+FN)
            TNR = TN / (TN+FP)

            loss = F.binary_cross_entropy(output.squeeze(1), data.y, 
                                          reduction='mean').item()
            accs.append(acc)
            TPRs.append(TPR)
            TNRs.append(TNR)
            losses.append(loss)
            
            labels += data.y.cpu().tolist()
            preds += output.squeeze().cpu().tolist()

    return np.mean(losses), np.mean(accs), np.mean(TPRs), np.mean(TNRs), labels, preds 


def PrimitiveTrackBuilder(model, device, test_loader, thld=0.5, min_nodes = 5, use_truth_labels = False):

    # Check how many parameters the model's forward method takes
    # If it's 4, the model take Edep as input feature
    Nparams = len(inspect.signature(model.forward).parameters)

    df_tracks = pd.DataFrame(columns=['x', 'y', 'z'])

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            # Move datat to device 
            data = data.to(device)

            # GNN inference
            if Nparams == 4:
                output = model(data.x, data.Edep, data.edge_index, data.edge_attr)
            else:
                output = model(data.x, data.edge_index, data.edge_attr)

            if use_truth_labels:
                # This method cheats, using truth info
                y_pred = data.y.bool()
            else: 
                # Extract prediction from GNN output
                y_pred = output.squeeze()>thld 

            data.edge_index = data.edge_index[:,y_pred]

            for subgraph in data.connected_components():

                if subgraph.x.shape[0] >= min_nodes:

                    df_new = pd.DataFrame([{'x': np.asarray(subgraph.x[:,0].cpu()), 'y': np.asarray(subgraph.x[:,1].cpu()), 'z' : np.asarray(subgraph.x[:,2].cpu())}])
                    df_tracks = pd.concat([df_tracks, df_new], ignore_index=True)

    return df_tracks


def map_digi_to_momentum(row):
    """
    Maps each Digi_trackID to its corresponding momentum from TruthTrack_P.
    If ID not found in TruthTrack_ID, assigns 0.
    """
    # Create a dictionary for fast lookup
    id_to_p = dict(zip(row['TruthTrack_ID'], row['TruthTrack_P']))
    
    # Map each Digi_trackID to its momentum (0 if not found)
    mapped_p = np.array([id_to_p.get(digi_id, 0.0) for digi_id in row['Digi_trackID']])
    
    return mapped_p


def load_pickle_files_to_dataframe(directory):
    """
    Load all pickle files from a directory into a single DataFrame.
    
    Parameters:
    -----------
    directory : str or Path
        Path to the directory containing pickle files
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame from all pickle files
    """
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Get all pickle files
    pickle_files = sorted(dir_path.glob("*.pkl")) + sorted(dir_path.glob("*.pickle"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {directory}")
    
    # Load all pickle files
    dataframes = []
    for pkl_file in pickle_files:
        df = pd.read_pickle(pkl_file)
        dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df


def get_signal_recoilID(row):
    # Get track ID of electron after dark brem
    # Remove .item() to get array of signal IDs in the case of multiple signal events
    return row.SimParticles_first[(row.SimParticles_processType == 11)&(row.SimParticles_pdgID == 11)].item()

def get_signal_recoilP(row):
    # index for dark brem recoil electron
    index = np.where(row.SimParticles_first == row.SignalID)[0][0]
    # Return initial P in MeV
    return (np.sqrt(row.SimParticles_px**2+row.SimParticles_py**2+row.SimParticles_pz**2)[index])