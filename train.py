import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'2 in tarim'
import warnings
warnings.filterwarnings('ignore')
from preprocessing import read_load_trace_data,preprocessing_gnn
import config as cf
import numpy as np
import pandas as pd
from torchinfo import summary
import torch
from tqdm import tqdm
from GNN_models import Net_GCN, Net_GAT,Net_MLP,Net_GatedGNN
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from threshold_throttling import threshold_throttleing
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
import torch.optim as optim


device = cf.device
batch_size = cf.batch_size
epoch_num = cf.epochs

#model = Net_GCN()
#model = Net_GAT()
model = Net_GatedGNN()
#model = Net_MLP()

print(summary(model))
optimizer = optim.Adam(model.parameters(), lr=cf.lr)
log=cf.Logger()

#%%

#import pdb
#pdb.set_trace()

def train(ep,train_loader,model_save_path):
    epoch_loss = 0
    model.train()
    for batch_idx, batch in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        batch = batch.to(device)
        output = model(batch)
        target = batch.t.view(-1,cf.num_classes).to(device)
        loss = F.binary_cross_entropy(output, target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            output = model(batch)
            target = batch.t.view(-1,cf.num_classes).to(device)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, loading, model_save_path,train_loader,test_loader):

    best_loss=0
    early_stop=cf.early_stop
    model.to(device)
    for epoch in range(epochs):
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            early_stop=cf.early_stop
        else:
            early_stop-=1
            log.logger.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break


#%% evaluation

def model_prediction(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for batch_idx, batch in enumerate(test_loader):
    #for data,ip,page,_ in tqdm(test_loader):
        batch = batch.to(device)
        output= model(batch)
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['future', 'y_score']]


def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

def run_val(test_loader,test_df,file_path,model_save_path):
    print("Validation start")
    test_df=model_prediction(test_loader, test_df,model_save_path)

    df_thresh={}
    app_name=file_path.split("/")[-1].split("-")[0]
    val_res_path=model_save_path+".val_res.csv"
    
    df_res, threshold=threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro")
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["app"],df_thresh["opt_th"],df_thresh["p"],df_thresh["r"],df_thresh["f1"]=[app_name],[threshold],[p],[r],[f1]
    
    df_res, _ =threshold_throttleing(test_df,throttle_type="fixed_threshold",threshold=0.5)
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["p_5"],df_thresh["r_5"],df_thresh["f1_5"]=[p],[r],[f1]
    
    pd.DataFrame(df_thresh).to_csv(val_res_path,header=1, index=False, sep=" ") #pd_read=pd.read_csv(val_res_path,header=0,sep=" ")
    print("Done: results saved at:", val_res_path)

#%% 

'''
Modify the file path to run
'''
SKIP_NUM=1
TRAIN_NUM = 3
TOTAL_NUM=4

#file_path="../data/410.bwaves-s0.txt.xz"
file_path="../data/654.roms-s0.txt.xz"

if not os.path.exists("./res"):
    os.makedirs("./res")
    
model_save_path = "./res/model_gnn.pkl"


#%%
log_path=model_save_path+".log"
log.set_logger(log_path)
log.logger.info("%s"%file_path)


train_data, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
df_train, df_test = preprocessing_gnn(train_data), preprocessing_gnn(eval_data)

train_loader = DataLoader(list(df_train.graph.values), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(list(df_test.graph.values), batch_size=batch_size)
log.logger.info("-------------Data Proccessed------------")

#%%
run_epoch(epoch_num, 0 ,model_save_path,train_loader,test_loader)
run_val(test_loader,df_test,file_path,model_save_path)


#%%
'''
GAT(51,072): 410: p,r,f1: 0.7969418160404228 0.9443799009727789 0.8644190512988532
GAT(36,128): 410:  0.7796133099724948 0.9353451882049605 0.8504084022071309
GAT(76,032): 410: p,r,f1: 0.8059511225947893 0.9394722731386761 0.8676046568799607
GAT(402,176): 0.8237940092311741 0.9399639652652525 0.8780532189511733

GCN (34,480): 410: p,r,f1: 0.7757994831927031 0.9336519520227565 0.8474375895561123
GCN (85,264): 410: 0.7668048835057341 0.93516074810426 0.8426560621331544


'''

