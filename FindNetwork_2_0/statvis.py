import os
from copy import copy
from types import SimpleNamespace

import bokeh.palettes
import img2pdf
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from neuprint import *


def LogInHemibrain(dataset='hemibrain:v1.2.1',token=''): # log in to hemibrain dataset
    '''
    Log in to hemibrain dataset;
    Please provide your own token, which can be obtained from https://neuprint.janelia.org/account
    '''
    client = Client(
        'neuprint.janelia.org',
        dataset = dataset,
        token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImtybGVuZzEyMTg0QGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUFUWEFKeTdKZ1JCeUFZYkt6YzFSbTl3ejV4X0luQmJydXNPOEg5MnllSVc9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MzI1MzQzNjJ9.ejDfFvsUcDuIm_3opGSGI0VDW_1ImNvD9zKEDImN9GA'
    )
    print("Logged in \ndataset: " + dataset)
    return client, dataset

def getCriteriaAndName(requiredNeurons):
    from neuprint import NeuronCriteria as NC
    if requiredNeurons == None:
        criteria = None
        fname = 'ALL'
    elif type(requiredNeurons[0]) == int:
        criteria = NC(bodyId=requiredNeurons)
        fname = str(requiredNeurons[0])
    elif requiredNeurons[0].find('.*') != -1:
        criteria = NC(type=requiredNeurons,regex=True)
        fname = requiredNeurons[0][:-2]
    else:
        criteria = NC(type=requiredNeurons)
        fname = requiredNeurons[0]
    if requiredNeurons != None and len(requiredNeurons) > 1:
        fname += '_etc'
    return criteria, fname

def removeSearchedNeurons(conn_df,searchedNeurons):
    '''remove neurons on searched layers'''
    neurons_post = conn_df['bodyId_post'].unique()
    common_neurons = np.intersect1d(neurons_post,searchedNeurons,assume_unique=True)
    df = conn_df[~conn_df['bodyId_post'].isin(common_neurons)]
    return df

def Conn2FullMat(source_df,target_df,conn_df,conn_type,weight_col='weight'): 
    '''convert connection table (conn_df) to a full connection matrix (keep zero connections)'''
    sbodyId = source_df.bodyId.tolist()
    tbodyId = target_df.bodyId.tolist()
    stype = source_df.type.unique().tolist()
    ttype = target_df.type.unique().tolist()
    sbodyId.sort()
    tbodyId.sort()
    stype.sort()
    ttype.sort()
    cmat_bodyId = pd.DataFrame(data=np.zeros([len(sbodyId),len(tbodyId)],dtype=int),index=sbodyId,columns=tbodyId)
    cmat_type = pd.DataFrame(data=np.zeros([len(stype),len(ttype)],dtype=int),index=stype,columns=ttype)
    for i in conn_df.index:
        bpre  = conn_df.at[i,'bodyId_pre']
        bpost = conn_df.at[i,'bodyId_post']
        bweight = conn_df.at[i,weight_col]
        cmat_bodyId.at[bpre,bpost] = bweight
    for i in conn_type.index:
        tpre  = conn_type.at[i,'type_pre']
        tpost = conn_type.at[i,'type_post']
        tweight = conn_type.at[i,weight_col]
        cmat_type.at[tpre,tpost] = tweight
    return cmat_bodyId,cmat_type

def calRC(cmat,threshold=0):
    '''calculate row and column sums of a connection matrix'''
    n_row,n_col = cmat.shape
    sourceN = [0]*n_col 
    targetN = [0]*n_row
    sum_col = [0]*n_col
    sum_row = [0]*n_row
    for i in range(n_row):
        for j in range(n_col):
            val = cmat.iat[i,j]
            sum_row[i] += val
            sum_col[j] += val
            if val > threshold:
                targetN[i] += 1
                sourceN[j] += 1
    cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=sourceN, axis=0))
    cmat_new = pd.DataFrame(np.insert(cmat_new.values, len(cmat_new.index), values=sum_col, axis=0))
    cmat_new.columns = cmat.columns
    cmat_new.index = list(cmat.index) + ['sourceN','sum_col']
    cmat_new.insert(loc=len(cmat.columns),column='targetN',value=targetN+[0,0])
    cmat_new.insert(loc=len(cmat.columns)+1,column='sum_row',value=sum_row+[0,sum(sum_row)])
    return cmat_new

def filtMat(cmat,axis=0,filt_range=[0,1],by='MR'): 
    '''identify columns whose maximums are in the range'''
    if by == 'MR': # maximum ratio
        nval = cmat.shape # nval = (n_row, n_col)
        criterion = [1]*nval[1-axis]
        maxVal = cmat.max(axis=axis)
        if filt_range[0] != filt_range[1]:
            for j in range(nval[1-axis]):
                if maxVal[j] <= filt_range[0] or maxVal[j] > filt_range[1]: # left open, right closed interval
                    criterion[j] = 0
        else:
            for j in range(nval[1-axis]):
                if maxVal[j] != filt_range[0]: # left open, right closed interval
                    criterion[j] = 0
        if axis == 0:
            cmat_new = pd.DataFrame(np.insert(cmat.values, len(cmat.index), values=criterion, axis=0))
            cmat_new.index = list(cmat.index) + ['sourceCriterion']
            cmat_new.columns = cmat.columns
            cmat_new = cmat_new.loc[:,cmat_new.loc['sourceCriterion'] == 1]
            cmat_new = cmat_new.iloc[:-1,:]
        elif axis == 1:
            cmat_new = cmat.copy()
            cmat_new.insert(loc=len(cmat.columns), column='targetCriterion', value=criterion)
            cmat_new = cmat_new.loc[cmat_new['targetCriterion'] == 1,:]
            cmat_new = cmat_new.iloc[:,:-1]
    elif by == 'N': # synapse number
        cmat_t = calRC(cmat) # new connection matrix
        if axis == 0:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] <= filt_range[1]]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[:,cmat_t.loc['sourceN'] >= filt_range[0]]
            cmat_t = cmat_t.iloc[:-2,:]
        elif axis == 1:
            if filt_range[0] != None and filt_range[1] != None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[0] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] <= filt_range[1],:]
            elif filt_range[1] == None:
                cmat_t = cmat_t.loc[cmat_t['targetN'] >= filt_range[0],:]
            cmat_t = cmat_t.iloc[:,:-2]
        cmat_new = cmat_t
    return cmat_new

def stMat(mat,axis=0):
    '''standardize matrix by row or column'''
    matt = calRC(mat)
    rowN,colN = matt.shape
    if axis == 0: # standardize by column
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[-1,j]
    elif axis == 1: # standardize by row
        for i in range(rowN-2):
            for j in range(colN-2):
                matt.iat[i,j] /= matt.iat[i,-1]
    return matt.iloc[:-2,:-2]

def VisConnMat(cmat,filename,title='',color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],showfig=True,fontsize=12): 
    '''visualize connection matrix'''
    textlabel = cmat.astype(str)
    textlabel.replace('0','',inplace=True)
    # heatmap
    fig = go.Figure(data=go.Heatmap(
        z = cmat,
        x = cmat.columns.astype(str),
        y = cmat.index.astype(str),
        colorscale = color_scale,
        # text = textlabel,
        texttemplate="%{text}",
        textfont={"size":10}
    ))
    fig.update_layout(title_text=title,font_size=fontsize)
    fig.write_html(filename, auto_open=showfig)

def ClusterMap(cmat:pd.DataFrame,filename,cmap='plasma',scale_ratio=3,reshape_factor=3,zs=None,method='median',showfig=False):
    '''clustermap of connection matrix'''
    
    (rowN,colN) = cmat.shape
    fig = sns.clustermap(cmat,
                    method=method,
                    figsize=(min(colN/scale_ratio+reshape_factor,900),min(rowN/scale_ratio+reshape_factor,900)),
                    dendrogram_ratio=(.2,.3),
                    z_score=zs,
                    cmap=cmap) # 'plasma','Blues','RdBu'
    fig.savefig(filename)
    if not showfig: plt.close()
    new_index = fig.dendrogram_row.reordered_ind
    new_columns = fig.dendrogram_col.reordered_ind
    newmat:pd.DataFrame = cmat.copy()
    newmat = cmat.iloc[new_index,new_columns]
    return fig, newmat

def RN2plot(dataR,dataN):
    '''convert max ratio matrix and connection number matrix to plot data'''
    c_mr = pd.DataFrame(dataR)
    c_mr = c_mr.reset_index()
    c_mr.columns = ['type','max_ratio']
    c_N = pd.DataFrame(dataN)
    c_N = c_N.reset_index()
    c_N.columns = ['type','N']
    c_plot = c_mr.merge(c_N,how='inner')
    c_plot = c_plot.sort_values(by=['N','max_ratio'],ascending=[True,False])
    return c_plot

def ConnHist(dataMat,cat,suffix): 
    '''histogram of connection distribution'''
    import matplotlib.pyplot as plt
    if cat.find('MR') != -1:
        binN = 10
    else:
        binN = max(int(dataMat.max()),5)
    fig,ax = plt.subplots(1,2,tight_layout=True)
    # pdf by counts
    ax[0].hist(dataMat,bins=binN,lw=0)
    ax[0].set_title('Distribution of '+cat)
    ax[0].set_xlabel(cat+' '+suffix)
    ax[0].set_ylabel('count')
    ax[0].grid(False)
    # cdf
    ax[1].hist(dataMat,cumulative=True,bins=binN,lw=0)
    ax[1].set_title('CDF of '+cat)
    ax[1].set_xlabel(cat+' '+suffix)
    ax[1].set_ylabel('count')
    ax[1].grid(False)
    return fig,ax

def VisConnDist(cmat,save_path,suffix='',showfig=True,save_format='.svg'): 
    '''visualize connection (source and target) distributions'''
    # distribution of max ratio of source neurons
    cmat_statR = stMat(cmat)
    dataR = cmat_statR.max()
    fig,_ = ConnHist(dataR,cat='MR_source',suffix=suffix) # max ratio of source neurons
    fig.savefig(os.path.join(save_path,'dist_MR_source_'+suffix+save_format),dpi=300)
    if not showfig: plt.close(fig)
    # distribution of source number
    cmat_statN = calRC(cmat)
    dataN = cmat_statN.iloc[-2,:-2] # row: sourceN
    fig,_ = ConnHist(dataN,cat='source N',suffix=suffix)
    fig.savefig(os.path.join(save_path,'dist_sourceN_'+suffix+save_format),dpi=300)
    if not showfig: plt.close(fig)
    # plot Max Ratio of source against sourceN
    c_plot = RN2plot(dataR,dataN)
    c_plot.columns = ['type_post','max_ratio','sourceN']
    c_plot.to_csv(os.path.join(save_path,'dataDist_source_'+suffix+'.csv'))
    fig,ax = plt.subplots(1,1,tight_layout=True,dpi=300)
    for i in c_plot.index:
        ax.scatter(c_plot.at[i,'sourceN'],c_plot.at[i,'max_ratio'],c='b',alpha=0.1,edgecolors='none')
    ax.grid(False)
    ax.set_xlabel('# of source '+suffix)
    ax.set_ylabel('max ratio of source '+suffix)
    fig.savefig(os.path.join(save_path,'MR_against_sourceN_'+suffix+save_format))
    if not showfig: plt.close(fig)
    # distribution of max ratio of target neurons
    cmat_statR = stMat(cmat,axis=1)
    dataR = cmat_statR.max(axis=1)
    fig,_ = ConnHist(dataR,cat='MR_target',suffix=suffix)
    fig.savefig(os.path.join(save_path,'dist_MR_target_'+suffix+save_format),dpi=300)
    if not showfig: plt.close(fig)
    # distribution of target number
    dataN = cmat_statN.iloc[:-2,-2] # row: targetN
    fig,_ = ConnHist(dataN,cat='target N',suffix=suffix)
    fig.savefig(os.path.join(save_path,'dist_targetN_'+suffix+save_format),dpi=300)
    if not showfig: plt.close(fig)
    # plot Max Ratio of source against targetN
    c_plot = RN2plot(dataR,dataN)
    c_plot.columns = ['type_pre','max_ratio','targetN']
    c_plot.to_csv(os.path.join(save_path,'dataDist_target_'+suffix+'.csv'))
    fig,ax = plt.subplots(1,1,tight_layout=True,dpi=300)
    for i in c_plot.index:
        ax.scatter(c_plot.at[i,'targetN'],c_plot.at[i,'max_ratio'],c='b',alpha=0.1,edgecolors='none')
    ax.grid(False)
    ax.set_xlabel('# of target '+suffix)
    ax.set_ylabel('max ratio of target '+suffix)
    fig.savefig(os.path.join(save_path,'MR_against_targetN_'+suffix+save_format))
    if not showfig: plt.close(fig)

def sortMatByMax(cmat,save_path,suffix,title='',by='sourceMR',filt_range=[0.5,1],clusterFlag=False,showfig=False): 
    '''sort connection matrix by max value of source/target neurons or number of source/target neurons'''
    # reorder columns(target neurons) by max{synapse number or percentage from source neurons or target neurons} —— N or Max Ratio (MR)
    # interval taken by filt_range is left open and right closed if by=='sourceMR' or 'targetMR'
    # interval taken by filt_range is left closed and right closed if by=='sourceN' or targetN
    suffix = suffix + '_' + by + '_'
    if by.find('source') != -1:
        axis = 0
    elif by.find('target') != -1:
        axis = 1
    if by.find('MR') != -1:
        suffix_new = suffix+str(int(filt_range[0]*100))+'to'+str(int(filt_range[1]*100))
        cmat_t = filtMat(stMat(cmat,axis=axis),axis=axis,filt_range=filt_range,by='MR')
    elif by.find('N') != -1:
        suffix_new = suffix+str(filt_range[0])+'to'+str(filt_range[1])
        cmat_t = filtMat(cmat,axis=axis,filt_range=filt_range,by='N')
    
    cmat_filt = cmat_t.copy()
    ind_max = cmat_filt.idxmax(axis=axis) # return the maximum value in each columns (axis=0) or rows (axis=1)
    if axis == 0:
        for j in cmat_filt.columns:
            ind_max_row = ind_max.at[j]
            for i in cmat_filt.index:
                if i != ind_max_row:
                    cmat_filt.at[i,j] = 0
        cmat_filt = calRC(cmat_filt)
        cmat_filt = cmat_filt.iloc[:-2,:]
        cmat_filt = cmat_filt.sort_values(by=['targetN','sum_row'],axis=0,ascending=[0,0])
        e_rowN,e_colN = cmat_filt.shape
        sorted_col = []
        for r in range(e_rowN):
            r_name = cmat_filt.index[r]
            curr_data = cmat_filt.iloc[r,:-2]
            curr_data = cmat_filt.iloc[r,:-2].to_numpy()
            asorted_col = np.argsort(-curr_data) # descending, returning the index
            sorted_col = np.append(sorted_col, asorted_col[:int(cmat_filt.at[r_name,'targetN'])]) # keep indexes of non-zero values
        cmat_re = cmat_t.copy() # rebuilt cmat
        cmat_filt = cmat_filt.iloc[:,:-2]
        cmat_re = cmat_re.loc[cmat_filt.index,cmat_filt.columns]
        emat = cmat_re.iloc[:,sorted_col]
    elif axis == 1:
        for i in cmat_filt.index:
            ind_max_col = ind_max.at[i]
            for j in cmat_filt.columns:
                if j != ind_max_col:
                    cmat_filt.at[i,j] = 0
        cmat_filt = calRC(cmat_filt)
        cmat_filt = cmat_filt.iloc[:,:-2]
        cmat_filt = cmat_filt.sort_values(by=['sourceN','sum_col'],axis=1,ascending=[0,0])
        e_rowN,e_colN = cmat_filt.shape
        sorted_row = []
        for r in range(e_colN):
            col_name = cmat_filt.columns[r]
            curr_data = cmat_filt.iloc[:-2,r]
            curr_data = cmat_filt.iloc[:-2,r].to_numpy()
            asorted_row = np.argsort(-curr_data) # descending, returning the index
            sorted_row = np.append(sorted_row, asorted_row[:int(cmat_filt.at['sourceN',col_name])]) # keep indexes of non-zero values
        cmat_re = cmat_t.copy() # rebuilt cmat
        cmat_filt = cmat_filt.iloc[:-2,:]
        cmat_re = cmat_re.loc[cmat_filt.index,cmat_filt.columns]
        emat = cmat_re.iloc[sorted_row,:]
    if not os.path.exists(os.path.join(save_path,'csv')): os.mkdir(os.path.join(save_path,'csv'))
    emat.to_csv(os.path.join(save_path,'csv','EorC_'+suffix_new+'.csv'))
    VisConnMat(emat.iloc[::-1],title=title,filename=os.path.join(save_path,'EorC_'+suffix_new+'.html'),showfig=showfig)
    if clusterFlag == True:
        _,emat_clusterd = ClusterMap(emat,filename=os.path.join(save_path,'EorC_'+suffix_new+'_clustered.png'))
        emat_clusterd.to_csv(os.path.join(save_path,'csv','EorC_'+suffix_new+'_clustered.csv'))
        
def DrawGraph(G,pos,edge_width,node_size=300,font_size=5,font_color='silver'):
    nodeN = nx.number_of_nodes(G)
    fig, ax = plt.subplots(figsize=(min(3*nodeN**0.5+3,50),min(3*nodeN**0.5+3,50)),dpi=150)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(G, pos=pos, width=edge_width)
    nx.draw_networkx_labels(G, pos=pos, font_size=font_size, font_color=font_color)
    ax.set_axis_off()
    ax.grid(False)
    return fig

def NetworkVis(source_df,target_df,conn_df,save_path='',by='bodyId',node_size=300,showfig=False,save_format='.svg'):
    G = nx.DiGraph()
    for i in source_df[by]:
        G.add_node(str(i),layer=0)
    for i in target_df[by]:
        G.add_node(str(i),layer=1)
    for i in conn_df.index:
        G.add_edge(str(conn_df.loc[i,by+'_pre']),str(conn_df.loc[i,by+'_post']),weight=conn_df.loc[i,'weight'])
    
    if set(source_df[by].tolist()) == set(target_df[by].tolist()):
        if nx.number_weakly_connected_components(G) > 1: # plot subgraphs but not show
            pos = nx.spring_layout(G,seed=410)
            # pos = nx.shell_layout(G)
            fig = DrawGraph(G,pos=pos,edge_width=np.log(conn_df.weight),node_size=node_size)
            fig.savefig(os.path.join(save_path,'Network_'+by+save_format))
            if not showfig: plt.close(fig)
            G_subs = list(nx.weakly_connected_components(G))
            for i in range(len(G_subs)):
                Gsub = G.subgraph(G_subs[i])
                if Gsub.number_of_nodes() > 1: # subgraphs with only one node will not be drawn
                    pos_i = nx.kamada_kawai_layout(Gsub)
                    fig = DrawGraph(Gsub,pos=pos_i,edge_width=np.log(conn_df.weight))
                    fig.savefig(os.path.join(save_path,'Network_'+by+'_subgraph_'+str(i)+save_format))
                    plt.close(fig)
        else:
            pos = nx.kamada_kawai_layout(G)
            fig = DrawGraph(G,pos=pos,edge_width=np.log(conn_df.weight))
            fig.savefig(os.path.join(save_path,'Network_'+by+save_format))
            if not showfig: plt.close(fig)
    else: # layered structure
        pos = nx.multipartite_layout(G, subset_key='layer')
        fig = DrawGraph(G,pos=pos,edge_width=np.log(conn_df.weight))
        fig.savefig(os.path.join(save_path,'Network_'+by+save_format))
        if not showfig: plt.close(fig)

def getAllPath(conn_data,targets,traversal_probability_threshold=0): ### debugging
    G = nx.DiGraph()
    for i in reversed(range(len(conn_data))): # wrong weights
        layer = conn_data.iat[i,0]
        layer_pre = int(layer[0])
        layer_post = int(layer[-1])
        node_pre = str(conn_data.iat[i,1])
        node_post = str(conn_data.iat[i,2])
        weight_i = conn_data.iat[i,3]
        travP_i = conn_data.iat[i,4]
        G.add_node(node_post,layer=layer_post)
        G.add_node(node_pre,layer=layer_pre)
        G.add_edge(node_pre,node_post,weight=weight_i,probability=travP_i)
    nodes_info = dict(G.nodes(data='layer'))
    connN = max(conn_data.conn_layer)
    layerN = int(connN[-1]) + 1
    sources = conn_data.loc[conn_data.conn_layer=='0->1']
    sources = sources.iloc[:,1].unique().tolist()
    paths = []
    pairN = len(sources) * len(targets)
    count = 0
    for source_i in sources:
        for target_j in targets:
            count += 1
            print('\rpath processed:','{:.2%}'.format(count/pairN),end='')
            if nx.has_path(G,str(source_i),str(target_j)):
                curr_paths = list(nx.all_simple_paths(G,str(source_i),str(target_j)))
                for p in reversed(range(len(curr_paths))):
                    pp = curr_paths[p]
                    if len(pp) > layerN: # exclude paths whose layers are not monotonically increasing
                        curr_paths.pop(p)
                    else: # exclude paths not following the layer order
                        for i in range(1,len(pp)-1):
                            node_layer = nodes_info[pp[i]]
                            if node_layer != i:
                                curr_paths.pop(p)
                                break
                paths += curr_paths
    print()
    path_blocks = []
    weights = []
    travPs = [] # traversal probability between nodes of each path
    travP = [] # traversal probability of the path, equal to prod(travPs[i])
    weights_min = []
    inter_layer_num = []
    for p in paths:
        block = ''
        w_p = []
        travP_p = []
        for ind in range(len(p)):
            block += (p[ind]+' -> ')
            if ind + 1 < len(p):
                edge_t = G.get_edge_data(p[ind],p[ind+1])
                weight_edge = edge_t['weight']
                travP_edge = edge_t['probability']
                w_p.append(weight_edge)
                travP_p.append(travP_edge)
        block = block[:-4]
        path_blocks.append(block)
        weights.append(w_p)
        weights_min.append(min(w_p))
        travPs.append(travP_p)
        travP.append(np.prod(travP_p))
        inter_layer_num.append(len(p)-2)
    source_nodes = [p[0] for p in paths]
    target_nodes = [p[-1] for p in paths]
    path_dict = {
        'path_block': path_blocks,
        'inter_layer_num': inter_layer_num,
        'traversal_probability': travP,
        'min_weight': weights_min,
        'traversal_probabilities': travPs,
        'weights': weights,
        'source': source_nodes,
        'target': target_nodes
    }
    path_df = pd.DataFrame.from_dict(path_dict)
    path_df = path_df.sort_values(by=['traversal_probability','inter_layer_num','min_weight'],ascending=[False,True,False])
    path_df = path_df.reset_index(drop=True)
    path_df = path_df.loc[path_df.traversal_probability >= traversal_probability_threshold]
    return path_df,paths

def EnrichConnectionTable(conn_table,traversal_probability_threshold=0):
    '''Add traversal probability, layer information to the connection table'''
    conn_df = conn_table.copy()
    df_post,_ = fetch_neurons(conn_df.bodyId_post.tolist())
    post_info = df_post[['bodyId','post']]
    post_info.columns = ['bodyId_post','post']
    conn_df.loc[conn_df.type_pre.isnull(),'type_pre'] = 'None'
    conn_df.loc[conn_df.type_post.isnull(),'type_post'] = 'None'
    conn_df = conn_df.merge(post_info,how='left',on='bodyId_post')
    conn_df.insert(loc=len(conn_df.columns),column='ratio_post',value=conn_df.weight/conn_df.post)
    conn_df.insert(loc=3,column='traversal_probability',value=conn_df.ratio_post/0.3)
    conn_df.loc[conn_df.traversal_probability > 1,'traversal_probability'] = 1
    conn_df.insert(loc=len(conn_df.columns),column='block_probability',value= 1 - conn_df.traversal_probability)
    
    conn_df = conn_df.loc[conn_df.traversal_probability >= traversal_probability_threshold]
    
    conn_type = conn_df.groupby(['type_pre','type_post']).sum()
    conn_type = conn_type.reset_index()
    conn_type = conn_type[['type_pre','type_post','weight']]
    conn_traversal = conn_df[['type_pre','type_post','block_probability']]
    conn_traversal = conn_traversal.groupby(['type_pre','type_post']).prod()
    conn_traversal = conn_traversal.reset_index()
    conn_type = conn_type.merge(conn_traversal,how='inner',on=['type_pre','type_post'])
    conn_type.insert(loc=3,column='traversal_probability',value = 1 - conn_type.block_probability)
    
    return conn_df,conn_type
    
def ConcatenateIMG2PDF(folder_path,file_format=['png','jpg'],filename='PDF_sum',include_subfolder=False):
    ''' Concatenate all images in a folder to a single PDF file.'''
    if 'jpg' in file_format:
        file_format.append('jpeg')
    elif 'jpeg' in file_format:
        file_format.append('jpg')
    file_format = list(set(file_format))
    files = os.listdir(folder_path)
    figs = []
    for f in files:
        ftype = os.path.splitext(f)[-1][1:]
        if ftype in file_format:
            figs.append(os.path.join(folder_path,f))
    figs.sort()
    if len(figs) > 0:
        with open(os.path.join(folder_path,filename + '.pdf'),'wb') as f_sum:
            f_sum.write(img2pdf.convert(figs))
        print('Concatenated {:d} pictures to PDF'.format(len(figs)))
    else:
        print('Found no pictures to concatenate.')
    
def Vis3S(data_df,**kwargs): 
    """ Visualize Soma, Synapses, Synapse ditributions
    Args:
        data_df (pandas.DataFrame): dataframe contains centroid, classification, axis lengths (Ellipse) or radius (Circle).
    """
    
    options = {
        "save_path" : '_3S',
        "title"     : None,
        "classby"   : 'type',
        "plane"     : 'xz',
        "alpha"     : .3,
        "dpi"       : 300,
        "toPlot"    : 'soma', # "soma" or "synapse_distribution" or "synapse"
        "xlim"      : (0,50000),
        "ylim"      : (50000,0), # reversed
        "showfig"   : False, # faster than True
        "facecolor" : bokeh.palettes.Set1[9],
        "site"      : None, # None, 'pre' or 'post'
        "snp_rois"   : None,
        "show_mesh"  : True,
        "mesh_roi"   : None,
        "roi_range"  : 'primary_rois', # {"primary_rois", "all_rois"}, see more details in neuprint.
        "mesh_color"    : [0.1,0.1,0.1],
        "mesh_alpha"    : 0.1,
        "confidence"    : 0,
        "synapseRadius" : 100,
        "synpase_file_path" : None,
        "save_format": '.png',
    }
    options.update(kwargs)
    if options['snp_rois'] != None and options['mesh_roi'] == None: options['mesh_roi'] = options['snp_rois']
    op = SimpleNamespace(**options)
    
    if op.show_mesh:
        roiunits = []
        for roi in op.mesh_roi:
            mesh_file = os.path.join('navis_roi_meshes_json',op.roi_range,roi+'.json')
            if os.path.exists(mesh_file):
                mesh = navis.Volume.from_json(os.path.join('navis_roi_meshes_json',op.roi_range,roi+'.json'))
                roiunits.append(mesh)
            else:
                print('mesh file %s.json not found!'%(roi))
        roimesh = navis.Volume.combine(roiunits)
    if op.toPlot == 'synapse':
        snp_file = pd.ExcelFile(op.synapse_file_path)
    summary_df = data_df.copy()
    if op.toPlot == 'soma':
        print('not found soma of %d neurons'%(summary_df['somaLocation'].isnull().sum()))
        summary_df = summary_df[summary_df['somaLocation'].notnull()]
    elif op.toPlot == 'synapse_distribution':
        print('drop %d neurons having no more than 1 synapses in the ROI'%((summary_df['snpN_roi']<=1).sum()))
        summary_df = summary_df[summary_df['snpN_roi'] > 1]
    print('drop %d unclassified neurons'%(summary_df[op.classby].isnull().sum()))
    summary_df = summary_df[summary_df[op.classby].notnull()]
    summary_df = summary_df.reset_index(drop=True)
    
    classes = sorted(summary_df[op.classby].unique().tolist())
    classN = len(classes)
    print('categorized by %s:'%(op.classby), classes)
    multi_factor = int(np.ceil(classN / len(op.facecolor)))
    if multi_factor > 1: 
        print('Repeated colors were used in plot.')
        op.facecolor *= multi_factor
    op.facecolor = op.facecolor[:classN]
    legend_handles = [mp.Patch(color=op.facecolor[i],label=classes[i]) for i in range(len(classes))]
    lower = int(np.sqrt(classN))
    upper = int(np.ceil(np.sqrt(classN)))
    if lower**2 <= classN <= lower*upper:
        rowN = lower
        colN = upper
    elif lower*upper < classN < upper**2:
        rowN = upper
        colN = upper
    rowN = max(rowN,2)
    colN = max(colN,2)
    print("subplot size: rowN = %d,colN = %d"%(rowN,colN))
    
    fig_sup, axes = plt.subplots(nrows=rowN,ncols=colN,sharex=True,sharey=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    np.vectorize(lambda axes:axes.axis('off'))(axes)
    fig_sup.suptitle(op.title+'_subplots')
    ellipses = []
    for i,cla in enumerate(classes):
        df = summary_df[summary_df[op.classby] == cla]
        ellipse_class = []
        ax_x = i % rowN
        ax_y = int(i / rowN)
        # print("subplot pos: row = %d,col = %d"%(ax_x,ax_y))
        for ind in df.index:
            if op.toPlot == 'soma':
                somaLoc_str = df.at[ind,'somaLocation'][1:-1].split(', ')
                name_str = 'xyz'
                somaLoc = {name_str[i]: int(somaLoc_str[i]) for i in range(3)}
                e = mp.Circle(xy = (somaLoc[op.plane[0]], somaLoc[op.plane[1]]),
                        radius = df.at[ind,'somaRadius'],
                        alpha = op.alpha,
                        facecolor = op.facecolor[i],
                )
                ellipse_class.append(e)
                ellipses.append(copy(e))
            elif op.toPlot == 'synapse_distribution':
                e = mp.Ellipse(xy = (df.at[ind,'centroid_'+op.plane[0]], df.at[ind,'centroid_'+op.plane[1]]),
                            width = df.at[ind,'error_'+op.plane[0]] * 2,
                            height = df.at[ind,'error_'+op.plane[1]] * 2,
                            angle = 0,
                            alpha = op.alpha,
                            facecolor = op.facecolor[i],
                )
                ellipse_class.append(e)
                ellipses.append(copy(e))
            elif op.toPlot == 'synapse':
                bodyid = int(df.at[ind,'bodyId'])
                snp_info = snp_file.parse(str(bodyid))
                if op.site != None:
                    snp_info = snp_info[snp_info.type == op.site]
                if op.snp_rois != None:
                    snp_info = snp_info[snp_info.roi.isin(op.snp_rois)]
                if op.confidence != None:
                    snp_info = snp_info[snp_info.confidence >= op.confidence]
                for ind in snp_info.index:
                    x = snp_info.at[ind,op.plane[0]]
                    y = snp_info.at[ind,op.plane[1]]
                    e = mp.Circle(xy=(x,y),
                                    radius=op.synapseRadius,
                                    alpha=op.alpha,
                                    facecolor=op.facecolor[i])
                    ellipse_class.append(e)
                    ellipses.append(copy(e))     
        for e in ellipse_class:
            axes[ax_x,ax_y].add_artist(e)
        navis.plot2d(roimesh,method='2d',ax=axes[ax_x,ax_y],view=(op.plane[0],op.plane[1]),color=op.mesh_color,alpha=op.mesh_alpha)
        axes[ax_x,ax_y].set_ylim(*op.ylim)
        axes[ax_x,ax_y].set_xlim(*op.xlim)
        axes[ax_x,ax_y].legend(handles=[legend_handles[i]],fancybox=True,framealpha=0)
        axes[ax_x,ax_y].set_alpha(0)
    fig_sup.savefig(op.save_path+'_sup'+op.save_format,transparent=True)
    if not op.showfig: plt.close(fig_sup)
    
    fig, ax = plt.subplots(tight_layout=True,dpi=op.dpi,subplot_kw={'aspect': 'equal'})
    
    fig.suptitle(op.title)
    for i,e in enumerate(ellipses):
        ax.add_artist(e)
    navis.plot2d(roimesh,method='2d',ax=ax,view=(op.plane[0],op.plane[1]),color=op.mesh_color,alpha=op.mesh_alpha) #########################################
    ax.set_ylim(*op.ylim)
    ax.set_xlim(*op.xlim)
    ax.legend(handles=legend_handles,fancybox=True,framealpha=0)
    ax.set_alpha(0)
    ax.set_axis_off()
    fig.savefig(op.save_path+op.save_format,transparent=True)
    if not op.showfig: plt.close(fig)
    
def fetchSynapseData(file,noi_df,start_point=0,mode='w'):
    index_to_process = noi_df.index[start_point:]
    for ind in index_to_process:
        bodyid = noi_df.at[ind,'bodyId']
        # snp_info_raw = fetch_synapses(bodyid)
        snp_info_raw = fetch_synapses(bodyid)
        with pd.ExcelWriter(file,mode=mode) as snp_writer:
            snp_info_raw.to_excel(snp_writer,sheet_name=str(bodyid))
        if mode == 'w': mode = 'a'
        print('\rfetched synapses: ','{:.2%}'.format((ind+1)/len(noi_df)),end='  ')
    print()

def getSynapses(snp_file_path,noi_df):
    isDataComplete = False
    isDataExist = False
    if os.path.isfile(snp_file_path):
        snp_excel = pd.ExcelFile(snp_file_path)
        isDataExist = True
        if len(snp_excel.sheet_names) == len(noi_df.index):
            isDataComplete = True
            print('Data were completed')
    if not isDataComplete:
        if isDataExist:
            p = len(snp_excel.sheet_names)
            print('Incomplete data existed: %d / %d'%(p,len(noi_df.index)))
            fetchSynapseData(snp_file_path,noi_df,start_point=p,mode='a')
        else:
            print('No data existed')
            fetchSynapseData(snp_file_path,noi_df)
    return 0 # data were saved to local directly

def sumSnpInfo(noi_df,info_df,para_df,summary_path,synapse_file_path,**kwargs):
    '''summarize synapse info'''
    options = {
        "snp_rois": None,
        "site": None,
        "confidence": None,
    }
    options.update(kwargs)
    
    if not os.path.isfile(summary_path):
        snp_excel = pd.ExcelFile(synapse_file_path)
        neuinfo_df = noi_df[['bodyId','instance','type','pre','post','somaLocation','somaRadius']]
        col_add = ['soma_x','soma_y','soma_z','centroid_x','centroid_y','centroid_z','error_x','error_y','error_z','snpN_roi','ratio']
        neuinfo_df = pd.concat([neuinfo_df,pd.DataFrame(columns=col_add)])
        for ind in neuinfo_df.index:
            bodyid = int(neuinfo_df.at[ind,'bodyId'])
            snp_info_raw = snp_excel.parse(str(bodyid))
            snp_info = snp_info_raw.copy()
            if options['snp_rois'] != None:
                if type(options['snp_rois']) == str:
                    options['snp_rois'] = [options['snp_rois']]
                snp_info = snp_info[snp_info.roi.isin(options['snp_rois'])]
            if options['site'] != None:
                snp_info = snp_info[snp_info.type == options['site']]
            if options['confidence'] != None:
                snp_info = snp_info[snp_info.confidence >= options['confidence']]
            centroid = snp_info[['x','y','z']].mean().tolist()
            errors = snp_info[['x','y','z']].std().tolist()
            
            if pd.notnull(neuinfo_df.at[ind,'somaLocation']):
                somaLoc_str = neuinfo_df.at[ind,'somaLocation'][1:-1].split(', ')
                somaLoc = [int(i) for i in somaLoc_str]
                neuinfo_df.loc[ind,['soma_x','soma_y','soma_z']] = somaLoc
            neuinfo_df.loc[ind,['centroid_x','centroid_y','centroid_z']] = centroid
            neuinfo_df.loc[ind,['error_x','error_y','error_z']] = errors
            neuinfo_df.at[ind,'snpN_roi'] = len(snp_info) # synapse number in the rois
            neuinfo_df.at[ind,'ratio'] = len(snp_info) / neuinfo_df.at[ind,options['site']] # proportion of synapses in the roi
            print('\rprocessing synapse info...','{:.2%}'.format((ind+1)/len(neuinfo_df)),end='  ')
        print()
        snp_summary_df = neuinfo_df.merge(info_df)
        with pd.ExcelWriter(summary_path) as w:
            para_df.to_excel(w,sheet_name='parameters')
            snp_summary_df.to_excel(w,sheet_name='snp_df')
    else:
        print('Processed synapse summary existed, please check the ROIs!')
        snp_summary_df = pd.read_excel(summary_path,sheet_name='snp_df',index_col=0,header=0)
    return snp_summary_df

def SankeyDirect(conn_matrix_type,**kwargs):
    options = {
        'file_path': None,
        "node_color": 'rgba(60,100,200,0.5)',
        "link_color": 'rgba(100,150,240,0.2)',
        "pad": 5,
        "thickness": 5,
        "font_size": 12,
        'title': 'Sankey diagram of connection map',
        "showfig": True,
    }
    options.update(kwargs)
    
    source_names = conn_matrix_type.index.to_list()
    target_names = conn_matrix_type.columns.to_list()
    source_names = [str(i) for i in source_names]
    target_names = [str(i) for i in target_names]
    label_names = source_names + target_names # all nodes
    source_list = []
    target_list = []
    value_list = []
    for source_i in range(len(source_names)):
        for target_j in range(len(target_names)):
            source_list.append(source_i)
            target_list.append(target_j+len(source_names))
            value_list.append(conn_matrix_type.iloc[source_i,target_j])

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = options['pad'],
            thickness = options['thickness'],
            line = dict(color = "black", width = 0),
            label = label_names,
            color = options['node_color'],
        ),
        link = dict(
            source = source_list,
            target = target_list,
            value = value_list,
            color = options['link_color'],
        )
    )])
    fig.update_layout(title_text=options['title'],font_size=options['font_size'])
    if options['file_path'] is None:
        options['file_path'] = options['title'] + '.html'
    fig.write_html(options['file_path'], auto_open=options['showfig'])