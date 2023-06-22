# connectome analysis module -- coana
import os
import shutil
import time
import logging
from dataclasses import dataclass, field

import cv2
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
import networkx as nx
import numpy as np
import pandas as pd
import plotly.io
import plotly.graph_objects as go
import seaborn as sns
from neuprint import *
from neuprint.utils import connection_table_to_matrix

sns.set()
from copy import copy
from datetime import datetime
from types import SimpleNamespace

import bokeh.palettes
import img2pdf

import statvis as sv

# Ignore the navis warning
logging.getLogger('navis').setLevel(logging.WARNING)

@dataclass
class FindNeuronConnection:
    '''
    Through the neuprint-python API, visit the hemibrain database for connectome data analysis:\n
    https://github.com/connectome-neuprint/neuprint-python \n
    https://connectome-neuprint.github.io/neuprint-python/docs \n
    see also the following links for more information:\n
    https://github.com/connectome-neuprint/neuPrintExplorer \n
    https://neuprint.janelia.org \n
    '''
    
    script_path: str = os.path.dirname(os.path.abspath(__file__))
    '''current absolute path of the script'''
    
    data_folder: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
    
    save_folder: str = '' # initialized in InitializeNeuronInfo()
    '''folder to save the current data'''
    
    server: str = 'https://neuprint.janelia.org'
    '''the neuprint server to visit, see https://neuprint.janelia.org for more information'''
    
    dataset: str = 'hemibrain:v1.2.1'
    '''
    the hemibrain dataset to visit, see https://neuprint.janelia.org for more information
    All available datasets are listed below:
    'fib19:v1.0', 
    'hemibrain:v0.9', 
    'hemibrain:v1.0.1', 
    'hemibrain:v1.1', 
    'hemibrain:v1.2.1', 
    'manc:v1.0'
    '''
    
    token: str = ''
    '''
    provide your own user token for accessing the hemibrain dataset\n
    visit https://neuprint.janelia.org to get your own Auth Token, you can find it in your account information
    '''
    
    client_hemibrain: Client = None
    '''neuprint client'''
    
    sourceNeurons: list = field(default_factory=list)
    '''
    Source neurons to find connection. All neurons in the list will be treated as a single source neuron group.\n
    Can be a list of neuron types or a list of neuron bodyIds, but must be a list even if only one item is in the list.\n
    All items in the list should be in the same category, that is, all types or all bodyIds.\n
    To search for all neurons, use None as input.\n
    To search for all neurons having a given type, use empty list [] as input.\n
    e.g. ['MBON01', MBON02', 'MBON03'] # neuron types\n
    e.g. ['MBON.*'] or ['MBON.*_R'] or ['.*_.*PN.*'] ... # all neurons whose type matches the regular expression\n
    e.g. [12345, 23456, 34567] # neuron bodyIds\n
    e.g. None # all neurons in the dataset\n
    e.g. [ ] or list() # all neurons having a given type\n
    All types of neurons can be found in the corresponding hemibrain dataset.\n
    see https://neuprint.janelia.org for more information.\n
    '''
    
    targetNeurons: list = field(default_factory=list)
    '''
    target neurons to find connection\n
    same as sourceNeurons
    '''
    
    largeTargetSet: bool = False
    '''if the target neuron set contains more than 16383 neurons (largeTargetSet will be set True), write excel transposed'''
    
    min_synapse_num: int = 10
    '''minimum number of synapses to be considered as connection'''
    
    min_traversal_probability: float = 0.001
    '''
    minimum traversal probability to be considered as connection\n
    traversal probability is calculated as \n
    max{1, w_ij / (W_j*0.3)}\n
    where w_ij is the number of synapses from neuron i to neuron j and W_j is the total number of post-synaptic sites of neuron j
    '''
    
    max_interlayer: int = 2
    '''maximum number of interlayers to be considered in connection'''
    
    run_date: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    '''date and time when the script is run'''
    
    source_fname: str = ''
    '''auto-generated file name for source neurons'''
    
    source_criteria: NeuronCriteria = None
    '''auto-generated neuron criteria for source neurons'''
    
    target_criteria: NeuronCriteria = None
    '''auto-generated neuron criteria for target neurons'''
    
    target_fname: str = ''
    '''auto-generated file name for target neurons'''
    
    custom_source_name: str = ''
    '''custom name for source neurons, used in plot and file name'''
    
    custom_target_name: str = ''
    '''custom name for target neurons, used in plot and file name'''
    
    parameter_dict = dict()
    '''dictionary to store all specified parameters'''
    
    parameter_df = pd.DataFrame()
    '''dataframe to store all specified parameters, converted from parameter_dict'''
    
    showfig: bool = True
    '''whether to show the figures'''
    
    link_color: str = 'rgba(100,150,240,0.2)'
    '''link color for Sankey diagram'''
    
    node_color: str = 'rgba(60,100,200,0.5)'
    '''node color for Sankey diagram'''
    
    target_color: str = 'rgba(120,40,70,0.7)'
    '''target node color for Sankey diagram, only works when interlayers exist'''
    
    default_mesh_rois = ['LH(R)','AL(R)','EB']
    '''default mesh rois to be plotted'''
    
    keyword_in_path_to_remove: str = 'None'
    '''path blocks including these keywords will be removed'''
    
    simple_fetch: bool = True
    '''
    when True, use neuprint.fetch_simple_connections() to fetch connections, for small sets of neurons and fast speed\n
    when False, use neuprint.fetch_adjacencies(), for large sets of neurons but slower
    '''
    
    def __post_init__(self):
        print('Initializing...')
        if self.client_hemibrain is None:
            self.client_hemibrain = Client(self.server, self.dataset, self.token)
        print(f'\rLogged in to {self.dataset}')
        
        if self.sourceNeurons is None or self.targetNeurons is None:
            print('\033[33mIt is not recommended to search for all neurons in the dataset.\n Using [] or list() to search for all neurons having a given type, instead.\033[0m')
        elif self.targetNeurons is None:
            self.largeTargetSet = True
    
    def InitializeNeuronInfo(self):
        ''' initialize neuron info '''
        print('Fetching source and target neurons...')
        self.source_df, _, source_fname_auto, self.source_criteria = sv.getNeurons(self.sourceNeurons)
        self.target_df, _, target_fname_auto, self.target_criteria = sv.getNeurons(self.targetNeurons)
        
        if self.max_interlayer > 2 or len(self.source_df) > 200:
            self.simple_fetch = False
            print('\033[33mLarge data detected!!! simple_fetch is set to False, using fetch_adjacencies()\033[0m')

        if len(self.target_df) > 16383: # 16383 is the maximum number of excel sheet rows
            self.largeTargetSet = True
        
        if self.custom_source_name:
            self.source_fname = self.custom_source_name
        else:
            self.source_fname = source_fname_auto
        
        if self.custom_target_name:
            self.target_fname = self.custom_target_name
        else:
            self.target_fname = target_fname_auto
        
        print('Processing:',self.source_fname,'to',self.target_fname)
        print(f'Source neurons ({self.source_fname}) in processing: {len(self.source_df)}')
        print(f'Target neurons ({self.target_fname}) in processing: {len(self.target_df)}')
        
        if not self.save_folder: # if save_folder is not specified, save in data_folder, with auto-generated name
            self.save_folder = os.path.join(self.data_folder, self.source_fname + '_to_' + self.target_fname)
        elif not os.path.isabs(self.save_folder): # if save_folder is not absolute path, save in data_folder with specified relative path and name
            self.save_folder = os.path.join(self.data_folder, self.save_folder)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        print(f'data will be saved in: {self.save_folder}\n')
        
        self.parameter_dict = {
            'source neurons': str(self.sourceNeurons),
            'source name': self.source_fname,
            'target neurons': str(self.targetNeurons),
            'target name': self.target_fname,
            'min synapse number': str(self.min_synapse_num),
            'min traversal probability': str(self.min_traversal_probability),
            'max interlayer': str(self.max_interlayer),
            'keyword in path to remove': self.keyword_in_path_to_remove,
            'server': self.server,
            'dataset': self.dataset,
            'run date': self.run_date,
        }
        
        # write parameters to txt file
        self.parameter_df = pd.DataFrame.from_dict(self.parameter_dict, orient='index', columns=['value'])
        self.parameter_df.reset_index(inplace=True)
        self.parameter_df.columns = ['parameter','value']
        self.parameter_txt = os.path.join(self.save_folder,'parameters.txt')
        with open(self.parameter_txt, 'w') as f: 
            f.write(f'Parameters for processing {self.source_fname} to {self.target_fname}:\n')
            for key, value in self.parameter_dict.items():
                keylen = len(key)
                f.write(f'{key}:{" "*(30-keylen)}{value}\n')
            f.write('\n')
    
    def PrintROIHierarchy(self):
        '''print the ROI hierarchy, with primary ROIs marked with *'''
        # Show the ROI hierarchy, with primary ROIs marked with '*'
        print('*: Primary ROI')
        print(fetch_roi_hierarchy(False, mark_primary=True, format='text'))
            
    def FindDirectConnections(self, full_data=False):
        '''
        find direct connections between source and target neurons
        full_data: whether to save the full connection table, if False, only visualize in heatmap, if True, run clustering and ...
        '''
        self.direct_folder = os.path.join(self.save_folder, f'direct_{self.min_synapse_num}snp')
        if not os.path.exists(self.direct_folder): os.makedirs(self.direct_folder)
        # fetch connection table
        if self.simple_fetch:
            self.conn_df: pd.DataFrame = fetch_simple_connections(upstream_criteria=self.source_criteria, downstream_criteria=self.target_criteria, min_weight=self.min_synapse_num)
        else:
            neuron_df, roi_conn_df = fetch_adjacencies(sources=self.source_criteria,targets=self.target_criteria,min_total_weight=self.min_synapse_num)
            self.conn_df = sv.merge_conn_roi(neuron_df, roi_conn_df)
        if self.conn_df.empty:
            print('\033[33mNo direct connections found.\033[0m\n')
            return
        # enrich connection information
        self.conn_df, self.conn_type = sv.EnrichConnectionTable(self.conn_df, self.min_traversal_probability)
        # fill empty values
        self.conn_df = self.conn_df.fillna("")
        self.source_df = self.source_df.fillna("")
        self.target_df = self.target_df.fillna("")
        print(f'Found connected neuron pairs: {len(self.conn_df)}')
        print(f'Total synapses between {self.source_fname} and {self.target_fname}: {self.conn_df.weight.sum()}')
        # convert connection table to matrix
        self.conn_matrix_bodyId: pd.DataFrame = connection_table_to_matrix(self.conn_df, 'bodyId', sort_by='type')
        self.conn_matrix_bodyId.index = self.conn_matrix_bodyId.index.astype(str)
        self.conn_matrix_bodyId.columns = self.conn_matrix_bodyId.columns.astype(str)
        self.conn_matrix_type: pd.DataFrame = connection_table_to_matrix(self.conn_df, 'type', sort_by='type')
        self.conn_matrix_type.index = self.conn_matrix_type.index.astype(str)
        self.conn_matrix_type.columns = self.conn_matrix_type.columns.astype(str)
        self.cmat_full_bodyId,self.cmat_full_type = sv.Conn2FullMat(self.source_df,self.target_df,self.conn_df,self.conn_type)
        self.transitionMat_bodyId,self.transitionMat_type = sv.Conn2FullMat(self.source_df,self.target_df,self.conn_df,self.conn_type,weight_col='traversal_probability')
        # 
        self.source_in_conn: pd.DataFrame = self.source_df[self.source_df['bodyId'].isin(self.conn_df['bodyId_pre'].unique())]
        self.source_in_conn = self.source_in_conn.reset_index(drop=True)
        self.target_in_conn: pd.DataFrame = self.target_df[self.target_df['bodyId'].isin(self.conn_df['bodyId_post'].unique())]
        self.target_in_conn = self.target_in_conn.reset_index(drop=True)
        print(f'{len(self.source_in_conn)} / {len(self.source_df)} source neurons involved in connections')
        print(f'{len(self.target_in_conn)} / {len(self.target_df)} target neurons involved in connections')
        with open(self.parameter_txt, 'a') as f:
            f.write(f'{len(self.source_in_conn)} / {len(self.source_df)} source {self.source_fname} neurons involved in connections\n')
            f.write(f'{len(self.target_in_conn)} / {len(self.target_df)} target {self.target_fname} neurons involved in connections\n')
            f.write('\n')
        
        output_excel_name = os.path.join(self.direct_folder,self.source_fname+'_to_'+self.target_fname+'_info_snp'+str(self.min_synapse_num)+'.xlsx')
        print(f'Saving connection info to excel file...')
        with pd.ExcelWriter(output_excel_name, mode='w', engine='xlsxwriter') as dataWriter:
            self.parameter_df.to_excel(dataWriter,sheet_name='parameters')
            worksheet = dataWriter.sheets['parameters']
            worksheet.set_column('A:A', 30, dataWriter.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            worksheet.set_column('B:B', 30, dataWriter.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            self.source_df.to_excel(dataWriter,sheet_name='source_info')
            self.target_df.to_excel(dataWriter,sheet_name='target_info')
            self.source_in_conn.to_excel(dataWriter,sheet_name='source_in_connection')
            self.target_in_conn.to_excel(dataWriter,sheet_name='target_in_connection')
            self.conn_df.to_excel(dataWriter,sheet_name='connection_info')
            self.conn_type.to_excel(dataWriter,sheet_name='connection_groupby_type')
            if not self.largeTargetSet:
                self.conn_matrix_bodyId.to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_type.to_excel(dataWriter,sheet_name='connectionMatrix_type')
                self.cmat_full_bodyId.to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                self.cmat_full_type.to_excel(dataWriter,sheet_name='connMat_type_full')
                self.transitionMat_bodyId.to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                self.transitionMat_type.to_excel(dataWriter,sheet_name='transmissionMat_type')
            else:
                self.conn_matrix_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_bodyId.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_bodyId')
                self.conn_matrix_type.transpose().to_excel(dataWriter,sheet_name='connectionMatrix_type')
                self.cmat_full_bodyId.transpose().to_excel(dataWriter,sheet_name='connMat_bodyId_full')
                self.cmat_full_type.transpose().to_excel(dataWriter,sheet_name='connMat_type_full')
                self.transitionMat_bodyId.transpose().to_excel(dataWriter,sheet_name='transmissionMat_bodyId')
                self.transitionMat_type.transpose().to_excel(dataWriter,sheet_name='transmissionMat_type')
        print('Done\n')
        self.VisualizeDirectConnections_simple()
        if full_data:
            self.VisualizeDirectConnections_complex()
        return 0
        
    def VisualizeDirectConnections_simple(self):
        # Visualize connection matrix in heatmap
        print('Visualizing connection matrix in heatmap...')
        sv.VisConnMat(self.conn_matrix_bodyId,
                title='heatmap of connection matrix: ' + self.source_fname + ' to ' + self.target_fname + '<br>based on bodyId',
                filename=os.path.join(self.direct_folder,'heatmap_connMatrix_bodyId_snp'+str(self.min_synapse_num)+'.html'),
                color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],showfig=self.showfig)
        sv.VisConnMat(self.conn_matrix_type,
                title='heatmap of connection matrix: ' + self.source_fname + ' to ' + self.target_fname + '<br>based on type',
                filename=os.path.join(self.direct_folder,'heatmap_connMatrix_type_snp'+str(self.min_synapse_num)+'.html'),
                color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],showfig=self.showfig)
        sv.VisConnMat(self.transitionMat_bodyId,
                title='heatmap of full transmission matrix: ' + self.source_fname + ' to ' + self.target_fname + '<br>based on bodyId',
                filename=os.path.join(self.direct_folder,'heatmap_transmissionMat_bodyId_snp'+str(self.min_synapse_num)+'.html'),
                color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(14,83,13)']],showfig=self.showfig)
        sv.VisConnMat(self.transitionMat_type,
                title='heatmap of full transmission matrix: ' + self.source_fname + ' to ' + self.target_fname + '<br>based on type',
                filename=os.path.join(self.direct_folder,'heatmap_transmissionMat_type_snp'+str(self.min_synapse_num)+'.html'),
                color_scale=[[0, 'rgb(255,255,255)'], [1, 'rgb(104,55,164)']],showfig=self.showfig)
        print('Done')
        # Visualize by sankey diagram and network graph, only for neuron type
        print('Visualizing by Sankey diagram and network graph...')
        sankey_name = 'sankey_type_snp'+str(self.min_synapse_num)+'.html'
        sv.SankeyDirect(self.conn_matrix_type,file_path=os.path.join(self.direct_folder,sankey_name),showfig=self.showfig,node_color=self.node_color,link_color=self.link_color)
        # sv.NetworkVis(self.source_df,self.target_df,self.conn_type,save_path=self.direct_folder,by='type',showfig=self.showfig,save_format='.svg')
        print('Done\n')
    
    
    def VisualizeDirectConnections_complex(self):
        '''plot connection distribution, clustering, normalized cluster, 2-D sorting by maximums'''
        
        # Visualize connection distribution
        print('plotting connection distribution...')
        save_path = os.path.join(self.direct_folder,'connection distribution')
        if not os.path.exists(save_path): os.makedirs(save_path)
        sv.VisConnDist(self.conn_matrix_type,save_path,suffix='type',showfig=self.showfig)
        sv.VisConnDist(self.conn_matrix_bodyId,save_path,suffix='bodyId',showfig=self.showfig)
        print('Done')
        
        ## clustering
        save_path = os.path.join(self.direct_folder,'clustering')
        save_format = '.svg'
        if not os.path.exists(save_path): os.makedirs(save_path)
        # clustering by type
        print('clustering by type...')
        _,matt_n = sv.ClusterMap(self.conn_matrix_type,cmap='Blues',filename=os.path.join(save_path,'cluster_type_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        _,matt_col = sv.ClusterMap(self.conn_matrix_type,zs=0,filename=os.path.join(save_path,'cluster_type_normCol_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig) # normalize vertically
        _,matt_row = sv.ClusterMap(self.conn_matrix_type,zs=1,filename=os.path.join(save_path,'cluster_type_normRow_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig) # normalize horizontally
        print('Done')
        # clustering by bodyId
        print('clustering by bodyId...')
        _,matb_n = sv.ClusterMap(self.conn_matrix_bodyId,cmap='Blues',scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        _,matb_col = sv.ClusterMap(self.conn_matrix_bodyId,zs=0,scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_normCol_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        _,matb_row = sv.ClusterMap(self.conn_matrix_bodyId,zs=1,scale_ratio=9,filename=os.path.join(save_path,'cluster_bodyId_normRow_snp'+str(self.min_synapse_num)+save_format),showfig=self.showfig)
        print('Done')
        
        # save clustered matrix
        print('saving clustered matrix...')
        with pd.ExcelWriter(os.path.join(save_path,'clustered_mat.xlsx')) as dataWriter:
            matt_n.to_excel(dataWriter,sheet_name='cluster_type')
            matt_col.to_excel(dataWriter,sheet_name='cluster_type_normCol')
            matt_row.to_excel(dataWriter,sheet_name='cluster_type_normRow')
            matb_n.to_excel(dataWriter,sheet_name='cluster_bodyId')
            matb_col.to_excel(dataWriter,sheet_name='cluster_bodyId_normCol')
            matb_row.to_excel(dataWriter,sheet_name='cluster_bodyId_normRow')
        print('Done\n')
        
        ## 2-D sorting by maximums
        print('2-D sorting by maximums...')
        save_path = os.path.join(self.direct_folder,'Expansion or Convergence')
        if not os.path.exists(save_path): os.makedirs(save_path)
        
        sourceMR_ranges = [[0.7,1],[0,0.7]]
        sourceN_ranges = [[1,1],[2,np.Inf]]
        targetMR_ranges = [[0.7,1],[0,0.7]]
        targetN_ranges = [[1,1],[2,np.Inf]]
        # for source neurons
        print('sorting source neurons...')
        for rr in sourceMR_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='source max ratio range (type): '+str(rr),suffix='type',by='sourceMR',filt_range=rr,clusterFlag=False,showfig=False)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='source max ratio range (bodyId): '+str(rr),suffix='bodyId',by='sourceMR',filt_range=rr)
        for rr in sourceN_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='source neuron number range (type): '+str(rr),suffix='type',by='sourceN',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='source neuron number range(bodyId): '+str(rr),suffix='bodyId',by='sourceN',filt_range=rr)
        print('Done')
        # # for target neurons
        print('sorting target neurons...')
        for rr in targetMR_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='target max ratio range (type): '+str(rr),suffix='type',by='targetMR',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='target max ratio range(bodyId): '+str(rr),suffix='bodyId',by='targetMR',filt_range=rr)
        for rr in targetN_ranges:
            sv.sortMatByMax(self.conn_matrix_type,save_path,title='target neuron number range (type): '+str(rr),suffix='type',by='targetN',filt_range=rr)
            sv.sortMatByMax(self.conn_matrix_bodyId,save_path,title='target neuron number range(bodyId): '+str(rr),suffix='bodyId',by='targetN',filt_range=rr)
        print('Done\n')
    
    def FindPath(self, find_bodyId_path=True):
        '''Find path between source and target neurons, adapted from FindInterClusterConnection.ipynb'''
        self.path_folder = os.path.join(self.save_folder,f'paths_{self.max_interlayer}L_{self.min_synapse_num}snp')
        if not os.path.exists(self.path_folder): os.makedirs(self.path_folder)
        targetNum = len(self.target_df)
        self.target_df.insert(loc=0,column='Checked',value=False)
        source_ID = self.source_df['bodyId'].unique() # convert to np.ndarray
        target_ID = self.target_df['bodyId'].unique()
        target_type = self.target_df['type'].unique()
        currLayer = 0
        targetNum_checked = 0
        Flag = True
        conn_layers = []
        searchedNeurons = source_ID
        # searching for target neurons
        while Flag and currLayer <= self.max_interlayer:
            if self.simple_fetch:
                conn_df: pd.DataFrame = fetch_simple_connections(upstream_criteria=NeuronCriteria(bodyId=source_ID),downstream_criteria=None,min_weight=self.min_synapse_num)
            else:
                neuron_df,roi_conn_df = fetch_adjacencies(sources=source_ID,targets=None,min_total_weight=self.min_synapse_num)
                conn_df = sv.merge_conn_roi(neuron_df,roi_conn_df)
            conn_df = sv.removeSearchedNeurons(conn_df,searchedNeurons)
            conn_layers.append(conn_df)
            post_ID = conn_df['bodyId_post'].unique()
            searchedNeurons = np.concatenate((searchedNeurons,post_ID),axis=0)
            print('fetched connections between L%d and L%d %d neurons    connection found: %d pairs'%(currLayer,currLayer+1,len(post_ID),len(conn_df)))
            ind = self.target_df['bodyId'].isin(post_ID)
            self.target_df.loc[ind,'Checked'] = True
            self.target_df.loc[ind,'Layer'] = currLayer + 1
            targetNum_checked = len(self.target_df[self.target_df['Checked'] == True])
            print('Total targets checked: %d / %d neurons'%(targetNum_checked,targetNum))
            if targetNum_checked == targetNum:
                Flag = False
            source_ID = post_ID
            currLayer += 1
            if len(post_ID) == 0:
                print('!!!NO NEURONS FOUND IN NEXT LAYER!!!')
                break
        if Flag: print('\nNOT All Target Neurons Traced')
        else: print('\nAll Target Neurons Traced')
        
        # searching layers
        conn_inpath = pd.DataFrame()
        conn_types = pd.DataFrame()
        post_ID = target_ID
        neuron_layers = [target_ID]
        weight_layers = {} # dict
        
        for i in reversed(range(len(conn_layers))): # searching for connection path from target neurons to source neurons
            conn: pd.DataFrame = conn_layers[i]
            conn_df = conn[conn['bodyId_post'].isin(post_ID)] # remove neurons not in the connection path
            if len(conn_df) == 0: continue # if not found target neurons in the last x searched layers, skip these layers (when max_interlayer is too large)
            conn_df, conn_type = sv.EnrichConnectionTable(conn_df)
            conn_df.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_type.insert(loc=0,column='conn_layer',value=str(i)+'->'+str(i+1))
            conn_inpath = pd.concat([conn_inpath,conn_df])
            conn_types = pd.concat([conn_types,conn_type])
            
            post_ID = conn_df['bodyId_pre'].unique()
            neuron_layers.append(post_ID)
            post_ID = np.concatenate((post_ID,target_ID)) # post ID for next cycle. include target_ID because all target neurons may not be at the last layer
            post_ID = np.unique(post_ID)
            weight_layers.update({str(i)+'->'+str(i+1): conn_df['weight'].sum()})
            
        neuron_layers.reverse()
        conn_inpath = conn_inpath.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_inpath = conn_inpath.reset_index(drop=True)
        conn_types = conn_types.sort_values(by=['conn_layer','traversal_probability','weight'],ascending=[True,False,False])
        conn_types = conn_types.reset_index(drop=True)

        totalweight_df = pd.DataFrame(weight_layers.items(),columns=['conn_layer','weight'])
        totalweight_df = totalweight_df.sort_values(by='conn_layer',ascending=True)

        source_inpath = conn_inpath.loc[conn_inpath.conn_layer=='0->1','bodyId_pre'].unique()
        self.source_df.insert(loc=0,column='isInPath',value=False)
        self.source_df.loc[self.source_df.bodyId.isin(source_inpath),'isInPath'] = True
        
        # saving data
        output_excel_name = os.path.join(self.path_folder,self.source_fname+'_to_'+self.target_fname+'_info_snp'+str(self.min_synapse_num)+'.xlsx')
        with pd.ExcelWriter(output_excel_name,mode='w',engine='xlsxwriter') as writer:
            self.parameter_df.to_excel(writer,sheet_name='parameters',index=False)
            worksheet = writer.sheets['parameters']
            worksheet.set_column('A:A', 30, writer.book.add_format({'bold': True, 'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            worksheet.set_column('B:B', 30, writer.book.add_format({'font_name': 'Arial', 'font_size': 11, 'align': 'left'}))
            
            self.source_df.to_excel(writer,sheet_name='source_neurons')
            self.target_df.to_excel(writer,sheet_name='target_neurons')
            totalweight_df.to_excel(writer,sheet_name='total_weight_layer')
            conn_inpath.to_excel(writer,sheet_name='connection_info')
            conn_types.to_excel(writer,sheet_name='connection_type')
        
        # get connection path (by type)
        path_df_type = pd.DataFrame()
        print('Analyzing path info by type:')
        path_df_type,_ = sv.getAllPath(conn_data = conn_types,
                                    targets = self.target_df.loc[self.target_df.Checked,'type'].unique().tolist(),
                                    traversal_probability_threshold = self.min_traversal_probability)
        path_df_type = sv.split_path(path_df_type)
        path_df_type, path_df_type_excluded = sv.path_filter(path_df_type,self.keyword_in_path_to_remove)
        
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            path_df_type.to_excel(writer,sheet_name='path_type')
            path_df_type_excluded.to_excel(writer,sheet_name='path_type_excluded')
        
        # get connection path (by bodyId)
        if find_bodyId_path:
            path_df_bodyId = pd.DataFrame()
            print('Analyzing path info by bodyId:')
            path_df_bodyId,_ = sv.getAllPath(conn_data = conn_inpath,
                                        targets = self.target_df.loc[self.target_df.Checked,'bodyId'].tolist(),
                                        traversal_probability_threshold = self.min_traversal_probability)
            if len(path_df_bodyId) > 1048575:
                path_df_bodyId = path_df_bodyId.iloc[:1048575,:]
                print('\033[33mWarning: Excel has a limit of 1048576 rows, only the first 1048575 rows are saved.\033[0m')
            with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
                path_df_bodyId.to_excel(writer,sheet_name='path_bodyId')
        
        # save interlayer info to excel
        interlayers = []
        for neurons in neuron_layers[1:]:
            n_df,_ = fetch_neurons(NeuronCriteria(bodyId=neurons))
            interlayers.append(n_df)
        with pd.ExcelWriter(output_excel_name, mode='a', engine='openpyxl') as writer:
            for i in range(len(interlayers)):
                interlayers[i].to_excel(writer,sheet_name='layer_'+str(i+1))
        print('Done\n')
        
        # Sankey diagram parameters
        nodes_type = [conn_types.loc[conn_types.conn_layer=='0->1','type_pre'].unique().tolist()]
        sourcebodyIds = []
        targetbodyIds = []
        weight_bodyId = []
        sourcetypes_ind = []
        targettypes_ind = []
        weight_type = []
        for i in range(len(neuron_layers)-1):
            # type
            conn_type = conn_types[conn_types.conn_layer == str(i)+'->'+str(i+1)]
            node_post = conn_type['type_post'].unique().tolist()
            node_pre = nodes_type[i]
            len_sum = sum([len(layer) for layer in nodes_type])
            len_pre = len_sum - len(node_pre)
            for j in conn_type.index:
                sourcetypes_ind.append(node_pre.index(conn_type.at[j,'type_pre'])+len_pre)
                targettypes_ind.append(node_post.index(conn_type.at[j,'type_post'])+len_sum)
                weight_type.append(conn_type.at[j,'weight'])
            nodes_type.append(node_post)
            # bodyId
            conn_df = conn_inpath[conn_inpath['conn_layer'] == str(i)+'->'+str(i+1)]
            for ind in conn_df.index:
                sourcebodyIds.append(conn_df.at[ind,'bodyId_pre'])
                targetbodyIds.append(conn_df.at[ind,'bodyId_post'])
                weight_bodyId.append(conn_df.at[ind,'weight'])
        node_type = sum(nodes_type,[])
        node_type_color = [self.node_color] * len(node_type)
        for nn,node in enumerate(node_type): # mark target as target_color
            for tar in target_type:
                if node == tar:
                    node_type_color[nn] = self.target_color
                    break
        
        # Sankey diagram by neuron type
        link_type_dict = {
            'source'     : sourcetypes_ind,
            'target'     : targettypes_ind,
            'weight'     : weight_type
        }
        link_type_df = pd.DataFrame(link_type_dict)
        fig_type = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 5,
                thickness = 5,
                line = dict(color = "black", width = 0),
                label = node_type,
                color = node_type_color
            ),
            link = dict(
                source = link_type_df['source'],
                target = link_type_df['target'],
                value = link_type_df['weight'],
                color = self.link_color
            )
        )])
        fig_type.update_layout(title_text='Sankey diagram of connection map<br>based on neuron type',font_size=12)
        fig_type.write_html(os.path.join(self.path_folder,'Sankey_type_snp'+str(self.min_synapse_num)+'.html'), auto_open=self.showfig)
        
        # Sankey diagram by neuron bodyId
        node_bodyId = list(set(sourcebodyIds+targetbodyIds))
        node_bodyId.sort()
        node_bodyId_color = [self.node_color] * len(node_bodyId)
        for nn,node in enumerate(node_bodyId): # mark target as target_color
            for tar in target_ID:
                if node == tar:
                    node_bodyId_color[nn] = self.target_color
                    break
        
        node_df,_ = fetch_neurons(node_bodyId)
        for ind in node_df.index: # convert <NoneType> to str 'None'
            if node_df.at[ind,'type'] == None:
                node_df.at[ind,'type'] = 'None'
        node_bodyId_label = [node_df.at[i,'type']+'_'+str(node_df.at[i,'bodyId']) for i in node_df.index]
        sourcebodyIds_ind = []
        targetbodyIds_ind = []
        for i in range(len(sourcebodyIds)):
            sourcebodyIds_ind.append(node_bodyId.index(sourcebodyIds[i]))
            targetbodyIds_ind.append(node_bodyId.index(targetbodyIds[i]))

        link_bodyId_dict = {
            'source'     : sourcebodyIds_ind,
            'target'     : targetbodyIds_ind,
            'weight'     : weight_bodyId
        }
        link_bodyId_df = pd.DataFrame(link_bodyId_dict)
        fig_bodyId = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 1,
                thickness = 5,
                line = dict(color = "black", width = 0),
                label = node_bodyId_label,
                color = node_bodyId_color,
            ),
            link = dict(
                source = link_bodyId_df['source'],
                target = link_bodyId_df['target'],
                value = link_bodyId_df['weight'],
                color = self.link_color,
            )
        )])
        fig_bodyId.update_layout(title_text='Sankey diagram of connection map<br>based on neuron bodyId',font_size=6)
        fig_bodyId.write_html(os.path.join(self.path_folder,'Sankey_bodyId_snp'+str(self.min_synapse_num)+'.html'), auto_open=self.showfig)
    
    def ROImat(self, requiredNeurons: list = None, folder_name: str = None, site: str = 'post', break_threshod: int = 1e3):
        """ get the distribution matrix of ROI by the given site of neurons.
        
        Only the R hemisphere is considered.
        
        Args:
            requiredNeurons (list, optional): _description_. Defaults to self.sourceNeurons.
            folder_name (str, optional): _description_. Defaults to self.source_fname.
            break_threshod (int, optional): _description_. Defaults to 1e3. synapse number of one neuron, if synapse number is greater than the break_threshod, it will be breaked in the axis.
        """
        
        if requiredNeurons == None:
            requiredNeurons = self.sourceNeurons
        required_criteria, auto_name = sv.getCriteriaAndName(requiredNeurons)
        if folder_name == None or folder_name == '':
            folder_name = auto_name
        print(f'Generating ROI distribution matrix of {folder_name} {site} synaptic sites...')
        neuron_df,roi_count_df = fetch_neurons(required_criteria) # Fetch neuron info from hemibrain server.
        neuron_df: pd.DataFrame = neuron_df
        neuron_df.sort_values(by='type',inplace=True) # The order of neuron_df will be the order in the distribution matrix.
        rpath = os.path.join(self.data_folder, '_'.join(['roi_distribution',folder_name,site]))
        if not os.path.exists(rpath): os.makedirs(rpath)
        
        roi_list = roi_count_df.roi.unique().tolist()
        roi_list.sort()
        roi_name = [] # custom name corresponding to "roi" property
        for roi in roi_list:
            if '(R)' in roi:
                name = roi[:-3]
            else:
                name = roi
            roi_name.append(name)
        
        # generate a template for the roi matrix
        distMat = pd.DataFrame(
            data = np.zeros([len(roi_list),len(neuron_df)],dtype=int),
            index = roi_list,
            columns = neuron_df[['bodyId','type']]
        ) 
        distMat.columns = pd.MultiIndex.from_tuples(distMat.columns) # Column names include both "type" and "bodyId"
        for col in distMat.columns:
            bodyId = col[0]
            temp_df = roi_count_df[roi_count_df.bodyId == bodyId]
            for ind in distMat.index:
                series_snp = temp_df.loc[temp_df.roi == ind, site]
                if not series_snp.empty:
                    snpN = series_snp.iat[0]
                    distMat.at[ind,col] = snpN
        distMat_new = distMat.copy(deep=True)
        columns_name = [] 
        for ind in neuron_df.index: # set the column name to the format, (type)_(bodyId)
            name_i = neuron_df.at[ind,'type'] + '_' + str(neuron_df.at[ind,'bodyId'])
            columns_name.append(name_i)
        distMat_new.index = roi_name
        distMat_new.columns = columns_name
        
        # group by type
        distMat_type = distMat.copy()
        distMat_type.columns = neuron_df.type
        distMat_type.index = roi_name
        distMat_type = distMat_type.groupby(distMat_type.columns, axis=1).sum()

        # group breaked data by type
        distMat_break = distMat_new.copy()
        distMat_break[distMat_break > break_threshod] = break_threshod # traverse plane and break z-axis

        distMat_type_break = distMat_type.copy()
        distMat_type_break = distMat_type_break.groupby(distMat_type_break.columns, axis=1).sum()
        distMat_type_break[distMat_type_break > break_threshod] = break_threshod

        
        print('Saving ROI distribution matrix...')
        file = os.path.join(rpath,'mat_ROI.xlsx')
        with pd.ExcelWriter(file) as w:
            neuron_df.to_excel(w,sheet_name='PN_R_info')
            neuron_df.to_excel(w,sheet_name='neuron_df')
            roi_count_df.to_excel(w,sheet_name='roi_count_df')
            distMat_type.to_excel(w,sheet_name='roi_mat_type')
            distMat_new.to_excel(w,sheet_name='roi_mat')
            distMat_break.to_excel(w,sheet_name='roi_mat_break')
            distMat_type_break.to_excel(w,sheet_name='roi_mat_type_break')
            distMat.to_excel(w,sheet_name='roi_mat_multilevelCol')
        # visualize roi distribution matrix by the VisConnMat function
        print('Visualizing ROI distribution matrix...')
        sv.VisConnMat(distMat_type,os.path.join(rpath,'roi_type_heatmap.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), grouped by type',showfig=self.showfig)
        sv.VisConnMat(distMat_new,os.path.join(rpath,'roi_heatmap.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site})',showfig=self.showfig)
        sv.VisConnMat(distMat_type_break,os.path.join(rpath,'roi_type_break.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), grouped by type, breaks data > {break_threshod}',showfig=self.showfig)
        sv.VisConnMat(distMat_break,os.path.join(rpath,'roi_mat_break.html'),fontsize=8,title=f'roi matrix of {folder_name} ({site}), breaks data > {break_threshod}',showfig=self.showfig)
        print('Done\n')
    
    def SynapseDistribution(self, requiredNeurons=None, folder_name=None, site='post', snp_rois=None, categories=['type'], info_df = pd.DataFrame()):
        """get and synapse distribution, adapted from PlotSynapses.ipynb
        Args:
            requiredNeurons (_type_, optional): _description_. Defaults to None.
            folder_name (_type_, optional): _description_. Defaults to None.
            site (str, optional): _description_. 'pre' or 'post' synaptic site, Defaults to 'post'.
            snp_rois (_type_, optional): _description_. Defaults to None (auto-generated roi list), if given, use the given roi list.
            visualization_threshod (_type_, optional): _description_. Defaults to 1e2. synaptic number threshold for auto-generated roi list
            categories (list, optional): _description_. Defaults to ['type']. other options can be used if info_df is given.
            info_df (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame(). neuron info dataframe, including given categories of classified neurons.
        """        
        
        para_dict = {
            'neurons': str(requiredNeurons),
            'name': str(folder_name),
            'site': site,
            'snp_rois': snp_rois,
            'dataset': self.dataset,
            'server': self.server,
            'run date': self.run_date,
        }
        if requiredNeurons == None:
            requiredNeurons = self.sourceNeurons
            para_dict.update({'neurons': str(requiredNeurons)})
        required_criteria, auto_name = sv.getCriteriaAndName(requiredNeurons)
        if folder_name == None or folder_name == '':
            folder_name = auto_name
            para_dict.update({'name': folder_name})
        rpath = os.path.join(self.data_folder, '_'.join(['synapse_distribution',folder_name,site]))
        if not os.path.exists(rpath): os.makedirs(rpath)
        
        neuron_info_path = os.path.join(rpath,'neuron_info_'+folder_name+'.xlsx')
        if not os.path.isfile(neuron_info_path):
            print('fetching neurons...')
            noi_df, roic_df = fetch_neurons(required_criteria) # neurons of interest, roi_count
            with pd.ExcelWriter(neuron_info_path) as w:
                noi_df.to_excel(w,sheet_name='neuron_df')
                roic_df.to_excel(w,sheet_name='roi_count')
            print('fetched %d neurons'%(len(noi_df)))
        else:
            print('neuron_df existed')
            noi_df = pd.read_excel(neuron_info_path,sheet_name='neuron_df',index_col=0,header=0)
            roic_df = pd.read_excel(neuron_info_path,sheet_name='roi_count',index_col=0,header=0)
            print('read %d neurons'%(len(noi_df)))
        
        if snp_rois is None:
            snp_rois = roic_df.groupby(by=['roi']).sum()
            snp_rois.reset_index(inplace=True)
            snp_rois = snp_rois.sort_values(by=[site],ascending=False).iloc[:4,:]
            snp_rois = snp_rois['roi'].tolist()
            para_dict.update({'snp_rois': snp_rois})
        para_df = pd.DataFrame.from_dict(para_dict, orient='index', columns=['value'])
        snp_file_path = os.path.join(rpath,'synapse_info_' + folder_name + '.xlsx')
        sv.getSynapses(snp_file_path,noi_df) # get synapse info and write to excel file, #snp_file_path
        roi_str = '_'.join(snp_rois)
        summary_path = os.path.join(rpath,'summary_' + folder_name + '_' + roi_str + '_' + site + '.xlsx')
        snp_summary_df = sv.sumSnpInfo(noi_df,para_df,summary_path,snp_file_path,site=site,snp_rois=snp_rois,info_df=info_df)
        
        # plot synapse distribution in each roi in the #para_dict['snp_rois]
        site_info = str(para_dict['site'])
        save_path = os.path.join(rpath,site_info)
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        for roi in para_dict['snp_rois']:
            print()
            summary_path = os.path.join(rpath,'_'.join(['summary',folder_name,roi,para_dict['site']]) + '.xlsx')
            print("current summary path: ", summary_path)
            snp_summary_df = sv.sumSnpInfo(noi_df,para_df,summary_path,snp_file_path,snp_rois=roi,site=para_dict['site'],info_df=info_df)
            pic_names = ['_'.join([folder_name,site_info,roi,suf]) for suf in categories]
            show_mesh_rois = self.default_mesh_rois + [roi]
            show_mesh_rois = sorted(list(set(show_mesh_rois)))
            for i,cla in enumerate(categories):
                sv.Vis3S(data_df=snp_summary_df,
                    save_path=os.path.join(save_path,pic_names[i]),
                    title=pic_names[i],
                    classby=cla,
                    toPlot='synapse_distribution',
                    mesh_roi=show_mesh_rois,
                    site=para_dict['site'],
                    snp_rois=roi,
                    )
        sv.ConcatenateIMG2PDF(save_path)
        
        # plot soma locations
        save_path = os.path.join(rpath,'soma_location')
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        pic_names = [folder_name+'_soma_'+suf for suf in categories]
        if para_dict['snp_rois'] != None:
            show_mesh_rois = self.default_mesh_rois + para_dict['snp_rois']
        else:
            show_mesh_rois = self.default_mesh_rois
        show_mesh_rois = sorted(list(set(show_mesh_rois)))
        show_mesh_rois = self.default_mesh_rois
        for i,cla in enumerate(categories):
            sv.Vis3S(data_df=snp_summary_df,
                save_path=os.path.join(save_path,pic_names[i]),
                title=pic_names[i],
                classby=cla,
                toPlot='soma',
                mesh_roi=show_mesh_rois,
                **para_dict)
        sv.ConcatenateIMG2PDF(save_path)
        
        # plot synapse locations
        site_info = str(para_dict['site'])
        save_path = os.path.join(rpath,site_info+'_synpases')
        print("current path to save data: ", save_path)
        if not os.path.exists(save_path): os.makedirs(save_path)
        pic_names = [folder_name+'_snp_'+site_info+'_'+suf for suf in categories]
        if para_dict['snp_rois'] != None:
            show_mesh_rois = self.default_mesh_rois + para_dict['snp_rois']
        else:
            show_mesh_rois = self.default_mesh_rois
        show_mesh_rois = sorted(list(set(show_mesh_rois)))
        show_mesh_rois = self.default_mesh_rois
        for i,cla in enumerate(categories):
            sv.Vis3S(data_df=snp_summary_df,
                save_path=os.path.join(save_path,pic_names[i]),
                synapse_file_path=snp_file_path,
                title=pic_names[i],
                classby=cla,
                toPlot='synapse',
                mesh_roi = show_mesh_rois,
                site='pre',
                confidence=0,
                snp_rois=None,
                dpi=600,
                )
        sv.ConcatenateIMG2PDF(save_path)

@dataclass
class VisualizeSkeleton:
    '''3-D visualize skeleton with synapses and brain roi meshes'''

    script_path: str = os.path.dirname(os.path.abspath(__file__))
    '''path to the script folder'''

    data_folder: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
    
    save_folder: str = '' 
    '''
    folder to save the current data
    # initialized in __post_init__, not customizable
    # You can set the "saveas" parameter to customize the folder name'''

    neuron_layers: str | list = ''
    '''
    layers of neurons to plot, can be:
        list of neuron layers: e.g. ['L1', 'L2', 'L3']; or \n
        str of neuron layers separated by '->': \n
        e.g. 'L1->L2->L3'. All type, instance (in regular expression), and bodyId are compatible.\n
    when use list, each layer can be neuron bodyIds, types, instances in regular expressions, or a list of them\n
    e.g. [['L1_0','L1_1'], ['L2_0','L2_1'], ['L3_0','L3_1']]
    '''

    custom_layer_names: list = field(default_factory=list)

    min_synapse_num: int = 10
    '''minimum number of synapses to fetch and plot'''

    saveas: str = None
    '''filename to save the plot, if an absolute path is given, ignore data_folder'''

    neuron_colors: tuple = bokeh.palettes.Category10[10]
    '''
    colors of neuron layers to plot \n
    list of colors, each item for each layer, i.e., item i for layer i, and item i can be a list of colors for each neuron in layer i, or a single color for all neurons in layer i \n
    if you want to assign different colors to different neurons in the same layer, the color list should be the same length as the number of neurons in the layer. \n
    color format: 'red', '#ff0000', (255,0,0), or a dict mapping bodyId to color, {bodyId: color}. \n
    See https://navis.readthedocs.io/en/latest/source/tutorials/generated/navis.plot3d.html#navis.plot3d for more details.
    '''

    neuron_alpha: float = 0.2
    '''alpha of neuron, only works when the radius of neuron exists (show_skeleton_radius=True)'''

    synapse_colors: tuple = bokeh.palettes.Category10[10]
    '''colors of synapse layers to plot'''

    synapse_size: int = 0
    '''
    size of synapse\n
    when synapse_mode='scatter', 1 to 10 is recommended\n
    when synapse_mode='sphere', 100 is recommended\n
    '''

    synapse_criteria: SynapseCriteria = None
    '''criteria to filter synapses'''

    synapse_mode: str = 'scatter'
    '''
    mode to plot synapses, 'scatter' or 'sphere' \n
    'scatter': plot synapses as scatter points, relative size to the view\n
    'sphere': plot synapses as spheres, absolute size in the figure \n
    '''
    
    synapse_alpha: float = 0.6
    '''alpha of synapse, only works when synapse_mode='sphere' '''

    mesh_roi: list = field(default_factory=list)
    '''
    meshes of brain ROIs to plot\n
    defaultly use ['LH(R)', 'AL(R)', 'EB'] to mark the position of the brain\n
    if you want to show the whole hemibrain, see show_hemibrain_mesh parameter. \n
    Available meshes: \n
    a'L(L) \n
    a'L(R) \n   
    AB(L) \n    
    AB(R) \n    
    AL(L)_ \n   
    AL(R) \n    
    alphaL(L) \n
    alphaL(R) \n
    AME(R) \n   
    AOTU(R) \n  
    ATL(L) \n   
    ATL(R) \n   
    AVLP(R) \n  
    b'L(L) \n   
    b'L(R) \n   
    bL(L) \n    
    bL(R) \n    
    BU(L) \n    
    BU(R) \n    
    CA(L) \n    
    CA(R) \n    
    CAN(R) \n   
    CRE(L) \n   
    CRE(R) \n   
    EB \n       
    EPA(L) \n   
    EPA(R) \n
    FB \n
    FLA(R) \n
    gL(L) \n
    gL(R) \n
    GNG \n
    GOR(L) \n
    GOR(R) \n
    IB \n
    ICL(L) \n
    ICL(R) \n
    IPS(R) \n
    LAL(L) \n
    LAL(R) \n
    LH(R) \n
    LO(R) \n
    LOP(R) \n
    ME(R) \n
    NO \n
    PB \n
    PED(R) \n
    PLP(R) \n
    PRW \n
    PVLP(R) \n
    SAD \n
    SCL(L) \n
    SCL(R) \n
    SIP(L) \n
    SIP(R) \n
    SLP(R) \n
    SMP(L) \n
    SMP(R) \n
    SPS(L) \n
    SPS(R) \n
    VES(L) \n
    VES(R) \n
    WED(R) \n
    '''

    mesh_color: tuple | list = (100, 100, 100, 0.2)
    '''
    color of brain meshes, single color or list of colors matching the length of mesh_roi
    single color: tuple including an alpha channel: (R, G, B, alpha)
    multiple colors: list of tuples, each tuple including an alpha channel: [(R1, G1, B1, alpha1), (R2, G2, B2, alpha2), ...]
    '''

    show_soma: bool = True
    '''whether to show soma'''

    show_fig: bool = True
    '''whether to show the figure'''

    skeleton_mode: str = 'tube'
    '''
    whether to plot the radius of skeleton or only skeleton lines\n
    'tube': plot the radius of skeleton\n
    'line': only plot skeleton lines\n
    when 'line', the file size will be significantly smaller and the rendering will be faster
    '''

    show_connectors: bool = False
    '''whether to fetch and plot the connectors, all pre- and post-synaptic sites of the neurons, for single layer of neurons'''

    use_size_slider: bool = True
    '''
    whether to use size slider to adjust the size of synapses\n
    only works when synapse_mode='scatter'
    '''

    legend_mode: str = 'normal'
    '''
    'normal': show legend for individual neurons\n
    'merge': merge all neurons in the same layer and show legend for each layer\n
    '''
    
    show_hemibrain_mesh: bool = False
    ''' 
    whether to plot the hemibrain mesh, if True, plot the hemibrain mesh, else only plot the meshes in mesh_roi \n
    change the color of hemibrain mesh by hemibrain_mesh_color parameter \n
    '''
    
    hemibrain_mesh_color: str = 'rgba(200, 230, 240, 0.1)'
    ''' 
    color of the hemibrain mesh, only works when show_hemibrain_mesh = True \n
    e.g. 'rgba(200, 230, 240, 0.1)' \n
    see more at https://plotly.com/python/discrete-color/ \n
    '''

    def __post_init__(self):
        if self.synapse_mode not in ['scatter', 'sphere']:
            raise ValueError('synapse_mode can only be "scatter" or "sphere"')
        if self.legend_mode not in ['normal', 'merge']:
            raise ValueError('legend_mode can only be "normal" or "merge"')
        if self.skeleton_mode not in ['line','tube']:
            raise ValueError('skeleton_mode can only be "line" or "tube"')
        
        # convert neuron_layers str to list, if is str
        if type(self.neuron_layers) is str:
            self.neuron_layers = self.neuron_layers.replace(' ','').split('->')
            for i,layer in enumerate(self.neuron_layers): # convert bodyId str to int
                if layer.isnumeric():
                    self.neuron_layers[i] = int(layer)
        
        if self.synapse_mode == 'scatter' and self.synapse_size == 0:
            self.synapse_size = 3
        elif self.synapse_mode == 'sphere' and self.synapse_size < 100:
            self.synapse_size = 100
            if self.use_size_slider:
                self.use_size_slider = False
                print('\033[33msize slider is not available for synapse_mode="sphere", automatically reset to False\033[0m')
            print('\033[33mSynapse size is too small (< 100) for sphere mode, automatically reset to 100\033[0m')
        
        if not self.mesh_roi:
            self.mesh_roi = ['LH(R)','AL(R)','EB']
        
        if len(self.neuron_layers) < len(self.neuron_colors): 
            self.neuron_colors = self.neuron_colors[:len(self.neuron_layers)]
        if len(self.neuron_layers)-1 < len(self.synapse_colors):
            self.synapse_colors = self.synapse_colors[:len(self.neuron_layers)-1]

        if self.skeleton_mode == 'line':
            self.show_skeleton_radius = False
            if self.neuron_alpha < 1:
                self.neuron_alpha = 1
                print('\033[33mneuron_alpha is not available for skeleton_mode="line", automatically reset to 1\033[0m')
        elif self.skeleton_mode == 'tube':
            self.show_skeleton_radius = True
        
        # fetch neurons and automatically generate layer names
        self.neuron_dfs = []
        self.roi_dfs = []
        self.layer_criteria = []
        self.layer_names = []
        for i in range(len(self.neuron_layers)):
            print(f'fetching neuron info of layer {i}...')
            ndf, rdf, auto_name, cri = sv.getNeurons(self.neuron_layers[i])
            self.neuron_dfs.append(ndf)
            self.roi_dfs.append(rdf)
            self.layer_criteria.append(cri)
            self.layer_names.append(auto_name)
        print('Fetched neuron layers')

        
        if self.custom_layer_names:
            self.layer_names = self.custom_layer_names
        if self.saveas is None:
            self.saveas = '_'.join(self.layer_names)
        self.save_folder = os.path.join(self.data_folder, 'plot3d_' + self.saveas.split('.')[0])
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        self.fig_3d = go.Figure()
        
        # save neuron dataframes to excel file
        file_path = os.path.join(self.save_folder, self.saveas+'_neuron_info.xlsx')
        for i in range(len(self.neuron_layers)):
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            with pd.ExcelWriter(file_path,mode=mode,engine='openpyxl') as writer:
                self.neuron_dfs[i].to_excel(writer, sheet_name=f'neuron_df{i}')
                self.roi_dfs[i].to_excel(writer, sheet_name=f'roi_count_df{i}')
    
    def plot_skeleton(self):
        for i in range(len(self.neuron_layers)):
            print(f'fetching skeletons of layer {i}...')
            neuron_vols = neu.fetch_skeletons(self.neuron_dfs[i],with_synapses=self.show_connectors)
            print('plotting...', end='')
            fig_layer = navis.plot3d(
                neuron_vols,
                backend='plotly',
                color=self.neuron_colors[i],
                alpha=self.neuron_alpha,
                soma=self.show_soma,
                # fig=self.fig_3d,
                radius=self.show_skeleton_radius,
                connectors=self.show_connectors,
            )
            fig_traces = fig_layer.data

            for j,trace in enumerate(fig_traces):
                if self.legend_mode == 'merge':
                    if j == 0:
                        trace.showlegend = True
                    else:
                        trace.showlegend = False
                    trace.name = self.layer_names[i]
                    trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'  # show full name in hover tooltip
                    trace.legendgroup = self.layer_names[i]
                    trace.hoverinfo = 'name'
                    self.fig_3d.add_trace(trace)
                elif self.legend_mode == 'normal':
                    trace.hoverinfo = 'name'
                    trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                    self.fig_3d.add_trace(trace)
                else:
                    raise ValueError(f'legend_mode {self.legend_mode} not supported')

            print('Done')
        return 0
    
    def plot_synapses(self):
        file_path = os.path.join(self.save_folder, self.saveas+'_connection_info.xlsx')
        for i in range(len(self.neuron_layers)-1):
            source_criteria = self.layer_criteria[i]
            target_criteria = self.layer_criteria[i+1]
            print(f'\rfetching synapses of layer {i} -> layer {i+1}...')
            conn_df = fetch_synapse_connections(
                source_criteria=source_criteria,
                target_criteria=target_criteria,
                min_total_weight=self.min_synapse_num,
                synapse_criteria=self.synapse_criteria,
            )
            if i == 0:
                mode = 'w'
            else:
                mode = 'a'
            with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl') as writer:
                conn_df.to_excel(writer, sheet_name=f'conn_df{i}_{i+1}')
            
            print('plotting...', end='')
            X = (conn_df['x_pre']+conn_df['x_post'])/2
            Y = (conn_df['y_pre']+conn_df['y_post'])/2
            Z = (conn_df['z_pre']+conn_df['z_post'])/2
            if self.synapse_mode == 'scatter':
                sp = go.Scatter3d(
                    x = X,
                    y = Y,
                    z = Z,
                    mode = 'markers',
                    name = f'synapses {i} -> {i+1} ({len(conn_df)})',
                    hoverinfo = 'name',
                    hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z}<br>name: %{fullData.name}<extra></extra>',
                    legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})',
                    marker = dict(
                        size = self.synapse_size,
                        color = self.synapse_colors[i],
                        symbol = 'circle',
                    ),
                )
                self.fig_3d.add_trace(sp)
            elif self.synapse_mode == 'sphere':
                for ind in range(len(X)):
                    x = X[ind]
                    y = Y[ind]
                    z = Z[ind]
                    sp = sv.build_sphere(x,y,z,r=self.synapse_size,color_scale=[self.synapse_colors[i]]*2,opacity=self.synapse_alpha)
                    sp.name = f'synapses {i} -> {i+1} ({len(conn_df)})'
                    sp.hoverinfo = 'name'
                    sp.legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})'
                    sp.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                    if ind == 0: sp.showlegend = True
                    self.fig_3d.add_trace(sp)
            print('Done')
        return 0
    
    def plot_mesh(self):
        if self.mesh_roi is None:
            return
        roiunits = []
        for roi in self.mesh_roi:
            mesh_file = os.path.join(self.script_path, 'navis_roi_meshes_json','primary_rois',roi+'.json')
            if os.path.exists(mesh_file):
                mesh = navis.Volume.from_json(mesh_file)
                roiunits.append(mesh)
            else:
                print('mesh file %s.json not found!'%(roi))
        # roimesh = navis.Volume.combine(roiunits)
        # roimesh.color = options['mesh_color']
        
        print('plotting mesh of brain regions...')
        for roi_i in range(len(roiunits)):
            if type(self.mesh_color) == list:
                roiunits[roi_i].color = self.mesh_color[roi_i]
            else:
                roiunits[roi_i].color = self.mesh_color
            fig_mesh = navis.plot3d(roiunits[roi_i],backend='plotly')
            mesh_traces = fig_mesh.data
            for ti, trace in enumerate(mesh_traces):
                if self.legend_mode == 'merge':
                    if ti == 0:
                        trace.showlegend = True
                    else:
                        trace.showlegend = False
                    trace.legendgroup = 'roi_mesh'
                elif self.legend_mode == 'normal':
                    trace.showlegend = True
                    trace.legendgroup = self.mesh_roi[roi_i]
                trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'  # show full name in hover tooltip
                trace.hoverinfo = 'name'
                trace.name = 'brain regions [' + self.mesh_roi[roi_i] + '...]'
            self.fig_3d.add_traces(mesh_traces)
        if self.show_hemibrain_mesh:
            import flybrains
            print('generating hemibrain mesh...')
            show_hemibrain_mesh = flybrains.JRCFIB2018Fraw
            fig_hemi = navis.plot3d(show_hemibrain_mesh,backend='plotly')
            hemi_traces = fig_hemi.data
            for trace in hemi_traces:
                trace.showlegend = True
                trace.name = 'hemibrain'
                trace.hoverinfo = 'none'
                trace.color = self.hemibrain_mesh_color
            self.fig_3d.add_traces(hemi_traces)
        print('Done')
        return 0
    
    def merge_mesh(self):
        mesh_units = []
        mesh_list = os.listdir(os.path.join('navis_roi_meshes_json','primary_rois'))
        for roi in mesh_list:
            mesh_file = os.path.join('navis_roi_meshes_json','primary_rois',roi)
            if os.path.exists(mesh_file) and not os.path.basename(mesh_file).startswith('.'):
                mesh = navis.Volume.from_json(mesh_file)
                mesh_units.append(mesh)
            else:
                print('mesh file %s.json not found!'%(roi))
        roimesh = navis.Volume.combine(mesh_units)
        roimesh.to_json(os.path.join('navis_roi_meshes_json','merged.json'))
        return roimesh
    
    def save_figure(self):
        # add sliders
        if self.use_size_slider:
            sliders = [
                dict(
                    active=self.synapse_size,
                    currentvalue={"prefix": "Synapse Size: "},
                    pad={"t": 50},
                    steps=[
                        dict(
                            label=str(size),
                            method="update",
                            args=[{"marker": {"size": size}}]
                        )
                        for size in list(range(0,11))
                    ],
                ),
            ]
        else:
            sliders = []
        
        # set layout
        self.fig_3d.update_layout(
            colorway = self.synapse_colors,
            sliders=sliders,
            scene=dict(
                dragmode='orbit',
                xaxis={'visible':False}, 
                yaxis={'visible':False},
                zaxis={'visible':False},
            ),
            scene_camera=dict(
                up=dict(x=0, y=0, z=-1),
                eye=dict(x=0, y=1.5, z=0),
                center=dict(x=0, y=0, z=0),
            ),
        )

        # save figure
        self.fig_path = os.path.join(self.save_folder,self.saveas)
        print(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
        self.fig_3d.write_html(self.fig_path+'.html',auto_open=self.show_fig)
        self.fig_3d.write_image(self.fig_path+'.png',scale=3)
        print('Done')
    
    def plot_neurons(self):
        self.plot_skeleton()
        self.plot_synapses()
        self.plot_mesh()
        self.save_figure()
        
    def export_video(self, fps=30, rotate_plane='xy', view_distance=1.6, synapse_size=1,**kwargs):
        '''
        export the rotating 3-D object to a video. rendering is slow, it helps to visualize complex objects and the video file is more portable and versatile.
        
        when file is too large, exporting may fail. try to reduce the resolution by setting "scale" in kwargs, or set "width" and "height" to specific values.
        
        fps: default 30
            frames per second, also determines the step size of rotation, 30 degrees per second.
        rotate_plane: default 'xy'
            the plane to rotate the object. can be 'xy', 'xz', 'yz'.
        view_distance: default 1.6,
            the relative distance between the camera and the center of the object.
        synapse_size: default 1,
            the size of the synapse markers.
        **kwargs: other arguments for plotly.offline.plot. see https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html
            In the kwargs, you can use "scale" to set the resolution of the video (e.g. scale=2 doubles the resolution), or set "width" and "height" to specific values.
            recommended values for scale: 2
        '''
        
        kwargs.update(kwargs)
        step = 30 / fps
        html_size = os.path.getsize(self.fig_path+'.html') / 1024 / 1024 # in MB
        if html_size > 100:
            print(f'\033[33mFigure is large. If rendering hangs, try to reduce the resolution by setting "scale", or "width" and "height" in kwargs to smaller values.\033[0m')
        # set layout
        fig_traces = self.fig_3d.data
        for trace in fig_traces:
            trace.showlegend = False
            if hasattr(trace,'marker'):
                trace.marker.size = synapse_size
        fig_layout = go.Layout(
            margin=dict(
                l=1,
                r=1,
                b=1,
                t=1,
                pad=0,
            ),
        )
        fig_new = go.Figure(data=fig_traces, layout=fig_layout)
        fig_new.update_layout(
            sliders=[], # remove sliders
            scene=dict(
                dragmode='orbit',
                xaxis={'visible':False}, 
                yaxis={'visible':False},
                zaxis={'visible':False},
            ),
            scene_camera=dict(
                up=dict(x=0, y=0, z=-1),
                eye=dict(x=0, y=view_distance, z=0),
            ),
        )
        pic_folder = os.path.join(self.save_folder,f'pics_{fps}fps_{rotate_plane}')
        if os.path.exists(pic_folder):
            shutil.rmtree(pic_folder)
        os.makedirs(pic_folder)
        if step > 0:
            steps_to_write = np.linspace(0,360,int(360/step),endpoint=False)
        elif step < 0:
            steps_to_write = np.linspace(360,0,int(360/step),endpoint=False)
        t0 = time.time()
        for i,deg in enumerate(steps_to_write):
            rad_i = np.deg2rad(deg)
            x = view_distance * np.sin(rad_i)
            y = view_distance * np.cos(rad_i)
            if rotate_plane == 'xy':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=0)))
            elif rotate_plane == 'yz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=0, y=x, z=y)))
            elif rotate_plane == 'xz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=0, z=y)))
            fig_path = os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg')
            fig_new.write_image(fig_path,**kwargs)
            cv2.waitKey(2000)
            ti = time.time()
            print(f'\rExporting image: {i+1}/{len(steps_to_write)}...Elapsed {ti-t0:.2f}s. Remaining {(ti-t0)/(i+1)*(len(steps_to_write)-i-1):.2f}s',end='    ')
        print('\nDone')
        imglist = os.listdir(pic_folder)
        img_eg = cv2.imread(os.path.join(pic_folder,imglist[0]))
        height, width, layers = img_eg.shape

        # forward video
        video_dir = os.path.join(self.save_folder,f'{self.saveas}_video_forward.mp4')
        out = cv2.VideoWriter(
            video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize=(width,height))
        for i,deg in enumerate(steps_to_write):
            img = cv2.imread(os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            print(f'\rwriting forward video: {i+1}/{len(steps_to_write)}...',end='  ')
        out.release()
        print('Done')
        # backward video
        video_dir = os.path.join(self.save_folder,f'{self.saveas}_video_backward.mp4')
        out = cv2.VideoWriter(
            video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize=(width,height))
        for i,deg in enumerate(steps_to_write[::-1]):
            img = cv2.imread(os.path.join(pic_folder,f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            print(f'\rwriting backward video: {i+1}/{len(steps_to_write)}...',end='  ')
        out.release()
        print('Done')
        return 0
