# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:43:58 2018

@author: WilliamCh
"""
#import pdb


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from shapely.geometry import Polygon,Point,box,LinearRing,LineString

#from xlrd import open_workbook
#from xlsxwriter.utility import xl_rowcol_to_cell as rc_c
#from xlsxwriter.utility import xl_cell_to_rowcol as c_rc
#from glob import glob
#import json
#import collections


#%%
#In case we have to use hacky workarounds for incomplete python install:
#import os
#os.chdir('C:\\Users\\WilliamCh\\Documents\\Shielding workload') 

#User interface




    

#%%
#Data import and cleaning functions
def match_columns_to_device_list(columns,device_list):
    for i, device in device_list.iterrows():
        if device.iloc[1:].tolist() == columns.tolist():
            break
    device_cols = device.iloc[1:].to_dict()
    device_cols = {v: k for k, v in device_cols.items()}
    return device_cols

def convert_numeric_columns(df,
        numeric_cols = ['Deviation index', 'Clinical EXI', 'Maximum EXI',
                        'Minimum EXI', 'Physical EXI', 'kV', 'mAs',
                        'SID', 'DAP', 'Dose']):
    df[numeric_cols] = df[numeric_cols].replace('[^\d.]+','',regex=True)
    for col in numeric_cols:
        if df[col].dtype == 'O':
            df.loc[df[col]=='',[col]] = 0
            df[col] = df[col].str.strip()
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df
        
def standardise_df_columns(df,device_list_fn = 'device_list.csv'):
    device_list = pd.read_csv('device_list.csv')
    cols = match_columns_to_device_list(df.columns,device_list)
    return df.rename(columns = cols)

def remove_spurious_data(df):
    #Remove any instances of phantom repeat exposures
    df = df.drop_duplicates(['Acq. date','Acq. time','DAP'])
    #Remove any instances where kV is 0
    df = df[df.kV > 30]
    return df

def format_date(df):
    #Format a better timestamp. first add preceeding zeroes:
    df[ 'Acq. time'] = df['Acq. time'].map('{:010}'.format)
    df[ 'Acq. date'] = df['Acq. date'].map('{:08}'.format)
    df['timestamp'] = pd.to_datetime(df['Acq. date'] + df['Acq. time'],format = '%Y%m%d%H%M%S%f')
    return df

def calculate_beam_area(df):
    width_height = df.Collimation.str.split('x')
    df['beam_area'] = width_height.str[0].astype(float) * width_height.str[1].astype(float)/10000
    return df
    
#Format the organ protocol into the mode, organ, and view
def analyse_organ_protocol(df):
    #Remove *, they're doing nothing for us
    OGP = df.OGP.copy()
    OGP = OGP.str.replace('*','')
    det_mode = OGP.str[0]
    OGP_split = OGP.str[2:].str.split(' ')
    organ = OGP_split.str[0]
    view_age_test = OGP_split.str[-1].str.contains('-')
    age = OGP_split.copy()
    age[~view_age_test] = ''
    age[view_age_test] = OGP_split.str[-1]
    OGP_split[view_age_test] = OGP_split[view_age_test].str[:-1]
    view = OGP_split.str[1:].str.join(' ').copy()
    organ = OGP_split.str[0].copy()
    df2 = pd.DataFrame({'OGP':OGP,'det_mode':det_mode,'organ':organ,'view':view,'age':age})
    df[df2.columns] = df2
    return df

#Use an input file to separate free exposures into floor and wall directed
def bin_organ_protocols(df,input_ogp_binning_fn = 'input_ogp.csv'):
    df['det_code'] = df.det_mode
    ogpdf = pd.read_csv(input_ogp_binning_fn,index_col = 'OGP')
    mask = df['det_mode'] == 'X'
    subset = df[mask].copy()
    
    cols = ['mAs','DAP','Dose']
    
    for i in ogpdf:
        temp = subset.copy()
        temp.det_code = i
        if i == 'T':
            temp.SID = df[df.det_mode == 'T'].SID.mean()
        else:
            temp.SID = df[df.det_mode == 'W'].SID.mean()
        multiplier = temp.OGP.map(ogpdf[i].to_dict())
        temp[cols] = temp[cols].multiply(multiplier,axis='index')
        df = df.append(temp)
    
    df = df[df.mAs != 0]
    df = df[df.det_code != 'X']
    
    return df



#Hacky function to convert DAP to Gycm2. Consider finesse.
def convert_DAP_to_Gycm2(df):
    DAPmean = df.DAP.mean()
    if DAPmean > 20:
        df.DAP = df.DAP/10
    return df

def import_data(fn = 'sample_exi_log.csv', device_list_fn = 'device_list.csv'):
    df = (pd.read_csv(fn)
            .pipe(standardise_df_columns,device_list_fn)
            .pipe(calculate_beam_area)
            .pipe(convert_numeric_columns)
            .pipe(remove_spurious_data)
            .pipe(format_date)
            .pipe(analyse_organ_protocol)
            .pipe(bin_organ_protocols)
            .pipe(convert_DAP_to_Gycm2)
            )
    return df
df = import_data()
#%%
#Functions that compute potentially useful statistics and values
def get_usage_during_periods(timestamps,dap):
    start_time = []
    total_dap = []
    hours = timestamps.dt.hour
    for i in range(24):
        m_hours = ((hours > i) & (hours < i + 8)) | ((hours + 24 > i) & (hours +24 < i + 8))
        cumdap = dap[m_hours].sum()
        if cumdap != cumdap:
            cumdap = 0
        total_dap.append(cumdap)
        start_time.append(i)
    return start_time, total_dap

def find_highest_usage_period(timestamps, dap):
    start_time, total_dap = get_usage_during_periods(timestamps, dap)
    argmax = np.array(total_dap).argmax()
    return start_time[argmax],total_dap[argmax]
    
def get_ratio_highest_usage_to_total(timestamps,dap):
    __,dapmax = find_highest_usage_period(timestamps,dap)
    return dapmax/sum(dap)

def get_dose_rescale_factor(df):
    #rescale_factors
    exi_duration_rescale = 7*24*60*60/(df.timestamp.iloc[-1]-df.timestamp.iloc[0]).total_seconds()
    weekend_rescale = 5/7
    busy_period_rescale = get_ratio_highest_usage_to_total(df.timestamp,df.DAP)
    weeks_to_years = 52.143 
    return exi_duration_rescale * weekend_rescale * busy_period_rescale * weeks_to_years
    
def lead_to_weight(thickness,commercial_weight = 0.44):
    return (thickness // commercial_weight + 1) * 5
    
#%%
#OGP stats analysis functions
def get_stats_from_grouped_data(df,group_col,value_cols):
    dfout = pd.DataFrame(columns = ['N_studies'])
    grouped_data = df.groupby(group_col)
    for v in value_cols:
        dfout[v +'_sum'] = grouped_data[v].sum()
        dfout[v +'_mean'] = grouped_data[v].mean()
        dfout[v +'_std'] = grouped_data[v].std()
        dfout[v +'_med'] = grouped_data[v].median()
    dfout['N_studies'] = grouped_data[v].count()
    return dfout

def get_OGP_stats(df):
    value_cols = ['DAP','mAs','kV','SID','Clinical EXI','Physical EXI']
    pivot_col = 'OGP'
    return get_stats_from_grouped_data(df,pivot_col,value_cols)
    
    
    
#%%
#Xraybarr integration functions
def create_xraybarr_spectrum(df,mask = None):
    try:
        if mask.any():
            pass
    except:
        mask = df == df
        
    view = df[mask].copy()
    output_list = []
    #number of patients in spectrum
    N_spectrum = len(view)
    output_list.append('%s   patients in this spectrum' % (N_spectrum))
    
    #number of patient procedures per week
    N_week = len(view)
    output_list.append('%s   # patient procedures done per week' % (N_week))
    
    #bin into 5 kVp increments, starting at 25 kVp
    view['kV_bin'] = view.kV // 5 * 5
    mAs_by_kV = view.groupby(['kV_bin']).mAs.sum()

    for kv in np.arange(25,151,5):
        try:
            mAs = mAs_by_kV.loc[kv]
        except:
            mAs = 0
        output_list.append('%s (mAmin @ %s kVp)' % (mAs,kv))
    #Properly calculate area here...
    weighted_mean_area = (view.mAs*view.beam_area).sum()/view.mAs.sum()
    output_list.append('%d  area of primary beam (cm2) at' % (weighted_mean_area))
    output_list.append('1.0  this primary distance (m)')
    output_list.append('150  leakage technique kVp')
    output_list.append('3.3  leakage technique mA')
    output_list.append('100  leakage exposure rate (mR/hr) at 1 m when operated at leak technique')
    
    return output_list,mAs_by_kV


def save_xraybarr_spectrum(df,mask,output_folder = 'output/',spectrum_name = 'default_spectrum'):
    output_list,mAs_by_kV = create_xraybarr_spectrum(df,mask)
    with open(output_folder+spectrum_name+'.spe','w') as file:
        file.writelines('\n'.join(output_list))
        
def make_xraybarr_spectrum_set(exi_fn,
#                      mask_column = 'det_code',
#                      categories = ['T','W','C','C2'],
#                      category_names = ['Table','Wall','Cross','Cross2'],
                      room_name = 'default',
                      output_folder = 'output/'
                      ):
    df = import_data(exi_fn)
    ratio = get_dose_rescale_factor(df)
    #mAs needs to be in mAmin/week, not year
    df.mAs = df.mAs * ratio / 52.143
    

    
    categories = df.det_code.unique()
    masks = [df.det_code == category for category in categories]
    mask_dict = dict(zip(categories,masks))
    
    Tmask = df.det_code == 'T'
    Tthresh = df[Tmask].beam_area.mean()
    Tlarge = Tmask & (df.beam_area > Tthresh)
    Tsmall = Tmask & (df.beam_area < Tthresh)
    mask_dict['Tlarge'] = Tlarge
    mask_dict['Tsmall'] = Tsmall
    mask_dict.pop('T')
    
    for cat, mask in mask_dict.items():
        save_xraybarr_spectrum(df,mask,output_folder = output_folder,spectrum_name = '_'.join([room_name,cat]))
#%%
#Use one row of df,dfr,dfs,organmap,distancemap to calculate dose
#Implements the BIR method for dose calculations
class Dose:
    def __init__(self,
                 df_fn = 'sample_exi_log.csv',
                 dfr_fn = 'input_rooms.csv',
                 dfs_fn='input_sources.csv',
                 att_coeff_fn='input_shielding_coefficients.csv'):
        
        self.df = import_data(df_fn)
        self.dfr = pd.read_csv(dfr_fn)
        self.dfr['added_attenuation'] = 0
        self.dfr.index = self.dfr.Zone
        
        self.dfs = pd.read_csv(dfs_fn)
        self.dfs = self.dfs.where(self.dfs==self.dfs,'')
        
        self.df_att_coeff = pd.read_csv(att_coeff_fn)
        
        self.dmap = self.make_distancemap(self.dfr,self.dfs)

    #Geometric functions
    def rect(self,p1,p2):
        xmin = min(p1[0],p2[0]) 
        xmax = max(p1[0],p2[0]) 
        ymin = min(p1[1],p2[1]) 
        ymax = max(p1[1],p2[1]) 
        return box(xmin, ymin, xmax, ymax)
    
    def make_rect_column(self,row):
        return self.rect((row['x1'],row['y1']),(row['x2'],row['y2']))
    
    def get_angle_from_lines(self,l1,l2):
        l1 = np.array(l1)
        v1 = l1[0,:]-l1[1,:]
        l2 = np.array(l2)
        v2 = l2[0,:]-l2[1,:]
        return np.arccos(v1.dot(v2)/(sum(v1**2)**.5 * sum(v2**2)**.5))*180/np.pi
    
    def is_room_in_beam(self,beam_line,room_ext):
        l = np.array(beam_line)
        p1 = l[0,:]
        p2 = l[1,:]
        v = p2 - p1
        p2 = p1 + v*1000
        beam_line_extended = LineString([p1,p2])
        return beam_line_extended.intersects(room_ext)
    
    def make_point_column(self,row,xn,yn):
        return Point(row[xn],row[yn])

    def make_distancemap(self,dfr,dfs):
        distancemap = {}
        for i, Srow in dfs.iterrows():
            distancemap[Srow.det_mode] = {}
            for j, Rrow in dfr.iterrows():
                output_dic = {}
                tube_loc = Point(Srow.tubex,Srow.tubey)
                target_loc = Point(Srow.targetx,Srow.targety)
                room_rect = self.rect((Rrow.x1,Rrow.y1),(Rrow.x2,Rrow.y2))
                output_dic['primary_distance'] = room_rect.exterior.distance(tube_loc) + 0.3
                output_dic['secondary_distance'] = room_rect.exterior.distance(target_loc) + 0.3
        #        output_dic['tertiary_distance'] = room_rect.exterior.distance(target_loc)
                
                tube_target_line = LineString([tube_loc,target_loc])
                rect_ext = LinearRing(room_rect.exterior.coords)
                if Srow.floor_target:
                    output_dic['angle'] = 90
                else:
                    room_closest_point_projection = rect_ext.project(target_loc)
                    room_closest_point = rect_ext.interpolate(room_closest_point_projection)
                    target_room_line = LineString([target_loc,room_closest_point])
                    output_dic['angle'] = self.get_angle_from_lines(tube_target_line,target_room_line)
        
                if Srow.floor_target:
                    output_dic['in_p_beam'] = False
                else:
                    output_dic['in_p_beam'] = self.is_room_in_beam(tube_target_line,rect_ext)
                
                distancemap[Srow.det_mode][Rrow.Zone] = output_dic
        return distancemap

    #Transmission calculation functions
    def get_transmission(self, thickness, material, kV):
        a,b,y = self.get_shielding_coefficients(material,kV)
        return ((1 + b/a)*np.exp(a*y*thickness) - b/a)**(-1/y)

    def get_necessary_shielding(self, transmission, material, kV):
        a,b,y = self.get_shielding_coefficients(material,kV)
        x = 1/(a*y)*np.log((transmission**(-y) + (b/a))/(1+(b/a)))
        return x
        
    def get_shielding_coefficients(self, material,kV):
        view = self.df_att_coeff.loc[self.df_att_coeff.Material == material, ['kV','a','b','y']]
        #view.index = view.kV
        xp = np.array(view['kV'])
        yp = np.array(view[['a','b','y']])
        a = np.interp(kV,xp,yp[:,0])
        b = np.interp(kV,xp,yp[:,1])
        y = np.interp(kV,xp,yp[:,2])
        return a,b,y
        
    #Dose calculation functions
    #Take DAP in Gycm^2, area in cm^2, distances in m returns dose in uGy
    def get_primary_dose(self,
                         DAP,
                         beam_area,
                         d_tube_target,
                         d_tube_room,
                         kV,
                         in_beam = False,
                         wall_leq = 0,
                         bucky_leq = 2):
        #DAP is scalar, distance, in_beam and attenuation for each room in array
        transmission = self.get_transmission(wall_leq + bucky_leq,'Lead',kV)
        if in_beam:
            return DAP/beam_area * (d_tube_target/d_tube_room)**2 * 1e6 * transmission
        else:
            return 0
    
    def get_scattered_dose(self,DAP,kV,distance,angle,wall_leq = 0):
        #DAP is scalar, distance, in_beam and attenuation for each room in array
        #In: DAP Gycm^2
        #Out: Dose uGy
        transmission = self.get_transmission(wall_leq,'Lead',kV)
        a = -1.042 * 10**-7
        b = 3.265 * 10**-5
        c = -2.751 * 10**-3
        d = 8.371 * 10**-2
        e = 1.578
        f = 5.987 * 10**-3
        S = (a*angle**4 + b*angle**3 + c*angle**2 + d*angle +e)*((kV-85)*f + 1)
        return S * DAP / distance**2 * transmission
        
    def get_tertiary_dose(self,DAP,distance,ceiling_height,barrier_height,transmission):
        return 0
    
    def get_leakage_dose(self,kV,mAs,distance,transmission):
        return 0
    
    def calculate_dose_for_df_row(self,dfg_row,ignore_attenuation = True):
        dfr = self.dfr
        dose_to_room = {}
#        ceiling_height = None
#        barrier_height = None
        kV = dfg_row.kV.mean()
        DAP = dfg_row.DAP.sum()
#        mAs = dfg_row.mAs.sum()
        beam_area = dfg_row.beam_area.mean()
        SID = dfg_row.SID.mean()/100
#        pdb.set_trace()
        
        for __, receiver_row in dfr.iterrows():
            #Get data for set room
            room_data = self.dmap[dfg_row.det_code.values[0]][receiver_row['Zone']]
            primary_distance = room_data['primary_distance']
            secondary_distance = room_data['secondary_distance']
            angle = room_data['angle']
            in_p_beam = room_data['in_p_beam']
            if ignore_attenuation:
                lead_eq = 0
            else:
                lead_eq = receiver_row.gypsum_thickness/320 + receiver_row.added_attenuation
                lead_eq = max(lead_eq,0)
#                transmission = self.get_transmission(lead_eq,'Lead',kV)
            
            #Use room data to calculate primary and scatter
            room_dose = {}
            room_dose['primary'] = self.get_primary_dose(DAP,beam_area,SID,primary_distance,kV,in_p_beam,lead_eq)
            room_dose['scatter'] = self.get_scattered_dose(DAP,kV,secondary_distance,angle,lead_eq)
#            room_dose['tertiary'] = self.get_tertiary_dose(DAP,secondary_distance,
#                                         ceiling_height,barrier_height,lead_eq)
#            room_dose['leakage'] = self.get_leakage_dose(kV,mAs,primary_distance,lead_eq)
            
            dose_to_room[receiver_row.Zone] = room_dose
    #        dose_to_room[receiver_row.Zone] =        {'dose':dose,'primary':primary,
    #                    'scatter':scatter,'tertiary':tertiary,'leakage':leakage}
        return pd.DataFrame(dose_to_room)
    
    #Returns dose in uGy, provided DAP is given in Gcm^2
    def calculate_dose(self,ignore_attenuation = True):
        df = self.df
        df = df.groupby(['det_code','kV'])
        output = df.apply(self.calculate_dose_for_df_row, ignore_attenuation)
        output = output*get_dose_rescale_factor(self.df)
        output.index = output.index.rename('contribution',level = 2)
        return output
    
    def save_verbose_data(self,ignore_attenuation = True, output_folder = 'output', output_name = 'test'):
        doses = self.calculate_dose()
        
        dfg = self.df.groupby(['det_code','kV'])
        DAP, mAs = dfg.DAP.sum(), dfg.mAs.sum()
        workload = pd.DataFrame([DAP,mAs]).T
        
        dose_short = doses.sum(level=2)
        dose_short = dose_short.T
        dose_short['total'] = dose_short.sum(axis=1)
        dose_short = dose_short.T
        
        factors = pd.DataFrame({'rescale_factor':[get_dose_rescale_factor(self.df)]})
        
        dose_short.to_csv('%s/%s_doses_summary.csv' % (output_folder,output_name))
        workload.to_csv('%s/%s_workload.csv' % (output_folder,output_name))
        doses.to_csv('%s/%s_doses.csv' % (output_folder,output_name))
        factors.to_csv('%s/%s_factors.csv' % (output_folder,output_name))
        return workload,doses,dose_short,factors
        
    
    def get_lead_req(self,
                     constraints = {'controlled':2,'public':0.5},
                     iterations = 1):
        #Start from 0 attenuation
        self.dfr.added_attenuation = 0
        
        #Constraints in mGy/yr
        self.dfr['constraint'] = self.dfr.Category.map(constraints) / self.dfr.Occupancy
        #dose_rescale_factor = get_dose_rescale_factor(self.df)
    
        #Map constraint categories, and convert to mGy/yr given:    
        #def get_lead_required(df,dfr,dfs,distancemap):
        dose_out = []
        lead_out = []
        for i in range(iterations):
            if self.dfr.added_attenuation.any():
                ignore_att = False
            else:
                ignore_att = True
            #Calculate dose
            doses = self.calculate_dose(ignore_att)
            if i == 0:
                self.dfr['raw_dose'] = doses.sum()/1000
                self.dfr['required_transmission'] = self.dfr.constraint/self.dfr.raw_dose/1000
        
            #Converts dose for entire Exi log to dose per year in mGy
            #todo: double check this conversion.
            doses = (doses.sum()).T
            doses = doses / 1000# * dose_rescale_factor
            dose_out.append(doses)
            #Compare to constraints    
            attenuation_required = self.dfr['constraint'] / doses
            #Compute an equivalent lead requirement
            lead = self.get_necessary_shielding(attenuation_required,'Lead',90)
            lead_out.append(lead)
            self.dfr.added_attenuation = self.dfr.added_attenuation + lead
            self.dfr['wall_weight'] = lead_to_weight(self.dfr.added_attenuation)
            
        return self.dfr.added_attenuation,dose_out,lead_out,self.dfr.wall_weight







D = Dose()
#a,b,c,d = D.save_verbose_data()
#df = import_data()
#
a,b,c,d = D.get_lead_req(iterations=3)

#leads = []
#for i in range(12):
#    leads.append(get_lead_req(df,D))
#    D.dfr.added_attenuation = D.dfr.added_attenuation + leads[i]
##    D.dfr[D.dfr]
#leads2 = np.array(leads)
#plt.plot(leads2[1:,:])
#%%
#Sample single 

#Reporting and graphing

def savetext(text,fn = 'test.html'):
    with open(fn,'w') as f:
        f.write(text)

class Report:
    def __init__(self,D,output_folder = 'output/test/'):
        self.D = D
        self.output_folder = output_folder

            
    
    def OGP_workload_plot(self):
        pass
    
        
    def room_lead_table(self):
        '''
        Create a table for reporting, based on dose calculation results
        '''
        headers = {
        'raw_dose':{'head':'Unattenuated dose (mGy/yr)','format':'{:,.1f}'.format},
        'constraint':{'head':'Dose constraint (mGy/yr)','format':'{:,.0f}'.format},
        'required_transmission':{'head':'Barrier transmission','format':'{:,.2f}'.format},
        'added_attenuation':{'head':'Min. lead eq. (mm)','format':'{:,.2f}'.format},
        'wall_weight':{'head':'Barrier lead weight (kg/m^2)','format':'{:,.0f}'.format}}
        
        key_coltitle = {k:headers[k]['head'] for k in headers.keys()}
        key_fmt = {k:headers[k]['format'] for k in headers.keys()}
        coltitle_fmt = {key_coltitle[k]:key_fmt[k] for k in headers.keys()}
        
        table = D.dfr.loc[:,headers.keys()].copy()
#        table.raw_dose = table.raw_dose/1000
    
        self.table = table.rename(columns = key_coltitle)
        self.table.to_excel('')


    def source_workload_plots(self):
        test = pd.pivot_table(df.reset_index(),index = 'kV',columns = 'det_code',values = 'DAP',aggfunc=np.sum)
        kvs = np.arange(test.index.unique().min(),test.index.unique().max()+1)
        test = test.loc[kvs]
        test[test!=test] = 0
        fig,axes = plt.subplots(nrows = 4,sharex = True,figsize = (6,9))
        test.plot(ax = axes,subplots = True,drawstyle="steps")
        #fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        fig.text(0.001, 0.5, r'Cumulative DAP (Gycm$^2$)', va='center', rotation='vertical')
        fig.show()
        

    def show_room(self):
        pass


'''






#test.plot(subplots = False,drawstyle="steps")

#

#R = Report(D)
#R.OGP_workload_plot()
#R.room_lead_table()
#R.source_workload_plots()


dft = pd.DataFrame()
grouped_df = df.groupby(['organ','view'])
dft['N'] = grouped_df.kV.count()
dft['mean_kV'] = grouped_df.kV.mean()
dft['cum_mAs'] = grouped_df.mAs.sum()
dft['mean_field_size'] = grouped_df.beam_area.mean()
dft['focus_distance'] = grouped_df.SID.mean()


#%%
    
#%%
#OLDPLOTS
#Plotting functions which should be deleted for shipping
dfogp = get_OGP_stats(df)
#dfogp.to_csv('ogp_stats.csv')

dfmode = get_stats_from_grouped_data(df,'det_mode',['DAP','mAs','kV','SID','Clinical EXI','Physical EXI'])

fig,ax = plt.subplots()


mAs_by_kV.plot(style='o',ax = ax)
ax.set_xlabel('kV')
ax.set_ylabel('Sum mAs')
fig.savefig('report/bin/kvspectrum.pdf')

#%%
plot_df = df.groupby(['det_mode','kV_bin'])['mAs'].sum().unstack('kV_bin').T

fig,ax = plt.subplots()
plot_df.plot(kind='bar',width = .9,ax = ax)
ax.set_ylabel('Cumulative mAs')
ax.legend(['Table','Wall','Free'])
fig.set_size_inches(12,6)
fig.savefig('report/bin/kV_spectrum_by_mode.pdf')



#%%
det_modes = ['X','W','T']
labels = ['Free','Wall','Table']


fig,axes = plt.subplots(1,len(det_modes),sharey=True)
fig.set_size_inches(12,6)
for i, det_mode in enumerate(det_modes):
    ax = axes[i]
    __, mAs_workload = create_xraybarr_spectrum(df,df.det_mode==det_mode)
    mAs_workload.plot(style = 'o',ax = ax)
    ax.set_xlabel('kV')
    ax.set_ylabel('Cumulative mAs')
    ax.set_title(labels[i])
fig.savefig('report/bin/kV_spectrum_by_mode_old.pdf')
    



#%%
exam_codes = {'X':'Free','W':'Wall bucky','T':'Table'}
exam_types = df.det_mode.unique()
decoded_types = [exam_codes[exam_type] for exam_type in exam_types]

#%%
#Show the total DAP as a function of exam type

fig,ax = plt.subplots()
ax.axes.set_xlabel('Exam type')
ax.axes.set_ylabel('Total DAP')
x = np.arange(len(exam_types))
ax.bar(x,dfmode['DAP_sum'])
ax.set_xticks(x)
ax.set_xticklabels(decoded_types)
fig.savefig('report/bin/type_distribution.pdf',fmt='pdf')

#%%
#kV boxplots
fig,ax = plt.subplots()
ax.axes.set_xlabel('Exam type')
ax.axes.set_ylabel('Mean kVp')
x = np.arange(len(dfmode))
ax.bar(x,dfmode['kV_mean'],yerr = dfmode['kV_std'])
ax.set_xticks(x)
ax.set_xticklabels(decoded_types)
fig.savefig('report/bin/kv_by_exam_type.pdf',fmt='pdf')
#%%
#kV boxplots
fig,ax = plt.subplots()
ax.axes.set_xlabel('Exam type')
ax.axes.set_ylabel('Mean kVp')
xdata = [df[df.det_mode=='X'].kV,df[df.det_mode=='W'].kV,df[df.det_mode=='T'].kV]
ax.boxplot(xdata)
#ax.set_xticks(x)
ax.set_xticklabels(decoded_types)
fig.savefig('report/bin/kv_by_exam_type_boxplot.pdf',fmt='pdf')

#%%
#Find the time when the highest DAP is delivered, make a nice plot to show it
h,cumdaps = get_usage_during_periods(df.timestamp,df.DAP)

fig,ax = plt.subplots()
ax.plot(h,cumdaps,'.')
ax.axes.set_xlabel('Start time for 8 hour timeblock (hr)')
ax.axes.set_ylabel('Total DAP') #unit? (mGy.cm^2)
fig.tight_layout()
fig.savefig('report/bin/Peak_work_times.pdf',fmt='pdf')

#%%
#Big plot showing all the organ protocol data

fig,axes = plt.subplots(nrows = 3,ncols = 1)
fig.set_size_inches(14,12)

mask = dfogp.DAP_sum > dfogp.DAP_sum.quantile(.6)

dfogp[mask].N_studies.plot(kind = 'bar', ax = axes[0],sharex = True)
axes[0].set_ylabel('Number of studies')

dfogp[mask].DAP_sum.plot(kind='bar',ax = axes[1])
axes[1].set_ylabel('Total DAP uGy/m^2')

dfogp[mask].kV_mean.plot(kind = 'bar', ax = axes[2],sharex = True,yerr=dfogp[mask].kV_std)
axes[2].set_ylabel('kVp')

axes[2].set_xlabel('Organ protocol')

fig.tight_layout()

fig.savefig('report/bin/ogp_high_usage.pdf')

#%%
#cumulative mAs delivered for each kV

df['kV_bin'] = df.kV // 10 * 10
testy = df.groupby(['kV_bin']).mAs.sum()
fig,ax = plt.subplots()
testy.plot(style='o',ax = ax)
ax.set_xlabel('kV')
ax.set_ylabel('Sum mAs')
plt.savefig('report/bin/kvspectrum.pdf')


'''