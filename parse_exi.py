# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:43:58 2018

@author: WilliamCh
"""
import pdb

import os
from string import Template

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from shapely.geometry import Point, box, LinearRing, LineString
from descartes import PolygonPatch
from PIL import Image


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
def match_columns_to_device_list(columns, device_list):
    """Find the entry in the device that matches the imported exi log"""
    for i, device in device_list.iterrows():
        if device.iloc[1:].tolist() == columns.tolist():
            break
    device_cols = device.iloc[1:].to_dict()
    device_cols = {v: k for k, v in device_cols.items()}
    return device_cols

def standardise_df_columns(df, device_list):
    """Convert the column names of a dataframe to the
    format used in Ysio exi logs, based on a mapping
    input file
    """
    cols = match_columns_to_device_list(df.columns, device_list)
    return df.rename(columns=cols)

def strip_df(df):
#    for column in df.columns:
#        df[column] = df[column].str.strip(' ,(!)')
    df = df.applymap(lambda x: x.strip(' ,(!)') if type(x) is str else x)
    return df

def enforce_column_types(df, device_list):
    """Data cleaning function that converts the columns
    to the format required in the device_list input file
    """
    
    column_types = device_list.loc['column_type']
    for col in df.columns:
        if column_types[col] in ['float64', 'int64', 'int']:
            try:
                df.loc[df[col]=='',col] = 0
            except:
                pass
        df[col] = df[col].astype(column_types[col], errors='ignore')
        mask = df[col] != df[col]
        df.loc[mask,col] = 0
    return df

def convert_numeric_columns(df,
                            numeric_cols=[
                                'Deviation index', 'Clinical EXI',
                                'Maximum EXI', 'Minimum EXI',
                                'Physical EXI', 'kV', 'mAs',
                                'SID', 'DAP', 'Dose']):

    df[numeric_cols] = df[numeric_cols].replace('[^\d.]+', '', regex=True)
    for col in numeric_cols:
        if df[col].dtype == 'O':
            df.loc[df[col] == '', [col]] = 0
            df[col] = df[col].str.strip()
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def remove_spurious_data(df):
    """Remove any instances of phantom repeat exposures"""
    df = df.drop_duplicates(['Acq. date', 'Acq. time', 'DAP'])
    #Remove any instances where kV is 0
    df = df[df.kV > 30]
    #Remove standard QA instances
    df = df[df.OGP.str[0] != 'Q']
    df = df[df.OGP.str[0] != '*Q']
    #If the OGP contains zero or only only one word, panic and drop the entry
    df = df.loc[df.OGP.str.split(' ').str.len() > 1, :]
    return df

def format_date(df):
    """Format a better timestamp. first add preceeding zeroes:"""
    df['Acq. time'] = df['Acq. time'].map('{:010}'.format)
    df['Acq. date'] = df['Acq. date'].map('{:08}'.format)
    df['timestamp'] = pd.to_datetime(df['Acq. date'] + df['Acq. time'],
                                     format='%Y%m%d%H%M%S%f')
    return df

def calculate_beam_area(df):
    """Calculate the area of the beam for each exi log entry"""
    width_height = df.Collimation.str.split('x')
    df['beam_area'] = (width_height.str[0].astype(float)
                       * width_height.str[1].astype(float)/10000)
    return df


def extract_organ_view(df, ogp_split, mask, n_organ_words):
    df.loc[mask, 'organ'] = (
        ogp_split.loc[mask].str[:n_organ_words].str.join(' '))
    df.loc[mask, 'view'] = (
        ogp_split.loc[mask].str[n_organ_words:].str.join(' '))
    return df

def analyse_organ_protocol(df):
    """Multi-stage unpacking of the organ protocol exi log entry
    Adds the following columns to the dataframe:
        det_mode -- ideally one of 'X', 'T' or 'W'
        organ -- Ideally the organ targetted by the protocol
        view -- Ideally an indication of which view was used
        age -- Some OGPs include age. Not all exposures have this entry
    """
    OGP = df.OGP.copy()
    #Remove *, they're doing nothing for us
    OGP = OGP.str.replace('*', '')

    #Split up the OGP into words
    OGP_split = OGP.str.split(' ')
    
    #First element is det mode, pop it from the list
    df['det_mode'] = OGP.str[0]
    OGP_split = OGP_split.str[1:]

    #Set up blank columns for age, organ and view
    df['age'] = OGP_split.copy()
    df.age = ''
    df['organ'] = df.age.copy()
    df['view'] = df.age.copy()

    #If the OGP contains '-', it is an age as the last word of the string
    #Extract this element
    view_age_test = OGP_split.str[-1].str.contains('-')
    df.loc[view_age_test, 'age'] = OGP_split.str[-1].loc[view_age_test]
    #If we extracted age, remove it from the OGP string
    OGP_split.loc[view_age_test] = OGP_split[view_age_test].str[:-1]

    #Analyse remaining words based on total number of words
    ogp_remaining_words = OGP_split.str.len()
    #Check if remaining words is 0. if it is, complain to radiographers
    #and also duplicate the age column. Naughty radiographer!
    mask = ogp_remaining_words == 0
    OGP_split[mask] = df.age[mask]

    #Find n_organ_words:n_view_words
    #If one remaining word, assume 1:0
    mask = ogp_remaining_words == 1
    df = extract_organ_view(df, OGP_split, mask, 1)

    #If two remaining words, assume 1:1
    mask = ogp_remaining_words == 2
    df = extract_organ_view(df, OGP_split, mask, 1)

    #If three remaining words, check last and first word.
    #If first is lumbar, assume 2:1
    mask = ogp_remaining_words == 3
    mask2 = OGP_split.str[0].str.contains('LUMBAR')
    df = extract_organ_view(df, OGP_split, mask & mask2, 2)

    #Otherwise assume 1:2
    df = extract_organ_view(df, OGP_split, mask & ~mask2, 1)

    #If there are four or more remaining words, naively assume 2 organ:rest view
    mask = ogp_remaining_words > 3
    df = extract_organ_view(df, OGP_split, mask & mask2, 2)

    return df

#Use an input file to separate free exposures into floor and wall directed
def bin_organ_protocols(df, input_ogp_binning_fn='input_ogp.csv'):
    df['det_code'] = df.det_mode
    ogpdf = pd.read_csv(input_ogp_binning_fn, index_col='OGP')
    mask = df['det_mode'] == 'X'
    subset = df[mask].copy()
    cols = ['mAs', 'DAP', 'Dose']

    for i in ogpdf:
        temp = subset.copy()
        temp.det_code = i
        if i == 'T':
            temp.SID = df[df.det_mode == 'T'].SID.mean()
        else:
            temp.SID = df[df.det_mode == 'W'].SID.mean()
        multiplier = temp.OGP.map(ogpdf[i].to_dict())
        ogpdf['organ'] = ogpdf.index.str.split(' ').str[1]
        ogpdf2 = ogpdf.drop_duplicates(subset='organ')
        ogpdf2.index = ogpdf2.organ
        ogpdf2.drop(columns='organ')
        multiplier2 = temp.organ.map(ogpdf2[i].to_dict())
        multmask = multiplier != multiplier
        multiplier[multmask] = multiplier2[multmask]

        #Now look at where the OGP map failed
        #Assume worst case distribution of T=1,C=1,2C=1,3C=0.1
        if i in ['T', 'C', '2C']:
            multiplier[multiplier != multiplier] = 1
        else:
            multiplier[multiplier != multiplier] = 0.1
        temp[cols] = temp[cols].multiply(multiplier, axis='index')
        df = df.append(temp)
    
    df.loc[df.SID!=df.SID,'SID'] = 130

    df = df.loc[df.mAs != 0].copy()
    df = df.loc[df.det_code != 'X'].copy()

    goodmask = df.det_code.isin(['W', 'T', 'C', '2C', '3C'])
#    print(df[~goodmask].count() + ' entries were excluded due to not being recognisable')
    df = df.loc[goodmask].copy()

    return df

#Hacky function to convert DAP to Gycm2. Consider finesse.
def convert_DAP_to_Gycm2(df):
#    DAPmean = df.DAP.mean()
    df.DAP = df.DAP/10
    return df

def import_data(exi_fn='sample_exi_log.csv', device_list_fn='device_list.csv'):
    device_list = pd.read_csv(device_list_fn, index_col='index')
    df = (pd.read_csv(exi_fn)
          .pipe(strip_df)
          .pipe(standardise_df_columns, device_list)
          .pipe(enforce_column_types,device_list)
          .pipe(calculate_beam_area)
          .pipe(remove_spurious_data)
          .pipe(format_date)
          .pipe(analyse_organ_protocol)
          .pipe(bin_organ_protocols)
          .pipe(convert_DAP_to_Gycm2)
         )
    return df

#%%
#Functions that compute potentially useful statistics and values
def get_usage_during_periods(timestamps, dap):
    start_time = []
    total_dap = []
    hours = timestamps.dt.hour
    for i in range(24):
        m_hours = (((hours > i) & (hours < i + 8))
                   | ((hours + 24 > i) & (hours +24 < i + 8)))
        cumdap = dap[m_hours].sum()
        if cumdap != cumdap:
            cumdap = 0
        total_dap.append(cumdap)
        start_time.append(i)
    return start_time, total_dap

def find_highest_usage_period(timestamps, dap):
    start_time, total_dap = get_usage_during_periods(timestamps, dap)
    argmax = np.array(total_dap).argmax()
    return start_time[argmax], total_dap[argmax]

def get_ratio_highest_usage_to_total(timestamps, dap):
    __, dapmax = find_highest_usage_period(timestamps, dap)
    return dapmax/sum(dap)

def get_dose_rescale_factor(df, breakdown=False):
    #rescale_factors
    exi_duration_rescale = (7*24*60*60
                            / (
                                df.timestamp.iloc[-1]-df.timestamp.iloc[0]
                            ).total_seconds()
                           )
    weekend_rescale = 5/7
    busy_period_rescale = get_ratio_highest_usage_to_total(df.timestamp, df.DAP)
    rescale_factor = (exi_duration_rescale
                      * weekend_rescale
                      * busy_period_rescale
                     )
    if breakdown:
        return {'exi_duration_rescale':exi_duration_rescale,
                'weekend_rescale':weekend_rescale,
                'busy_period_rescale':busy_period_rescale,
                'total':rescale_factor}
    return rescale_factor

def lead_to_weight(thickness, commercial_weight=0.44):
    return (thickness // commercial_weight + 1) * 5

#%%
#OGP stats analysis functions
def get_stats_from_grouped_data(df, group_col, value_cols):
    dfout = pd.DataFrame(columns=['N_studies'])
    grouped_data = df.groupby(group_col)
    for v in value_cols:
        dfout[v +'_sum'] = grouped_data[v].sum()
        dfout[v +'_mean'] = grouped_data[v].mean()
        dfout[v +'_std'] = grouped_data[v].std()
        dfout[v +'_med'] = grouped_data[v].median()
    dfout['N_studies'] = grouped_data[v].count()
    return dfout

def get_OGP_stats(df):
    value_cols = ['DAP', 'mAs', 'kV', 'SID', 'Clinical EXI', 'Physical EXI']
    pivot_col = 'OGP'
    return get_stats_from_grouped_data(df, pivot_col, value_cols)

#%%
#Xraybarr integration functions
def create_xraybarr_spectrum(df, mask=None):
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
    mAmin_by_kV = view.groupby(['kV_bin']).mAs.sum() / 60

    for kv in np.arange(25, 151, 5):
        try:
            mAmin = mAmin_by_kV.loc[kv]
        except:
            mAmin = 0
        output_list.append('%s (mAmin @ %s kVp)' % (mAmin, kv))
    #Properly calculate area here...
    weighted_mean_area = (view.mAs*view.beam_area).sum()/view.mAs.sum()
    SID = view.SID.mean()/100

    output_list.append('%d  area of primary beam (cm2) at'
                       % (weighted_mean_area))
    output_list.append('%.3f this primary distance (m)'
                       % (SID))
    output_list.append('150  leakage technique kVp')
    output_list.append('3.3  leakage technique mA')
    output_list.append('100  leakage exposure rate (mR/hr) at 1 m when'
                       + ' operated at leak technique')

    return output_list, mAmin_by_kV


def save_xraybarr_spectrum(df,
                           mask,
                           output_folder='output/',
                           spectrum_name='default_spectrum'):

    output_list, mAs_by_kV = create_xraybarr_spectrum(df, mask)
    with open(output_folder+spectrum_name+'.spe', 'w') as file:
        file.writelines('\n'.join(output_list))

def make_xraybarr_spectrum_set(exi_fn,
                               room_name='default',
                               output_folder='output/',
                               D=None
                              ):
    if D:
        df = D.df
    else:
        df = import_data(exi_fn=exi_fn)
        
    ratio = get_dose_rescale_factor(df)
    #mAs needs to be in mAmin/week, not year
    df.mAs = df.mAs * ratio

    categories = df.det_code.unique()
    masks = [df.det_code == category for category in categories]
    mask_dict = dict(zip(categories, masks))

    Tmask = df.det_code == 'T'
    Tthresh = df[Tmask].beam_area.mean()
    Tlarge = Tmask & (df.beam_area > Tthresh)
    Tsmall = Tmask & (df.beam_area < Tthresh)
    mask_dict['Tlarge'] = Tlarge
    mask_dict['Tsmall'] = Tsmall
    mask_dict.pop('T')

    for cat, mask in mask_dict.items():
        save_xraybarr_spectrum(df,
                               mask,
                               output_folder=output_folder,
                               spectrum_name='_'.join([room_name, cat])
                              )
    if D:
        make_xraybarr_barrier_setup(D, output_folder, room_name, mask_dict)

def make_xraybarr_barrier_setup(D, output_folder, room_name, mask_dict):
    #Preload the templates from text files
    with open('input/bar_header.txt', 'r') as f:
        header_template = Template(f.read())
    with open('input/bar_tube.txt', 'r') as f:
        tube_template = Template(f.read())
    
    tubes = ['T', 'T', 'W', 'C', '2C']
    names = ['Tlarge', 'Tsmall','W','C','2C']
    rooms = D.dmap['T'].keys()
#    mapping_data = {}
#    dict_of_df = {k: pd.DataFrame(v) for k,v in D.dmap.items()}
#    df = pd.concat(dict_of_df, axis=1).T
    for room in rooms:
        #MAKE THE HEADER
        head_map = {'institution':'AUTO HOSPITAL',
                    'xrayroom':'AUTO XRAY ROOM',
                    'receiver':room,
                    'occupancy':D.dfr.loc[room,'Occupancy'],
                    'constraint':D.dfr.loc[room,'Constraint']*50/1000}
        output_string = header_template.substitute(head_map)
        
        #Make all the tubes
        for i in range(len(tubes)):
            tube_map = D.dmap[tubes[i]][room]
            tube_map['in_p_beam'] = int(tube_map['in_p_beam'])
            tube_map['name'] = names[i]
            tube_map['fn'] = '_'.join([room_name, names[i]])+'.spe'
            tube_map['longfn'] = output_folder + tube_map['fn']
            tube_map['SID'] = D.df.loc[mask_dict[names[i]],'SID'].mean()/100
            view = D.df.loc[mask_dict[names[i]],:]
            tube_map['area'] = (view.mAs*view.beam_area).sum()/view.mAs.sum()
            tube_map['patients'] = 1
            tube_map['mAmin'] = 1
            tube_map['leak_kVp'] = 150
            tube_map['leak_mA'] = 3.3
            
            output_string = output_string + tube_template.safe_substitute(tube_map)
            
        with open(output_folder + room + '.bar', 'w') as f:
            f.write(output_string)

#%%
#Geometric functions
def make_distancemap(dfr, dfs):
    distancemap = {}
    for i, Srow in dfs.iterrows():
        distancemap[Srow.det_mode] = {}
        for j, Rrow in dfr.iterrows():
            output_dic = {}
            tube_loc = Point(Srow.tubex, Srow.tubey)
            target_loc = Point(Srow.targetx, Srow.targety)
            room_rect = rect((Rrow.x1, Rrow.y1), (Rrow.x2, Rrow.y2))
            output_dic['primary_distance'] = (
                room_rect.exterior.distance(tube_loc) + 0.3)
            output_dic['secondary_distance'] = (
                room_rect.exterior.distance(target_loc) + 0.3)
    #        output_dic['tertiary_distance'] = room_rect.exterior.distance(target_loc)

            tube_target_line = LineString([tube_loc, target_loc])
            rect_ext = LinearRing(room_rect.exterior.coords)
            if Srow.floor_target:
                output_dic['angle'] = 90
            else:
                room_closest_point_projection = rect_ext.project(
                    target_loc)
                room_closest_point = rect_ext.interpolate(
                    room_closest_point_projection)
                target_room_line = LineString(
                    [target_loc, room_closest_point])
                output_dic['angle'] = get_angle_from_lines(
                    tube_target_line, target_room_line)

            if Srow.floor_target:
                output_dic['in_p_beam'] = False
            else:
                output_dic['in_p_beam'] = is_room_in_beam(
                    tube_target_line, rect_ext)

            distancemap[Srow.det_mode][Rrow.Zone] = output_dic
    return distancemap

def rect(p1, p2):
    xmin = min(p1[0], p2[0])
    xmax = max(p1[0], p2[0])
    ymin = min(p1[1], p2[1])
    ymax = max(p1[1], p2[1])
    return box(xmin, ymin, xmax, ymax)

def make_rect_column(row):
    return rect((row['x1'], row['y1']), (row['x2'], row['y2']))

def get_angle_from_lines(l1, l2):
    l1 = np.array(l1)
    v1 = l1[0, :] - l1[1, :]
    l2 = np.array(l2)
    v2 = l2[0, :] - l2[1, :]
    return (
        np.arccos(v1.dot(v2) / (sum(v1**2)**.5 * sum(v2**2)**.5)) * 180/np.pi)

def is_room_in_beam(beam_line, room_ext):
    l = np.array(beam_line)
    p1 = l[0, :]
    p2 = l[1, :]
    v = p2 - p1
    p2 = p1 + v*1000
    beam_line_extended = LineString([p1, p2])
    return beam_line_extended.intersects(room_ext)

def make_point_columns(row, xn, yn):
    return Point(row[xn], row[yn])
#%%
#Use one row of df,dfr,dfs,organmap,distancemap to calculate dose
#Implements the BIR method for dose calculations
class Dose:
    def __init__(self,
                 exi_fn='sample_exi_log.csv',
                 dfr_fn='input_rooms.csv',
                 dfs_fn='input_sources.csv',
                 att_coeff_fn='input_shielding_coefficients.csv'):

        self.df = import_data(exi_fn=exi_fn)
        self.import_dfr(dfr_fn)

        self.dfs = pd.read_csv(dfs_fn)
        self.dfs = self.dfs.where(self.dfs == self.dfs, '')

        self.df_att_coeff = pd.read_csv(att_coeff_fn)

        self.dmap = make_distancemap(self.dfr, self.dfs)
        self.make_geo_shapes()

    def import_dfr(self, dfr_fn):
        self.dfr = pd.read_csv(dfr_fn)
        self.dfr['added_attenuation'] = 0
        self.dfr.index = self.dfr.Zone
        #If coordinates are in imagej format:
        if 'BX' in self.dfr.columns:
            self.dfr['x1'] = self.dfr.BX
            self.dfr['y1'] = self.dfr.BY - self.dfr.Height
            self.dfr['x2'] = self.dfr.BX + self.dfr.Width
            self.dfr['y2'] = self.dfr.BY

    def export_distancemap(self, output_folder=False):
        dmap = {(outerKey, innerKey): values
                for outerKey, innerDict in self.dmap.items()
                for innerKey, values in innerDict.items()}
        dmap = pd.DataFrame(dmap).T.swaplevel(0, 1, 0).sort_index(0)
        if output_folder:
            dmap.to_csv(output_folder + 'distancemap.csv')
        return dmap
    
    def make_geo_shapes(self):
        self.dfr['rect'] = self.dfr.apply(make_rect_column, axis=1)
        self.dfs['tubeP'] = self.dfs.apply(make_point_columns,
                args=['tubex', 'tubey'], axis=1)
        self.dfs['targetP'] = self.dfs.apply(make_point_columns,
                args=['targetx', 'targety'], axis=1)

    #Transmission calculation functions
    def get_transmission(self, thickness, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return ((1 + b/a) * np.exp(a*y*thickness) - b/a)**(-1/y)

    def get_necessary_shielding(self, transmission, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        x = 1 / (a*y) * np.log((transmission**(-y) + (b/a)) / (1+(b/a)))
        return x

    def get_shielding_coefficients(self, material, kV):
        view = self.df_att_coeff.loc[self.df_att_coeff.Material == material,
                                     ['kV', 'a', 'b', 'y']]
        #view.index = view.kV
        xp = np.array(view['kV'])
        yp = np.array(view[['a', 'b', 'y']])
        a = np.interp(kV, xp, yp[:, 0])
        b = np.interp(kV, xp, yp[:, 1])
        y = np.interp(kV, xp, yp[:, 2])
        return a, b, y

    #Dose calculation functions
    #Take DAP in Gycm^2, area in cm^2, distances in m returns dose in uGy
    def get_primary_dose(self,
                         DAP,
                         beam_area,
                         d_tube_target,
                         d_tube_room,
                         kV,
                         in_beam=False,
                         wall_leq=0,
                         bucky_leq=1):
        transmission = self.get_transmission(wall_leq + bucky_leq, 'Lead', kV)
        if in_beam:
            return (DAP/beam_area * (d_tube_target/d_tube_room)**2
                    * 1e6 * transmission)
        return 0

    def get_scattered_dose(self,
                           DAP,
                           kV,
                           distance,
                           angle,
                           wall_leq=0):
        #In: DAP as Gycm^2
        #Out: Dose as uGy
        transmission = self.get_transmission(wall_leq, 'Lead', kV)
        a = -1.042e-7
        b = 3.265e-5
        c = -2.751e-3
        d = 8.371e-2
        e = 1.578
        f = 5.987e-3
        S = ((a*angle**4 + b*angle**3 + c*angle**2 + d*angle + e)
             * ((kV-85)*f + 1)
            )
#        S= 0.031 * kV + 2.5
        return S * DAP / distance**2 * transmission

    def get_tertiary_dose(self, DAP, distance, ceiling_height,
                          barrier_height, transmission):
        self.df
        return 0

    def get_leakage_dose(self, kV, mAs, distance, transmission):
        self.df
        return 0

    def calculate_dose_for_df_row(self, dfg_row, ignore_attenuation=True):
        dfr = self.dfr
        all_rooms_dose = {}
#        ceiling_height = None
#        barrier_height = None
        kV = dfg_row.kV.mean()
        DAP = dfg_row.DAP.sum()
#        mAs = dfg_row.mAs.sum()
        beam_area = dfg_row.beam_area.mean()
        SID = dfg_row.SID.mean()/100
        for __, receiver_row in dfr.iterrows():
            #Get data for set room
            room_data = (
                self.dmap[dfg_row.det_code.values[0]][receiver_row['Zone']])
            primary_distance = room_data['primary_distance']
            secondary_distance = room_data['secondary_distance']
            angle = room_data['angle']
            in_p_beam = room_data['in_p_beam']
            if ignore_attenuation:
                lead_eq = 0
            else:
                lead_eq = (receiver_row.gypsum_thickness/320
                           + receiver_row.added_attenuation
                          )
                lead_eq = max(lead_eq, 0)

            #Use room data to calculate primary and scatter
            room_dose = {}
            room_dose['primary'] = self.get_primary_dose(DAP, beam_area,
                     SID, primary_distance, kV, in_p_beam, lead_eq)
            room_dose['scatter'] = self.get_scattered_dose(DAP, kV,
                     secondary_distance, angle, lead_eq)
#            room_dose['tertiary'] = self.get_tertiary_dose(DAP,
#                     secondary_distance,
#                     ceiling_height,barrier_height,lead_eq)
#            room_dose['leakage'] = self.get_leakage_dose(kV,mAs,
#                     primary_distance,lead_eq)
            all_rooms_dose[receiver_row.Zone] = room_dose
        return pd.DataFrame(all_rooms_dose)

    #Returns dose in uGy per week, provided DAP is given in Gcm^2
    def calculate_dose(self, ignore_attenuation=True):
        df = self.df
        df = df.groupby(['det_code', 'kV'])
        output = df.apply(self.calculate_dose_for_df_row, ignore_attenuation)
        output = output * get_dose_rescale_factor(self.df)
        output.index = output.index.rename('contribution', level=2)
        return output

    def save_verbose_data(self,
                         output_folder='output',
                         output_name='test',
                         ignore_attenuation=True):
        doses = self.calculate_dose()

        dfg = self.df.groupby(['det_code', 'kV'])
        DAP, mAs = dfg.DAP.sum(), dfg.mAs.sum()
        workload = pd.DataFrame([DAP, mAs]).T
        
        dose_bytube = doses.sum(level=(0, 2))

        dose_short = doses.sum(level=2)
        dose_short = dose_short.T
        dose_short['total'] = dose_short.sum(axis=1)
        dose_short = dose_short.T

        factors = pd.Series(get_dose_rescale_factor(self.df, True))

        dose_bytube.to_csv('%s/%s_doses_by_tube.csv'
                          % (output_folder, output_name))
        dose_short.to_csv('%s/%s_doses_summary.csv'
                          % (output_folder, output_name))
        workload.to_csv('%s/%s_workload.csv'
                        % (output_folder, output_name))
        doses.to_csv('%s/%s_doses.csv' % (output_folder, output_name))
        factors.to_csv('%s/%s_factors.csv' % (output_folder, output_name))
        self.export_distancemap().to_csv('%s/%s_distances.csv'
                                         % (output_folder, output_name))
        return workload, doses, dose_short, factors

    def get_lead_req(self,
                     iterations=3):
        #Start from 0 attenuation
        self.dfr.added_attenuation = 0
        self.dfr['Limit'] = self.dfr.Constraint / self.dfr.Occupancy

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
                self.dfr['raw_dose'] = doses.sum()
                self.dfr['required_transmission'] = (
                    self.dfr.Limit / self.dfr.raw_dose)


            #Converts dose for entire Exi log to dose per week in uGy
            #todo: double check this conversion.
            doses = (doses.sum()).T
            doses = doses
            dose_out.append(doses)
            #Compare to constraints
            attenuation_required = self.dfr.Limit / doses
            #Compute an equivalent lead requirement
            lead = self.get_necessary_shielding(attenuation_required,
                                                'Lead', 90)
            lead_out.append(lead)
            self.dfr.added_attenuation = self.dfr.added_attenuation + lead
            self.dfr['wall_weight'] = lead_to_weight(self.dfr.added_attenuation)

        return (self.dfr.added_attenuation,
                dose_out, lead_out,
                self.dfr.wall_weight
               )

#%%
'''
Reporting and graphing functions
'''
#Plotting functions
def add_point_to_ax(ax, point, text, textoffset=-.3):
    try:
        point = Point(point)
    except:
        pass
    patch = PolygonPatch(point.buffer(.15))
    ax.add_patch(patch)
    text_position = point.x+textoffset, point.y+textoffset
    ax.text(*text_position, text,
            verticalalignment='center',
            horizontalalignment='center')

def add_rect_to_ax(ax, rect, text, color):
    patch = PolygonPatch(rect, color=color)
    ax.add_patch(patch)
    ax.plot(*rect.exterior.xy, color='black')
    a = np.array(rect.bounds).reshape(2, 2)
    text_position = (a[0, :] + a[1, :]) / 2
    ax.text(*text_position, text,
            verticalalignment='center',
            horizontalalignment='center')

def add_arrow_to_ax(ax, P1, P2, text):
    p1 = np.array(P1.coords[0])
    p2 = np.array(P2.coords[0])

    length = p2-p1
    if ~(p1-p2).any():
        add_point_to_ax(ax, p1, text)
    else:
        ax.arrow(*p1, *length, width=.1, length_includes_head=True)
        ax.text(*(p1+.25), text)
def color_from_wall_weight(wall_weight):
    return cm.viridis.colors[int(wall_weight)*9+30]

#Reporting and graphing
class Report:
    def __init__(self, D, output_folder=False):
        self.D = D
        self.output_folder = output_folder
        try:
            os.mkdir(self.output_folder)
        except:
            pass
        if output_folder:
            self.OGP_workload_plot()
            self.room_lead_table()
            self.source_workload_plots()
            self.show_room()

    def OGP_workload_plot(self):
        #Big plot showing all the organ protocol data
        dfogp = get_OGP_stats(self.D.df)
        fig, axes = plt.subplots(nrows=3, ncols=1)
        fig.set_size_inches(14, 12)

        mask = dfogp.DAP_sum > dfogp.DAP_sum.quantile(.5)

        dfogp[mask].N_studies.plot(kind='bar', ax=axes[0], sharex=True)
        axes[0].set_ylabel('Number of studies')
        dfogp[mask].DAP_sum.plot(kind='bar', ax=axes[1])
        axes[1].set_ylabel('Total DAP uGy/m^2')
        dfogp[mask].kV_mean.plot(kind='bar',
                                 ax=axes[2], sharex=True,
                                 yerr=dfogp[mask].kV_std)
        axes[2].set_ylabel('kVp')
        axes[2].set_xlabel('Organ protocol')

        fig.tight_layout()
        if self.output_folder:
            fig.savefig(self.output_folder + 'ogp_stats.pdf')
            plt.close(fig)
        else:
            fig.show()

    def room_lead_table(self):
        if ~self.D.dfr.added_attenuation.any():
            self.D.get_lead_req(iterations=3)
        '''
        Create a table for reporting, based on dose calculation results
        '''

        headers = {
            'raw_dose':{'head':'Unattenuated dose (uGy/wk)', 'format':0},
            'Limit':{'head':'Dose constraint (uGy/wk)', 'format':0},
            'required_transmission':{'head':'Barrier transmission', 'format':3},
            'added_attenuation':{'head':'Min. lead eq. (mm)', 'format':2},
            'wall_weight':{'head':'Barrier lead weight (kg/m^2)', 'format':0}
        }

        key_coltitle = {k:headers[k]['head'] for k in headers.keys()}
        key_fmt = {k:headers[k]['format'] for k in headers.keys()}
        coltitle_fmt = {key_coltitle[k]:key_fmt[k] for k in headers.keys()}

        table = self.D.dfr.loc[:, headers.keys()].copy()
        table = table.round(key_fmt)

        self.table = table.rename(columns=key_coltitle)
        if self.output_folder:
            self.table.to_excel(self.output_folder + 'results_table.xlsx')
            with open(self.output_folder + 'results_table_latex.txt','w') as f:
                 f.write(self.table.to_latex())

    def source_workload_plots(self):
        dfp = pd.pivot_table(self.D.df.reset_index(),
                             index='kV',
                             columns='det_code',
                             values='DAP',
                             aggfunc=np.sum,
                            )
        kvs = np.arange(dfp.index.unique().min(), dfp.index.unique().max()+1)
        dfp = dfp.loc[kvs]
        dfp[dfp != dfp] = 0
        namemap = self.D.dfs[['det_mode', 'exam_type']]
        namemap.index = namemap.det_mode
        namemap = namemap.exam_type.to_dict()
        dfp = dfp.rename(columns=namemap)
        dfp['total'] = dfp.sum(axis=1)
        fig, axes = plt.subplots(nrows=3,
                                 ncols=2,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(7, 9),
                                )
        axes = axes.reshape(6,)
        
        
        dfp.plot(ax=axes, subplots=True, drawstyle="steps")
        
        #fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        fig.text(0.001, 0.5, r'Cumulative DAP (Gycm$^2$)',
                 va='center',
                 rotation='vertical',
                )
        if self.output_folder:
            fig.savefig(self.output_folder + 'workload_plots.pdf')
            plt.close(fig)
        else:
            fig.show()

    def show_room(self, im_fn=None):
        """Create a figure based on the imported source and room
        geometric data. Save it or show it depending how the function
        is called.
        """
        xrange = (min(self.D.dfr.x1.min(), self.D.dfr.x2.min()),
                  max(self.D.dfr.x1.max(), self.D.dfr.x2.max()))
        yrange = (min(self.D.dfr.y1.min(), self.D.dfr.y2.min()),
                  max(self.D.dfr.y1.max(), self.D.dfr.y2.max()))

        fig, ax = plt.subplots(figsize=(9, 9))

        for i, row in self.D.dfr.iterrows():
            text = row.Zone
            text= text + '\nConstraint: ' + str(row.Constraint) + ' uSv/week'
            text = text + '\nO: ' + str(row.Occupancy)
            if self.D.dfr.added_attenuation.any():
                text = text + '\n '+str(row.wall_weight) + ' kg/m^2'
                color = color_from_wall_weight(row.wall_weight)
            else:
                color = color_from_wall_weight(0)

            add_rect_to_ax(ax, row.rect, text, color)

        for i, row in self.D.dfs.iterrows():
            add_arrow_to_ax(ax, row.tubeP, row.targetP, row.det_mode)

        if self.D.dfr.added_attenuation.any():
            legend_weights = np.arange(0, 25.1, 5)
            legend_colors = [color_from_wall_weight(w) for w in legend_weights]
            legend_lines = [Line2D([0], [0], color=c, lw=4) for c in legend_colors]
            legend_labels = [str(w) + ' kg/m^2' for w in legend_weights]
            ax.legend(legend_lines, legend_labels, loc=6,
                      bbox_to_anchor=(1, 0.5))
            
        ax.set_title('Floor plan')
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_aspect(1)
        
        if im_fn:
            self.image_room_overlay(ax,im_fn)
            

        
        if self.output_folder:
            if not im_fn:
                save_fn = 'room_layout.pdf'
            else:
                save_fn = 'room_overlay.pdf'
            fig.savefig(self.output_folder + save_fn, bbox_inches='tight')
            plt.close(fig)
        else:
            fig.show()

    def image_room_overlay(self, ax, im_fn):
        im = Image.open(im_fn)
        try:
            res = np.array(im.info['resolution'])
        except:
            return
        scale = (np.array(im.size)/res).astype('float')
        ax.set_xlim(0, scale[0])
        ax.set_ylim(0, scale[1])
        ax.set_aspect(1)
        ax.imshow(im, extent=[0,scale[0],0,scale[1]])

#%%
def full_report(exi_fn, dfs_fn, dfr_fn, folder, room_name, imname = None):
    for fn in [folder, folder+room_name,
               folder+room_name+'/XRAYBARRspectrums',
               folder+room_name+'/verbose']:
        try:
            os.mkdir(fn)
        except:
            pass
    output_folder = folder+room_name+'/'
    
    D = Dose(exi_fn, dfs_fn=dfs_fn, dfr_fn=dfr_fn)
    D.get_lead_req(iterations=3)
    D.save_verbose_data(output_folder+'verbose', room_name)
    D.export_distancemap(output_folder+'verbose/')
    R = Report(D, output_folder)
    try:
        R.show_room(imname)        
    except Exception as e:
        print(e)
    make_xraybarr_spectrum_set(exi_fn, room_name, output_folder + 'XRAYBARRspectrums/', D)
    
if __name__ == '__main__':
#    full_report('sample_exi_log.csv','input_sources.csv','input_rooms.csv','output/','testroom1')
    D = Dose()
#    D.get_lead_req(iterations=1)
#    D.save_verbose_data()
#    R = Report(D)
#    R.OGP_workload_plot()
#    R.room_lead_table()
#    R.source_workload_plots()
#    R.show_room()
