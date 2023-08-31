import streamlit as st
import pandas as pd
import requests

#fcols = feiner columns that he uses for processing
fcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb','UPI']

# allcols = the rest of the columns we actually want + feiner cols
allcols = fcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']

def feinerize(datafr):
    #separate timestamp into two columns
    datafr['Time'] = pd.to_datetime(datafr['Time'])
    datafr['Date Calc'] = datafr['Time'].dt.date
    datafr['Time Calc'] =  datafr['Time'].dt.time

    #separate patient ID into two columns
    datafr[['Subject', 'Sample']] = datafr['Patient ID'].astype(str).str.split(pat='.', expand=True)

    # rename columns 
    datafr = datafr.rename(columns={"Time": 'Time Stamp',
    'pCO2 (mmHg)': 'pCO2',
    'pO2 (mmHg)':'pO2',
    'sO2 (%)': 'sO2',
    'COHb (%)':'COHb',
    'MetHb (%)':'MetHb',
    'tHb (g/dL)':'tHb',
    'Accession number':'UPI',
    'K+ (mmol/L)':'K+',
    'Na+ (mmol/L)':'Na+',
    'Ca++ (mmol/L)':'Ca++',
    'Cl- (mmol/L)':'Cl-',
    'Glu (mmol/L)': 'Glucose',
    'Lac (mmol/L)':'Lactate',
    'p50(act) (mmHg)':'p50',
    'cBase(Ecf) (mmol/L)':'cBase'})

    return datafr[allcols]


#start layout

st.header('Import ABL files into RedCap')

st.write('Instructions:....')

csv1 = st.file_uploader('ABL csv1', type='csv')
csv2 = st.file_uploader('ABL csv2', type='csv')

if csv1 is not None:
    df1 = pd.read_csv(csv1, encoding='ANSI')
    df1 = feinerize(df1)

edited_df = st.data_editor(df1, num_rows='dynamic')