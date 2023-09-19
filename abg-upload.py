import streamlit as st
import pandas as pd
import numpy as np
import requests

#fcols = feiner columns that he uses for processing
fcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'UPI', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']

#rcols = redcap columns including Session when it is created
rcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'UPI', 'Session', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']

# allcols = the rest of the columns we actually want + feiner cols
allcols = fcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']

# allcols_r = all columns we want + redcap cols
allcols_r = rcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']

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

#############################################
st.subheader('Step 1: upload files')

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

if uploaded_files != []:
    st.session_state['uploaded'] = True
else:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state['uploaded'] = False
    st.write('Please upload a file')

if st.session_state['uploaded']==True:
    if st.button('Combine CSVs'):
        dfs = []
        for num, file in enumerate(uploaded_files):
            df = pd.read_csv(file, encoding = 'cp1252')
            dfs.append(df)
        df1 = pd.concat(dfs, ignore_index=True)
        df1 = feinerize(df1)
        st.session_state['combined'] = True
        st.session_state['df1'] = df1

##############################################
if 'combined' in st.session_state:
    st.subheader('Step 2: Correct errors')
    edited_df = st.data_editor(st.session_state['df1'], num_rows='dynamic', key='data_editor')
    #count the number of null values in Subject, sample, patient id, and UPI columns
    df1= edited_df
    st.session_state['errors'] = False
    if edited_df['Subject'].isnull().sum() >0:
        st.write('Subject column has null values')
        st.session_state['errors'] = True
    if edited_df['Sample'].isnull().sum() >0:
        st.write('Sample column has null values')
        st.session_state['errors'] = True
    if edited_df['Patient ID'].isnull().sum() >0:
        st.write('Patient ID column has null values')
        st.session_state['errors'] = True
    if edited_df['UPI'].isnull().sum() >0:
        st.write('UPI column has null values')
        st.session_state['errors'] = True
##############################################
if 'errors' not in st.session_state:
    st.write('')
elif st.session_state['errors'] == False:
    st.subheader('Step 3: Add session numbers')
    # get list of all UPIs and put into a new df
    upis = st.session_state['df1']['UPI'].value_counts().index
    upi_df = pd.DataFrame(upis, columns=['UPI'])
    upi_df['Session'] = np.nan
    upi_edits = st.data_editor(upi_df, num_rows='dynamic', key='upi_editor')
    if st.button('Add Session Numbers to file'):
        #check if any of the values in the Session column are null
        if upi_edits['Session'].isnull().sum() >0:
            st.write('please fill in all session values')
        else:
            st.session_state['upi_df'] = upi_edits.reset_index()
            st.session_state['finaldf'] = edited_df.merge(upi_edits, on='UPI',how='left')
            st.session_state['finaldf'] = st.session_state['finaldf'][allcols_r]
            st.write(st.session_state['finaldf'])

if 'finaldf' in st.session_state:
    st.subheader('Step 3: Upload & Download')
    one, two = st.columns(2)
    with one:
        st.button('Upload to RedCap')
    with two:
        st.button('Download CSV')