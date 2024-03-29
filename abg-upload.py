import streamlit as st
import pandas as pd
import numpy as np
import requests
from redcap import Project

def load_project(key):
    api_key = st.secrets[key]
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    return project

project = load_project('REDCAP_ABG')


#fcols = feiner columns that he uses for processing
fcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'UPI', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']

#rcols = redcap columns including Session when it is created
rcols = ['Subject', 'Time Stamp', 'Date Calc', 'Time Calc', 'Sample', 'Patient ID', 'UPI', 'Session', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']

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
    'cBase(Ecf) (mmol/L)':'cBase'},)

    return datafr[allcols]


#start layout

st.header('Import ABL files into RedCap')

st.write('This app will allow you to upload ABL files and import them into RedCap. Please follow the steps below.')
st.markdown('''
    1. **Upload files**: Drag and drop the raw CSVs from the abls into the spot below.
    2. **Correct errors**: The app will alert you if there are missing values in the Subject, Sample, Patient ID, or UPI columns. You can correct these errors in the table below.
    3. **Add session numbers**: The app will alert you if there are missing session numbers. Every UPI should have a session number. You can add session numbers in the table below.
    4. **Upload to RedCap**: Once you have corrected all errors, you can upload the file to RedCap.
         ''')

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
            df = pd.read_csv(file, encoding = 'cp1252', converters={'Patient ID': str})
            dfs.append(df)
        df1 = pd.concat(dfs, ignore_index=True)
        df1 = feinerize(df1)
        st.session_state['combined'] = True
        st.session_state['df1'] = df1

##############################################
if 'combined' in st.session_state:
    st.subheader('Step 2: Correct errors')
    edited_df = st.data_editor(st.session_state['df1'].sort_values(by='Time Stamp'), num_rows='dynamic', key='data_editor')
    #count the number of null values in Subject, sample, patient id, and UPI columns
    df1= edited_df
    st.session_state['errors'] = False
    if edited_df['Subject'].isnull().sum() >0:
        st.write('Subject column has null values: ', edited_df[edited_df['Subject'].isnull()]['Time Stamp'].tolist())
        st.session_state['errors'] = True
    if edited_df['Sample'].isnull().sum() >0:
        st.write('Sample column has null values: ', edited_df[edited_df['Sample'].isnull()]['Time Stamp'].tolist())
        st.session_state['errors'] = True
    if edited_df['Patient ID'].isnull().sum() >0:
        st.write('Patient ID column has null values: ', edited_df[edited_df['Patient ID'].isnull()]['Time Stamp'].tolist())
        st.session_state['errors'] = True
    if edited_df['UPI'].isnull().sum() >0:
        st.write('UPI column has null values: ', edited_df[edited_df['UPI'].isnull()]['Time Stamp'].tolist())
        st.session_state['errors'] = True
    if sum(edited_df['Patient ID'] == "0000") >0:
        st.write('The row with Patient ID 0000 will be dropped') # not sure if we even need any message here...
        edited_df = edited_df[edited_df["Patient ID"] != "0000"]
##############################################
if 'errors' not in st.session_state:
    st.write('')
elif st.session_state['errors'] == False:
    st.subheader('Step 3: Add session numbers')
    upi_df = st.session_state['df1'].groupby(by=['Date Calc','UPI']).first().reset_index()
    
    #get list of UPI and Dates
    upi_df = upi_df[['Date Calc','UPI']]
    upi_df['Session'] = np.nan
    upi_edits = st.data_editor(upi_df, num_rows='dynamic', key='upi_editor')

    if st.button('Add Session Numbers to file'):
        #check if any of the values in the Session column are null
        if upi_edits['Session'].isnull().sum() >0:
            st.write('please fill in all session values')
            #write the time stamps of the rows with null values
            st.write(upi_edits[upi_edits['Session'].isnull()]['Time Stamp'].tolist())
        else:
            #merge session numbers into the original df 
            st.session_state['upi_df'] = upi_edits.reset_index()
            st.session_state['finaldf'] = edited_df.merge(upi_edits, on=['Date Calc','UPI'],how='left')
            st.session_state['finaldf'] = st.session_state['finaldf'][allcols_r]
            st.session_state['finaldf'].rename_axis('record_id', inplace=True)
                        
            #drop subject column. I know this doesn't make a lot of sense given that we added during feinerize
            #but this column doesn't exist in redcap, so I want to remove it. 
            # initally i wanted to have a function to be able to download a feiner dataframe, but I think we'll just create a separate script for that
            st.session_state['finaldf'] = st.session_state['finaldf'].drop(columns=['Subject'])
            
            #rename columns to match case in redcap
            st.session_state['finaldf'] = st.session_state['finaldf'].rename(columns={"record_id": 'record_id', 
                                                                                      'Subject': 'subject',
                                                                                      'Time Stamp': 'time_stamp',
                                                                                      "Date Calc": 'date_calc',
                                                                                      "Time Calc": 'time_calc',
                                                                                        'Sample': 'sample',
                                                                                        'Patient ID': 'subject',
                                                                                        'UPI': 'patient_id',
                                                                                        'Session': 'session',
                                                                                        'pH':'ph',
                                                                                        'pCO2':'pco2',
                                                                                        'pO2':'po2',
                                                                                        'sO2':'so2',
                                                                                        'COHb':'cohb',
                                                                                        'MetHb':'methb',
                                                                                        'tHb':'thb',
                                                                                        'K+':'k',
                                                                                        'Na+':'na',
                                                                                        'Ca++':'ca',
                                                                                        'Cl-':'cl',
                                                                                        'Glucose':'glucose',
                                                                                        'Lactate':'lactate',
                                                                                        'p50':'p50',
                                                                                        'cBase':'cbase'})
            #re order columns
            st.session_state['finaldf'] = st.session_state['finaldf'][['subject', 'time_stamp', 'date_calc', 'time_calc', 'sample', 'patient_id', 'session', 'ph', 'pco2', 'po2', 'so2', 'cohb', 'methb', 'thb', 'k', 'na', 'ca', 'cl', 'glucose', 'lactate', 'p50', 'cbase']]

if 'finaldf' in st.session_state:
    st.subheader('Step 3: Upload & Download')
    st.write(st.session_state['finaldf'])
    one, two = st.columns(2)
    with one:
        # the estimated upload time is .1 seconds per row
        # display in minutes and seconds
        st.write('Estimated upload time: ', round(len(st.session_state['finaldf'])*.1/60, 2), ' minutes')
        if st.button('Upload to RedCap'):
            with st.spinner('Uploading to RedCap...'):
                r = project.import_records(st.session_state['finaldf'], 
                                    import_format='df', 
                                    overwrite='normal', 
                                    force_auto_number= True)
            st.write('Successfully uploaded ', r, ' rows')
        st.download_button('Download CSV', data=st.session_state['finaldf'].to_csv(index=False), file_name='ABL_upload.csv', mime='text/csv')
    with two:
        st.write("")