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
    print('feinerizing')
    datafr['Time'] = pd.to_datetime(datafr['Time'])
    print('time converted')
    datafr['Date Calc'] = datafr['Time'].dt.date
    print('date calc')
    datafr['Time Calc'] =  datafr['Time'].dt.time

    #separate patient ID into two columns
    try:
        datafr[['Subject', 'Sample']] = datafr['Patient ID'].astype(str).str.split(pat='.', expand=True)
    except Exception as e:
        st.error('Error splitting Patient ID column into "Subject" and "Sample". Expecting "3.21" or similar. Check the Patient ID column for errors.')
        st.stop()
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
    # ---- UPI validation: catch nulls, blanks, and invalid entries ----
    upi_str = edited_df['UPI'].astype('string').str.strip()
    invalid_upi_mask = upi_str.eq('') | upi_str.isna()
    # Try converting to numeric and flag anything that fails
    upi_num = pd.to_numeric(upi_str, errors='coerce')
    invalid_upi_mask |= upi_num.isna()
    if invalid_upi_mask.any():
        st.write('UPI column has missing or invalid values:',
                edited_df.loc[invalid_upi_mask, 'Time Stamp'].tolist())
        st.session_state['errors'] = True
    # ---- Consistency check: each (Subject, Date Calc) matches exactly one UPI----
    _tmp = edited_df.copy()
    # Normalize to avoid false mismatches
    _tmp['Date Calc_norm'] = pd.to_datetime(_tmp['Date Calc'], errors='coerce').dt.date
    _tmp['Subject_norm']   = _tmp['Subject'].astype('string').str.strip()
    _tmp['UPI_norm'] = pd.to_numeric(_tmp['UPI'], errors='coerce').astype('Int64')
    _tmp_valid = _tmp.dropna(subset=['Date Calc_norm', 'Subject_norm', 'UPI_norm'])
    # Count unique UPI per (Subject, Date Calc)
    nuniq_per_group = (
    _tmp_valid.groupby(['Subject_norm','Date Calc_norm'])['UPI_norm']
              .nunique()
    )
    # Violations: groups with >1 distinct UPI
    err_group = nuniq_per_group[nuniq_per_group > 1]
    if not err_group.empty:
        st.error("Each ABG file (Subject, Date Calc) must map to exactly one UPI. Conflicts found.")
        # Detail rows to help fix: show all conflicting rows
        detail = (
            _tmp_valid
            .set_index(['Subject_norm','Date Calc_norm'])
            .loc[err_group.index] 
            .reset_index()
            [['Time Stamp','Subject_norm','Date Calc_norm','UPI_norm']]
            .sort_values(['Subject_norm','Date Calc_norm','UPI_norm','Time Stamp'])
            .rename(columns={
                'Subject_norm':'Subject',
                'Date Calc_norm':'Date Calc',
                'UPI_norm':'UPI'
            })
            .reset_index(drop=True)
        )
        st.dataframe(detail)
        st.session_state['errors'] = True
    
    # store the corrected dataframe for Step 3
    st.session_state['edited_df'] = edited_df
    
##############################################
if 'errors' not in st.session_state:
    st.write('')
elif st.session_state['errors'] == False:
    st.subheader('Step 3: Add session numbers')
    
    # use the user-edited table from Step 2 if available; fall back to df1
    edited_df_for_sessions = st.session_state.get('edited_df', st.session_state['df1']).copy()

    # normalize merge keys so they match later
    edited_df_for_sessions['Date Calc'] = pd.to_datetime(edited_df_for_sessions['Date Calc'], errors='coerce')
    edited_df_for_sessions['UPI'] = pd.to_numeric(edited_df_for_sessions['UPI'], errors='coerce').astype('Int64')

    # build unique (Date Calc, UPI) pairs from Step 2 edited data
    upi_df = (
        edited_df_for_sessions[['Date Calc', 'UPI', 'Time Stamp']]
        .drop_duplicates(subset=['Date Calc', 'UPI'])
        .sort_values(['Date Calc', 'UPI'])
        .reset_index(drop=True)
    )
    upi_df['Session'] = np.nan
    upi_edits = st.data_editor(upi_df, num_rows='dynamic', key='upi_editor')
    
    # treat blanks as NaN, strip spaces, and coerce to Int64; invalids become NaN
    sess_str = upi_edits['Session'].astype('string').str.strip()
    upi_edits['Session'] = pd.to_numeric(sess_str, errors='coerce').astype('Int64')
    
    # normalize keys on the mapping as well (to match merge dtypes later)
    upi_edits['Date Calc'] = pd.to_datetime(upi_edits['Date Calc'], errors='coerce')
    upi_edits['UPI'] = pd.to_numeric(upi_edits['UPI'], errors='coerce').astype('Int64')
    
    blank_row_mask = upi_edits[['Date Calc', 'UPI', 'Session']].isna().all(axis=1)
    if blank_row_mask.any():
        upi_edits = upi_edits.loc[~blank_row_mask].copy()

    if st.button('Add Session Numbers to file'):
        # 0) check if any of the values in the Session column are null
        if upi_edits['Session'].isnull().sum() >0:
            st.write('please fill in all session values')
            #write the time stamps of the rows with null values
            st.write(upi_edits[upi_edits['Session'].isnull()]['Time Stamp'].tolist())
            st.session_state['errors'] = True
            st.session_state.pop('finaldf', None)
            st.stop()
            
        # 1) check if Session maps to exactly one UPI
        ed = upi_edits.copy()
        ed['_SessionStr'] = ed['Session'].astype('Int64').astype('string').str.strip()
        ed['_UPIStr']     = ed['UPI'].astype('Int64').astype('string').str.strip()
        sess_to_upi = (
            ed.dropna(subset=['_SessionStr','_UPIStr'])
            .groupby('_SessionStr')['_UPIStr'].nunique()
        )
        err_sess = sess_to_upi[sess_to_upi > 1]
        if not err_sess.empty:
            st.error("Some Session values map to multiple UPIs: " +
                    ", ".join([f"session {s} ({n} UPIs)" for s, n in err_sess.items()]))
            st.session_state['errors'] = True
            st.session_state.pop('finaldf', None)
            st.stop()
                
        # 2) check if the session number and patient ID pair entered matches with what is in REDCap SESSION, if the session number exists in REDCap SESSION database. 
        session_proj = load_project('REDCAP_SESSION')
        df_session = pd.DataFrame(session_proj.export_records())
        df_session['_record_str']  = df_session['record_id'].astype('string').str.strip()
        df_session['_patient_str'] = df_session['patient_id'].astype('string').str.strip()
        check = ed[['Time Stamp', '_SessionStr', '_UPIStr']].merge(
            df_session[['_record_str', '_patient_str']],
            left_on='_SessionStr', right_on='_record_str', how='left'
        )
        mismatches = check[
            check['_patient_str'].notna() & (check['_patient_str'] != check['_UPIStr'])
        ]
        if not mismatches.empty:
            st.error("Session ↔ UPI mismatch vs REDCap SESSION:")
            st.dataframe(
                mismatches[['Time Stamp', '_SessionStr', '_UPIStr', '_patient_str']]
                .rename(columns={'_SessionStr':'Session',
                                '_UPIStr':'UPI (ABG file)',
                                '_patient_str':'UPI (REDCap SESSION)'})
            )
            st.session_state['errors'] = True
            st.session_state.pop('finaldf', None)
            st.stop()
            
        # 3) check if the Session already exists in REDCap ABG database
        df_abg = pd.DataFrame(project.export_records())
        s_abg = df_abg['session'].astype('string').str.strip()             
        s_ed  = ed['Session'].astype('Int64').astype('string').str.strip()     
        # list the duplicates (unique IDs)
        session_already_in_redcap = sorted(set(s_ed.dropna()) & set(s_abg.dropna()))
        if session_already_in_redcap:
            st.error(f"These Session IDs already exist in REDCap ABG: {session_already_in_redcap}")
            st.session_state['errors'] = True
            st.session_state.pop('finaldf', None)
            st.stop()
        
        # ONLY when all checks pass, build finaldf:
        st.session_state['errors'] = False
        #merge session numbers into the original df 
        st.session_state['upi_df'] = upi_edits.reset_index()
        # drop 'Time Stamp' from upi_edits
        st.session_state['upi_df'] = upi_edits.reset_index()
        upi_edits.drop(columns=['Time Stamp'], inplace=True)
        # Make sure the edited_df keys have the same dtype as upi_edits before merging
        edited_df['Date Calc'] = pd.to_datetime(edited_df['Date Calc'], errors='coerce')
        edited_df['UPI'] = pd.to_numeric(edited_df['UPI'], errors='coerce').astype('Int64')
        st.session_state['finaldf'] = edited_df.merge(upi_edits, on=['Date Calc','UPI'],how='left')
        st.session_state['finaldf'] = st.session_state['finaldf'][allcols_r]
        st.session_state['finaldf'].rename_axis('record_id', inplace=True)
                    
        #drop subject column. I know this doesn't make a lot of sense given that we added during feinerize
        #but this column doesn't exist in redcap, so I want to remove it. 
        # initally i wanted to have a function to be able to download a feiner dataframe, but I think we'll just create a separate script for that
        st.session_state['finaldf'] = st.session_state['finaldf'].drop(columns=['Subject'])
        
        #rename columns to match case in redcap
        st.session_state['finaldf'] = st.session_state['finaldf'].rename(columns={"record_id": 'record_id', 
                                                                                #   'Subject': 'subject',
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

if ('finaldf' in st.session_state) and (st.session_state.get('errors') is False): # make sure passing all checks before uploading
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