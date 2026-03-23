import streamlit as st
import pandas as pd
import numpy as np
import requests
from redcap import Project

def load_project(key, api_url):
    api_key = st.secrets[key]
    project = Project(api_url, api_key)
    return project

#fcols = feiner columns that he uses for processing
fcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'UPI', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']

#rcols = redcap columns including Session when it is created
rcols = ['Subject', 'Time Stamp', 'Date Calc', 'Time Calc', 'Sample', 'Patient ID', 'UPI', 'Session', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb','machine_serial']

# allcols = the rest of the columns we actually want + feiner cols
allcols = fcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']

# allcols_r = all columns we want + redcap cols
allcols_r = rcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']


def _format_date_human(date_value):
    dt = pd.to_datetime(date_value, errors='coerce')
    if pd.isna(dt):
        return str(date_value)
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def normalize_id(value):
    if pd.isna(value):
        return pd.NA
    value_str = str(value).strip()
    if value_str == '' or value_str.lower() == 'nan':
        return pd.NA
    try:
        return str(int(float(value_str)))
    except (TypeError, ValueError):
        return value_str


def split_patient_id_columns(patient_id_series, location='UCSF'):
    patient_id = patient_id_series.astype('string').str.strip()
    if location == 'Uganda':
        integer_sample = patient_id.map(normalize_id).astype('string').str.strip()
        has_dot = patient_id.str.contains(r'\.', regex=True, na=False)
        sample_part = integer_sample.mask(has_dot, pd.NA)
        invalid_pid_mask = patient_id.isna() | patient_id.eq('') | has_dot | sample_part.isna() | sample_part.eq('')
        # Uganda ABG files use a simple integer Patient ID that maps directly
        # to the sample number.
        subject_part = sample_part.mask(invalid_pid_mask, pd.NA)
        sample_part = sample_part.mask(invalid_pid_mask, pd.NA)
        return subject_part, sample_part, invalid_pid_mask

    split_pid = patient_id.str.split('.', n=1, expand=True)
    if split_pid.shape[1] < 2:
        split_pid[1] = pd.NA

    subject_part = split_pid[0].astype('string').str.strip()
    sample_part = split_pid[1].astype('string').str.strip()
    has_dot = patient_id.str.contains(r'\.', regex=True, na=False)
    invalid_pid_mask = (
        patient_id.isna() | patient_id.eq('') |
        (~has_dot) |
        subject_part.isna() | subject_part.eq('') |
        sample_part.isna() | sample_part.eq('')
    )

    subject_part = subject_part.mask(invalid_pid_mask, pd.NA)
    sample_part = sample_part.mask(invalid_pid_mask, pd.NA)
    return subject_part, sample_part, invalid_pid_mask

def extract_serial_number(filename):
    """
    Extract the serial number from filename.
    Expected format: "PatLog - 2026-01-06 10_18_31_I393-092Rxxxxxxx31.csv"
    Serial number is expected to be between the last '-' and '.csv'
    """
    try:
        if not filename.endswith('.csv'):
            raise ValueError(f"File {filename} is not a CSV file")
        
        name_without_ext = filename[:-4]  # Remove '.csv'
        
        # Find the last '-' and extract everything after it
        last_dash_index = name_without_ext.rfind('-')
        if last_dash_index == -1:
            raise ValueError(f"Filename format not expected: {filename}")
        
        serial_number = name_without_ext[last_dash_index + 1:]
        if not serial_number or serial_number.strip() == '':
            raise ValueError(f"Serial number not found in filename: {filename}")
        
        return serial_number.strip()
    
    except Exception as e:
        raise ValueError(f"Error extracting serial number from {filename}: {str(e)}")
    
def feinerize(datafr, serial_number, location='UCSF'):
    #separate timestamp into two columns
    print('feinerizing')
    datafr['Time'] = pd.to_datetime(datafr['Time'])
    print('time converted')
    datafr['Date Calc'] = datafr['Time'].dt.date
    print('date calc')
    datafr['Time Calc'] =  datafr['Time'].dt.time

    #separate patient ID into two columns
    subject_part, sample_part, invalid_pid_mask = split_patient_id_columns(
        datafr['Patient ID'],
        location=location,
    )
    if invalid_pid_mask.any():
        bad_rows = datafr.loc[invalid_pid_mask, ['Time', 'Patient ID']].copy()
        bad_rows['Time'] = bad_rows['Time'].astype('string')
        bad_rows['Patient ID'] = bad_rows['Patient ID'].astype('string')
        problem_rows = "; ".join(
            [f"Time {row['Time']} with Patient ID '{row['Patient ID']}'"
             for _, row in bad_rows.iterrows()]
        )
        expected_format = "a simple integer like '21'" if location == 'Uganda' else "format like '3.21'"
        st.warning(
            'Some Patient ID values could not be split into "Subject" and "Sample". '
            f"Problem rows: {problem_rows}. Expected {expected_format}. "
            'Those rows will stay editable in Step 2 so you can correct them.'
        )

    datafr['Subject'] = subject_part
    datafr['Sample'] = sample_part
    
    # Add machine serial number to all rows
    datafr['machine_serial'] = serial_number
    
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

    return datafr[allcols + ['machine_serial']]


#start layout

st.header('Import ABL files into RedCap')

st.write('This app will allow you to upload ABL files and import them into RedCap. Please follow the steps below.')
st.markdown('''
    1. **Upload files**: Drag and drop the raw CSVs from the abls into the spot below.
    2. **Correct errors**: The app will alert you if there are missing values in the Subject, Sample, Patient ID, or UPI columns. You can correct these errors in the table below.
    3. **Add session numbers**: The app will alert you if there are missing session numbers. Every UPI should have a session number. You can add session numbers in the table below.
    4. **Upload to RedCap**: Once you have corrected all errors, you can upload the file to RedCap.
         ''')

st.subheader('Select Location')
location = st.selectbox('Select Location', ['UCSF', 'Uganda'], index=0)

if location == 'Uganda':
    # abg_key = 'Uganda_REDCAP_ABG'
    # session_key = 'Uganda_REDCAP_SESSION'
    # api_url = 'https://redcap.ace.ac.ug/api/'
    abg_key = 'Uganda_REDCAP_ABG_UCSF'
    session_key = 'Uganda_REDCAP_SESSION_UCSF'
    api_url = 'https://redcap.ucsf.edu/api/'
else:
    abg_key = 'REDCAP_ABG'
    session_key = 'REDCAP_SESSION'
    api_url = 'https://redcap.ucsf.edu/api/'

project = load_project(abg_key, api_url)

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
        current_filename = None
        try:
            for num, file in enumerate(uploaded_files):
                current_filename = file.name
                # Extract serial number from filename
                serial_number = extract_serial_number(file.name)
                st.info(f"File: {file.name} → Machine Serial: {serial_number}")
                
                df = pd.read_csv(file, encoding = 'cp1252', converters={'Patient ID': str})
                df = feinerize(df, serial_number, location=location)
                dfs.append(df)
            df1 = pd.concat(dfs, ignore_index=True)
            # df1 = feinerize(df1)
            st.session_state['combined'] = True
            st.session_state['df1'] = df1
            
        except ValueError as e:
            st.error(str(e))
            if current_filename:
                st.error(
                    "Problem filename: "
                    f"{current_filename}. Please use format 'PatLog - YYYY-MM-DD HH_MM_SS-SERIALNUMBER.csv'."
                )
            else:
                st.error("Please ensure all filenames follow the format: 'PatLog - YYYY-MM-DD HH_MM_SS-SERIALNUMBER.csv'.")
            st.stop()
        
##############################################
if 'combined' in st.session_state:
    st.subheader('Step 2: Correct errors')
    edited_df = st.data_editor(st.session_state['df1'].sort_values(by='Time Stamp'), num_rows='dynamic', key='data_editor')
    # Rebuild Subject and Sample from the edited Patient ID values so users can
    # fix bad IDs here instead of being blocked earlier in the upload flow.
    edited_df['Subject'], edited_df['Sample'], invalid_pid_mask = split_patient_id_columns(
        edited_df['Patient ID'],
        location=location,
    )
    #count the number of null values in Subject, sample, patient id, and UPI columns
    df1= edited_df
    st.session_state['errors'] = False
    subject_null_mask = edited_df['Subject'].isnull() & ~invalid_pid_mask
    sample_null_mask = edited_df['Sample'].isnull() & ~invalid_pid_mask
    if subject_null_mask.any():
        st.write('Subject column has null values: ', edited_df.loc[subject_null_mask, 'Time Stamp'].tolist())
        st.session_state['errors'] = True
    if sample_null_mask.any():
        st.write('Sample column has null values: ', edited_df.loc[sample_null_mask, 'Time Stamp'].tolist())
        st.session_state['errors'] = True
    if invalid_pid_mask.any():
        expected_format = "a simple integer like '21'" if location == 'Uganda' else "'3.21'"
        st.write(
            f"Patient ID is missing or not in the right format ({expected_format}). Please fix Patient ID in these rows. Subject and Sample are filled automatically from Patient ID: ",
            edited_df.loc[invalid_pid_mask, 'Time Stamp'].tolist()
        )
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
    # UCSF Patient ID values encode subject/sample (for example, "3.21"), so
    # it is reasonable there to expect each subject/date pair to map to one
    # accession number. Uganda Patient ID values are sample numbers, and the
    # same sample number can legitimately recur on the same date under
    # different accession numbers, so we skip this check there.
    if location == 'UCSF':
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
            conflict_rows = (
                _tmp_valid
                .set_index(['Subject_norm', 'Date Calc_norm'])
                .loc[err_group.index]
                .reset_index()
            )
            conflict_summary = []
            for (subject, date_calc), grp in conflict_rows.groupby(['Subject_norm', 'Date Calc_norm']):
                upis = sorted(grp['UPI_norm'].dropna().astype('Int64').astype('string').unique().tolist())
                conflict_summary.append(
                    f"Subject {subject} on {_format_date_human(date_calc)} listed with UPI {', '.join(upis)}"
                )
            conflict_lines = "\n".join([f"- {item}" for item in conflict_summary])
            st.error(
                "Each Subject and Date pair must map to exactly one UPI.\n\n"
                "Conflicts found:\n"
                f"{conflict_lines}\n\n"
                "Please check and update."
            )
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
            sess_upi_map = (
                ed.dropna(subset=['_SessionStr', '_UPIStr'])
                .groupby('_SessionStr')['_UPIStr']
                .apply(lambda s: sorted(set(s.astype('string').tolist())))
            )
            conflict_items = [
                f"Session {s} listed with UPI {', '.join(sess_upi_map.loc[s])}" for s in err_sess.index
            ]
            conflicting_lines = "\n".join([f"- {item}" for item in conflict_items])
            st.error(
                "Some Session values map to multiple UPIs.\n\n"
                "Conflicts found:\n"
                f"{conflicting_lines}\n\n"
                "Please check and update."
            )
            st.session_state['errors'] = True
            st.session_state.pop('finaldf', None)
            st.stop()
                
        # 2) check if Session ↔ UPI matches REDCap SESSION when reference records exist
        session_proj = load_project(session_key, api_url)
        df_session = pd.DataFrame(session_proj.export_records())
        required_session_cols = {'record_id', 'patient_id'}
        if df_session.empty or not required_session_cols.issubset(df_session.columns):
            st.info(
                "REDCap SESSION has no records yet (or is missing record_id/patient_id). "
                "Skipping Session ↔ UPI cross-check."
            )
        else:
            df_session['_record_str'] = df_session['record_id'].astype('string').str.strip()
            df_session['_patient_str'] = df_session['patient_id'].astype('string').str.strip()
            df_session = df_session[['_record_str', '_patient_str']]
            df_session['_record_str'] = df_session['_record_str'].replace('', pd.NA)
            df_session['_patient_str'] = df_session['_patient_str'].replace('', pd.NA)
            df_session = df_session.dropna(subset=['_record_str', '_patient_str'])
            check = ed[['Time Stamp', '_SessionStr', '_UPIStr']].merge(
                df_session,
                left_on='_SessionStr', right_on='_record_str', how='left'
            )
            mismatches = check[
                check['_patient_str'].notna() & (check['_patient_str'] != check['_UPIStr'])
            ]
            if not mismatches.empty:
                mismatch_summary = []
                mismatch_unique = mismatches[['_SessionStr', '_UPIStr', '_patient_str']].drop_duplicates()
                for _, row in mismatch_unique.iterrows():
                    mismatch_summary.append(
                        f"Session {row['_SessionStr']} has UPI {row['_UPIStr']} in file but {row['_patient_str']} in REDCap SESSION"
                    )
                mismatch_lines = "\n".join([f"- {item}" for item in mismatch_summary])
                mismatch_header = (
                    "Session number entered does not match with records in REDCap SESSION."
                    if len(mismatch_summary) == 1
                    else "Session numbers entered do not match with records in REDCap SESSION."
                )
                st.error(
                    f"{mismatch_header}\n\n"
                    "Conflicts found:\n"
                    f"{mismatch_lines}\n\n"
                    "Please check and update."
                )
                st.dataframe(
                    mismatches[['Time Stamp', '_SessionStr', '_UPIStr', '_patient_str']]
                    .rename(columns={'_SessionStr': 'Session',
                                    '_UPIStr': 'UPI (ABG file)',
                                    '_patient_str': 'UPI (REDCap SESSION)'})
                )
                st.session_state['errors'] = True
                st.session_state.pop('finaldf', None)
                st.stop()
            
        # 3) check if the Session/UPI/machine_serial combination already exists in REDCap ABG database
        df_abg = pd.DataFrame(project.export_records())
        required_abg_cols = {'session', 'patient_id', 'machine_serial'}
        if df_abg.empty or not required_abg_cols.issubset(df_abg.columns):
            st.info(
                "REDCap ABG has no existing session/patient_id/machine_serial records yet. "
                "Skipping duplicate Session/UPI/machine check."
            )
        else:
            df_abg['_SessionStr'] = df_abg['session'].astype('string').str.strip().map(normalize_id)
            df_abg['_UPIStr'] = df_abg['patient_id'].astype('string').str.strip().map(normalize_id)
            df_abg['_MachineSerialStr'] = df_abg['machine_serial'].astype('string').str.strip()
            df_abg['_SessionStr'] = df_abg['_SessionStr'].replace('', pd.NA)
            df_abg['_UPIStr'] = df_abg['_UPIStr'].replace('', pd.NA)
            df_abg['_MachineSerialStr'] = df_abg['_MachineSerialStr'].replace('', pd.NA)
            abg_pairs = (
                df_abg[['_SessionStr', '_UPIStr', '_MachineSerialStr']]
                .dropna(subset=['_SessionStr', '_UPIStr', '_MachineSerialStr'])
                .drop_duplicates()
            )

            duplicate_check_rows = edited_df.copy()
            duplicate_check_rows['Date Calc'] = pd.to_datetime(duplicate_check_rows['Date Calc'], errors='coerce')
            duplicate_check_rows['UPI'] = pd.to_numeric(duplicate_check_rows['UPI'], errors='coerce').astype('Int64')

            # Build duplicate-check keys from the uploaded ABG rows, then add the
            # Session values from the helper table. We do it this way because the
            # helper table does not include machine_serial, but the uploaded rows do.
            incoming_pairs = (
                duplicate_check_rows[['Time Stamp', 'Date Calc', 'UPI', 'machine_serial']]
                .merge(
                    upi_edits[['Date Calc', 'UPI', 'Session']],
                    on=['Date Calc', 'UPI'],
                    how='left'
                )
            )
            incoming_pairs['_SessionStr'] = incoming_pairs['Session'].astype('string').str.strip().map(normalize_id)
            incoming_pairs['_UPIStr'] = incoming_pairs['UPI'].astype('string').str.strip().map(normalize_id)
            incoming_pairs['_MachineSerialStr'] = incoming_pairs['machine_serial'].astype('string').str.strip()
            incoming_pairs['_SessionStr'] = incoming_pairs['_SessionStr'].replace('', pd.NA)
            incoming_pairs['_UPIStr'] = incoming_pairs['_UPIStr'].replace('', pd.NA)
            incoming_pairs['_MachineSerialStr'] = incoming_pairs['_MachineSerialStr'].replace('', pd.NA)
            incoming_pairs = (
                incoming_pairs[['Time Stamp', '_SessionStr', '_UPIStr', '_MachineSerialStr']]
                .dropna(subset=['_SessionStr', '_UPIStr', '_MachineSerialStr'])
                .drop_duplicates()
            )

            duplicate_rows = (
                incoming_pairs
                .merge(abg_pairs, on=['_SessionStr', '_UPIStr', '_MachineSerialStr'], how='inner')
                .sort_values(['_SessionStr', '_UPIStr', '_MachineSerialStr', 'Time Stamp'])
            )
            if not duplicate_rows.empty:
                duplicate_pairs = duplicate_rows[['_SessionStr', '_UPIStr', '_MachineSerialStr']].drop_duplicates()
                duplicate_items = [
                    f"Session {row['_SessionStr']} with UPI {row['_UPIStr']} on machine {row['_MachineSerialStr']}"
                    for _, row in duplicate_pairs.iterrows()
                ]
                duplicate_lines = "\n".join([f"- {item}" for item in duplicate_items])
                st.error(
                    "These Session, UPI, and machine serial combinations already exist in REDCap ABG.\n\n"
                    "Conflicts found:\n"
                    f"{duplicate_lines}\n\n"
                    "Please check and update."
                )
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
        st.session_state['finaldf'] = st.session_state['finaldf'][['subject', 'time_stamp', 'date_calc', 'time_calc', 'sample', 'patient_id', 'session', 'ph', 'pco2', 'po2', 'so2', 'cohb', 'methb', 'thb', 'k', 'na', 'ca', 'cl', 'glucose', 'lactate', 'p50', 'cbase', 'machine_serial']]

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
