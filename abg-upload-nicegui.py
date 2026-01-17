from nicegui import ui, app
import pandas as pd
import numpy as np
import requests
from redcap import Project
import os
from io import BytesIO
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column definitions
fcols = ['Time Stamp', 'Date Calc', 'Time Calc', 'Subject', 'Sample', 'Patient ID', 'UPI', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']
rcols = ['Subject', 'Time Stamp', 'Date Calc', 'Time Calc', 'Sample', 'Patient ID', 'UPI', 'Session', 'pH','pCO2', 'pO2', 'sO2','COHb','MetHb','tHb']
allcols = fcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']
allcols_r = rcols + ['K+', 'Na+','Ca++','Cl-','Glucose','Lactate','p50','cBase']

class AppState:
    """Centralized state management for the application"""
    def __init__(self):
        self.uploaded_files: List = []
        self.df1: Optional[pd.DataFrame] = None
        self.edited_df: Optional[pd.DataFrame] = None
        self.upi_df: Optional[pd.DataFrame] = None
        self.finaldf: Optional[pd.DataFrame] = None
        self.errors: bool = False
        self.combined: bool = False
        self.project: Optional[Project] = None
        self.session_project: Optional[Project] = None

    def reset(self):
        """Reset all state"""
        self.__init__()

# Global state instance
state = AppState()

def load_project(key: str) -> Project:
    """Load a REDCap project using API key from environment or config"""
    # In production, use environment variables or secure config
    # For now, this expects REDCAP_ABG and REDCAP_SESSION as env vars
    api_key = os.getenv(key)
    if not api_key:
        logger.error(f"API key {key} not found in environment")
        ui.notify(f"Error: {key} not configured. Set environment variable.", type='negative')
        return None
    api_url = 'https://redcap.ucsf.edu/api/'
    return Project(api_url, api_key)

def feinerize(datafr: pd.DataFrame) -> pd.DataFrame:
    """Transform raw ABL data into feiner format"""
    try:
        logger.info('Feinerizing data...')
        # Separate timestamp into two columns
        datafr['Time'] = pd.to_datetime(datafr['Time'])
        datafr['Date Calc'] = datafr['Time'].dt.date
        datafr['Time Calc'] = datafr['Time'].dt.time

        # Separate patient ID into two columns
        try:
            datafr[['Subject', 'Sample']] = datafr['Patient ID'].astype(str).str.split(pat='.', expand=True)
        except Exception as e:
            ui.notify('Error splitting Patient ID column into "Subject" and "Sample". Expecting "3.21" or similar.', type='negative')
            raise

        # Rename columns
        datafr = datafr.rename(columns={
            "Time": 'Time Stamp',
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
            'cBase(Ecf) (mmol/L)':'cBase'
        })

        return datafr[allcols]
    except Exception as e:
        logger.error(f"Error in feinerize: {e}")
        raise

def validate_step2_data(edited_df: pd.DataFrame) -> tuple[bool, list]:
    """Validate data in step 2. Returns (has_errors, error_messages)"""
    errors = []

    # Check for null values
    if edited_df['Subject'].isnull().sum() > 0:
        timestamps = edited_df[edited_df['Subject'].isnull()]['Time Stamp'].tolist()
        errors.append(f'Subject column has null values: {timestamps}')

    if edited_df['Sample'].isnull().sum() > 0:
        timestamps = edited_df[edited_df['Sample'].isnull()]['Time Stamp'].tolist()
        errors.append(f'Sample column has null values: {timestamps}')

    if edited_df['Patient ID'].isnull().sum() > 0:
        timestamps = edited_df[edited_df['Patient ID'].isnull()]['Time Stamp'].tolist()
        errors.append(f'Patient ID column has null values: {timestamps}')

    # UPI validation: catch nulls, blanks, and invalid entries
    upi_str = edited_df['UPI'].astype('string').str.strip()
    invalid_upi_mask = upi_str.eq('') | upi_str.isna()
    upi_num = pd.to_numeric(upi_str, errors='coerce')
    invalid_upi_mask |= upi_num.isna()

    if invalid_upi_mask.any():
        timestamps = edited_df.loc[invalid_upi_mask, 'Time Stamp'].tolist()
        errors.append(f'UPI column has missing or invalid values: {timestamps}')

    # Consistency check: each (Subject, Date Calc) matches exactly one UPI
    _tmp = edited_df.copy()
    _tmp['Date Calc_norm'] = pd.to_datetime(_tmp['Date Calc'], errors='coerce').dt.date
    _tmp['Subject_norm'] = _tmp['Subject'].astype('string').str.strip()
    _tmp['UPI_norm'] = pd.to_numeric(_tmp['UPI'], errors='coerce').astype('Int64')
    _tmp_valid = _tmp.dropna(subset=['Date Calc_norm', 'Subject_norm', 'UPI_norm'])

    nuniq_per_group = (_tmp_valid.groupby(['Subject_norm','Date Calc_norm'])['UPI_norm'].nunique())
    err_group = nuniq_per_group[nuniq_per_group > 1]

    if not err_group.empty:
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
        errors.append("Each ABG file (Subject, Date Calc) must map to exactly one UPI. Conflicts found:")
        errors.append(detail.to_string())

    return len(errors) > 0, errors

def validate_session_data(upi_edits: pd.DataFrame) -> tuple[bool, list]:
    """Validate session data in step 3. Returns (has_errors, error_messages)"""
    errors = []

    # Check 0: All sessions must have values
    if upi_edits['Session'].isnull().sum() > 0:
        timestamps = upi_edits[upi_edits['Session'].isnull()]['Time Stamp'].tolist()
        errors.append(f'Please fill in all session values: {timestamps}')
        return True, errors

    # Check 1: Session maps to exactly one UPI
    ed = upi_edits.copy()
    ed['_SessionStr'] = ed['Session'].astype('Int64').astype('string').str.strip()
    ed['_UPIStr'] = ed['UPI'].astype('Int64').astype('string').str.strip()
    sess_to_upi = (ed.dropna(subset=['_SessionStr','_UPIStr']).groupby('_SessionStr')['_UPIStr'].nunique())
    err_sess = sess_to_upi[sess_to_upi > 1]

    if not err_sess.empty:
        err_list = [f"session {s} ({n} UPIs)" for s, n in err_sess.items()]
        errors.append(f"Some Session values map to multiple UPIs: {', '.join(err_list)}")
        return True, errors

    # Check 2: Session ↔ UPI must match REDCap SESSION database
    try:
        session_proj = load_project('REDCAP_SESSION')
        if not session_proj:
            errors.append("Could not load REDCap SESSION project")
            return True, errors

        df_session = pd.DataFrame(session_proj.export_records())
        df_session['_record_str'] = df_session['record_id'].astype('string').str.strip()
        df_session['_patient_str'] = df_session['patient_id'].astype('string').str.strip()

        check = ed[['Time Stamp', '_SessionStr', '_UPIStr']].merge(
            df_session[['_record_str', '_patient_str']],
            left_on='_SessionStr', right_on='_record_str', how='left'
        )

        mismatches = check[check['_patient_str'].notna() & (check['_patient_str'] != check['_UPIStr'])]

        if not mismatches.empty:
            errors.append("Session ↔ UPI mismatch vs REDCap SESSION:")
            errors.append(mismatches[['Time Stamp', '_SessionStr', '_UPIStr', '_patient_str']].to_string())
            return True, errors
    except Exception as e:
        errors.append(f"Error checking REDCap SESSION: {e}")
        return True, errors

    # Check 3: Session must not already exist in REDCap ABG database
    try:
        if not state.project:
            state.project = load_project('REDCAP_ABG')

        if not state.project:
            errors.append("Could not load REDCap ABG project")
            return True, errors

        df_abg = pd.DataFrame(state.project.export_records())
        s_abg = df_abg['session'].astype('string').str.strip()
        s_ed = ed['Session'].astype('Int64').astype('string').str.strip()

        session_already_in_redcap = sorted(set(s_ed.dropna()) & set(s_abg.dropna()))

        if session_already_in_redcap:
            errors.append(f"These Session IDs already exist in REDCap ABG: {session_already_in_redcap}")
            return True, errors
    except Exception as e:
        errors.append(f"Error checking REDCap ABG: {e}")
        return True, errors

    return False, errors

def create_ui():
    """Create the main UI"""

    # Initialize REDCap project
    try:
        state.project = load_project('REDCAP_ABG')
    except Exception as e:
        logger.error(f"Failed to load REDCap project: {e}")

    with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
        ui.label('Import ABL files into RedCap').classes('text-3xl font-bold mb-4')

        ui.markdown('''
This app will allow you to upload ABL files and import them into RedCap. Please follow the steps below:

1. **Upload files**: Drag and drop the raw CSVs from the ABLs into the spot below.
2. **Correct errors**: The app will alert you if there are missing values in the Subject, Sample, Patient ID, or UPI columns. You can correct these errors in the table below.
3. **Add session numbers**: The app will alert you if there are missing session numbers. Every UPI should have a session number. You can add session numbers in the table below.
4. **Upload to RedCap**: Once you have corrected all errors, you can upload the file to RedCap.
        ''')

        # Step 1: Upload files
        with ui.card().classes('w-full'):
            ui.label('Step 1: Upload Files').classes('text-2xl font-bold')

            upload_container = ui.column()
            step2_container = ui.column().classes('w-full')
            step3_container = ui.column().classes('w-full')
            step4_container = ui.column().classes('w-full')

            async def handle_upload(e):
                """Handle file upload"""
                state.uploaded_files = e.content
                ui.notify(f'Uploaded {len(state.uploaded_files)} file(s)', type='positive')

            async def combine_csvs():
                """Combine uploaded CSV files"""
                if not state.uploaded_files:
                    ui.notify('Please upload files first', type='warning')
                    return

                try:
                    dfs = []
                    for file_content in state.uploaded_files:
                        # file_content is bytes
                        df = pd.read_csv(BytesIO(file_content), encoding='cp1252', converters={'Patient ID': str})
                        dfs.append(df)

                    state.df1 = pd.concat(dfs, ignore_index=True)
                    state.df1 = feinerize(state.df1)
                    state.combined = True
                    state.edited_df = state.df1.copy()

                    ui.notify('Files combined successfully!', type='positive')

                    # Show step 2
                    create_step2()

                except Exception as e:
                    ui.notify(f'Error processing files: {e}', type='negative')
                    logger.error(f"Error in combine_csvs: {e}")

            with upload_container:
                ui.upload(
                    label='Choose CSV files',
                    multiple=True,
                    auto_upload=True,
                    on_upload=handle_upload
                ).props('accept=.csv').classes('w-full')

                ui.button('Combine CSVs', on_click=combine_csvs).props('color=primary')

            def create_step2():
                """Create Step 2: Correct Errors"""
                step2_container.clear()

                with step2_container:
                    ui.separator()
                    ui.label('Step 2: Correct Errors').classes('text-2xl font-bold mt-4')

                    # Create editable data grid
                    sorted_df = state.edited_df.sort_values(by='Time Stamp')

                    # Convert dataframe to dict format for aggrid
                    grid_data = sorted_df.to_dict('records')

                    # Define columns for aggrid
                    columns = [{'field': col, 'editable': True, 'sortable': True, 'filter': True}
                              for col in sorted_df.columns]

                    error_display = ui.column().classes('w-full')

                    grid = ui.aggrid({
                        'columnDefs': columns,
                        'rowData': grid_data,
                        'rowSelection': 'multiple',
                        'stopEditingWhenCellsLoseFocus': True,
                        'defaultColDef': {
                            'flex': 1,
                            'minWidth': 100,
                            'resizable': True
                        }
                    }).classes('w-full h-96')

                    async def validate_and_proceed():
                        """Validate data and proceed to step 3"""
                        # Get updated data from grid
                        updated_data = await grid.get_selected_rows()
                        if not updated_data:
                            # If no rows selected, get all rows
                            updated_data = grid.options['rowData']

                        # Convert back to DataFrame
                        state.edited_df = pd.DataFrame(updated_data)

                        # Remove rows with Patient ID == "0000"
                        if sum(state.edited_df['Patient ID'] == "0000") > 0:
                            ui.notify('Rows with Patient ID 0000 will be dropped', type='info')
                            state.edited_df = state.edited_df[state.edited_df["Patient ID"] != "0000"]

                        # Validate
                        has_errors, error_messages = validate_step2_data(state.edited_df)

                        error_display.clear()
                        with error_display:
                            if has_errors:
                                state.errors = True
                                for msg in error_messages:
                                    ui.label(msg).classes('text-red-600')
                                ui.notify('Please fix errors before proceeding', type='negative')
                            else:
                                state.errors = False
                                ui.notify('Validation passed!', type='positive')
                                create_step3()

                    ui.button('Validate and Continue', on_click=validate_and_proceed).props('color=primary')

            def create_step3():
                """Create Step 3: Add Session Numbers"""
                if state.errors:
                    return

                step3_container.clear()

                with step3_container:
                    ui.separator()
                    ui.label('Step 3: Add Session Numbers').classes('text-2xl font-bold mt-4')

                    # Normalize merge keys
                    edited_df_for_sessions = state.edited_df.copy()
                    edited_df_for_sessions['Date Calc'] = pd.to_datetime(edited_df_for_sessions['Date Calc'], errors='coerce')
                    edited_df_for_sessions['UPI'] = pd.to_numeric(edited_df_for_sessions['UPI'], errors='coerce').astype('Int64')

                    # Build unique (Date Calc, UPI) pairs
                    upi_df = (
                        edited_df_for_sessions[['Date Calc', 'UPI', 'Time Stamp']]
                        .drop_duplicates(subset=['Date Calc', 'UPI'])
                        .sort_values(['Date Calc', 'UPI'])
                        .reset_index(drop=True)
                    )
                    upi_df['Session'] = np.nan
                    state.upi_df = upi_df

                    # Create editable grid for sessions
                    session_grid_data = upi_df.to_dict('records')
                    session_columns = [{'field': col, 'editable': (col == 'Session'), 'sortable': True, 'filter': True}
                                      for col in upi_df.columns]

                    session_error_display = ui.column().classes('w-full')

                    session_grid = ui.aggrid({
                        'columnDefs': session_columns,
                        'rowData': session_grid_data,
                        'rowSelection': 'multiple',
                        'stopEditingWhenCellsLoseFocus': True,
                        'defaultColDef': {
                            'flex': 1,
                            'minWidth': 150,
                            'resizable': True
                        }
                    }).classes('w-full h-64')

                    async def add_session_numbers():
                        """Add session numbers and validate"""
                        # Get updated data from grid
                        updated_session_data = session_grid.options['rowData']
                        upi_edits = pd.DataFrame(updated_session_data)

                        # Normalize Session column
                        sess_str = upi_edits['Session'].astype('string').str.strip()
                        upi_edits['Session'] = pd.to_numeric(sess_str, errors='coerce').astype('Int64')

                        # Normalize keys
                        upi_edits['Date Calc'] = pd.to_datetime(upi_edits['Date Calc'], errors='coerce')
                        upi_edits['UPI'] = pd.to_numeric(upi_edits['UPI'], errors='coerce').astype('Int64')

                        # Remove blank rows
                        blank_row_mask = upi_edits[['Date Calc', 'UPI', 'Session']].isna().all(axis=1)
                        if blank_row_mask.any():
                            upi_edits = upi_edits.loc[~blank_row_mask].copy()

                        # Validate
                        has_errors, error_messages = validate_session_data(upi_edits)

                        session_error_display.clear()
                        with session_error_display:
                            if has_errors:
                                state.errors = True
                                state.finaldf = None
                                for msg in error_messages:
                                    ui.label(msg).classes('text-red-600')
                                ui.notify('Please fix errors before proceeding', type='negative')
                            else:
                                state.errors = False

                                # Merge session numbers into original df
                                upi_edits_merge = upi_edits.drop(columns=['Time Stamp']).copy()

                                # Ensure dtypes match before merge
                                edited_df = state.edited_df.copy()
                                edited_df['Date Calc'] = pd.to_datetime(edited_df['Date Calc'], errors='coerce')
                                edited_df['UPI'] = pd.to_numeric(edited_df['UPI'], errors='coerce').astype('Int64')

                                state.finaldf = edited_df.merge(upi_edits_merge, on=['Date Calc','UPI'], how='left')
                                state.finaldf = state.finaldf[allcols_r]
                                state.finaldf.rename_axis('record_id', inplace=True)

                                # Drop Subject column and rename for REDCap
                                state.finaldf = state.finaldf.drop(columns=['Subject'])
                                state.finaldf = state.finaldf.rename(columns={
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
                                    'cBase':'cbase'
                                })

                                # Reorder columns
                                state.finaldf = state.finaldf[['subject', 'time_stamp', 'date_calc', 'time_calc', 'sample',
                                                               'patient_id', 'session', 'ph', 'pco2', 'po2', 'so2', 'cohb',
                                                               'methb', 'thb', 'k', 'na', 'ca', 'cl', 'glucose', 'lactate',
                                                               'p50', 'cbase']]

                                ui.notify('Session numbers added successfully!', type='positive')
                                create_step4()

                    ui.button('Add Session Numbers to File', on_click=add_session_numbers).props('color=primary')

            def create_step4():
                """Create Step 4: Upload & Download"""
                if state.finaldf is None or state.errors:
                    return

                step4_container.clear()

                with step4_container:
                    ui.separator()
                    ui.label('Step 4: Upload & Download').classes('text-2xl font-bold mt-4')

                    # Display final dataframe
                    final_grid_data = state.finaldf.to_dict('records')
                    final_columns = [{'field': col, 'sortable': True, 'filter': True}
                                    for col in state.finaldf.columns]

                    ui.aggrid({
                        'columnDefs': final_columns,
                        'rowData': final_grid_data,
                        'defaultColDef': {
                            'flex': 1,
                            'minWidth': 100,
                            'resizable': True
                        }
                    }).classes('w-full h-96')

                    with ui.row().classes('gap-4 mt-4'):
                        with ui.column():
                            est_time = round(len(state.finaldf) * 0.1 / 60, 2)
                            ui.label(f'Estimated upload time: {est_time} minutes')

                            async def upload_to_redcap():
                                """Upload data to REDCap"""
                                if not state.project:
                                    ui.notify('REDCap project not configured', type='negative')
                                    return

                                try:
                                    with ui.spinner():
                                        r = state.project.import_records(
                                            state.finaldf,
                                            import_format='df',
                                            overwrite='normal',
                                            force_auto_number=True
                                        )
                                    ui.notify(f'Successfully uploaded {r} rows', type='positive')
                                except Exception as e:
                                    ui.notify(f'Upload failed: {e}', type='negative')
                                    logger.error(f"Upload error: {e}")

                            ui.button('Upload to RedCap', on_click=upload_to_redcap).props('color=primary')

                            # Download button
                            csv_data = state.finaldf.to_csv(index=False)
                            ui.button(
                                'Download CSV',
                                on_click=lambda: ui.download(csv_data.encode(), 'ABL_upload.csv')
                            ).props('color=secondary')

@ui.page('/')
def index():
    """Main page"""
    create_ui()

# Configuration notes
ui.markdown('''
---
**Configuration Notes:**

For this application to work, you need to set the following environment variables:
- `REDCAP_ABG`: Your REDCap API key for the ABG project
- `REDCAP_SESSION`: Your REDCap API key for the SESSION project

Example:
```bash
export REDCAP_ABG="your_api_key_here"
export REDCAP_SESSION="your_session_api_key_here"
python abg-upload-nicegui.py
```
''').classes('text-sm text-gray-600 p-4')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='ABG Upload to RedCap',
        port=8080,
        reload=True,
        show=True
    )
