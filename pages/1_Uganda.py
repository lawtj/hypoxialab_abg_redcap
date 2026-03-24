import pandas as pd
import streamlit as st

from shared import extract_serial_number, format_date_human, load_project, normalize_id


PAGE_PREFIX = "uganda"


def state_key(name):
    return f"{PAGE_PREFIX}_{name}"


def clear_page_state():
    for key in [
        state_key("uploaded"),
        state_key("combined"),
        state_key("df1"),
        state_key("errors"),
        state_key("edited_df"),
        state_key("session_df"),
        state_key("finaldf"),
    ]:
        st.session_state.pop(key, None)


measurement_cols = [
    "pH",
    "pCO2",
    "pO2",
    "sO2",
    "COHb",
    "MetHb",
    "tHb",
    "K+",
    "Na+",
    "Ca++",
    "Cl-",
    "Glucose",
    "Lactate",
    "p50",
    "cBase",
]

editable_cols = (
    [
        "Time Stamp",
        "Date Calc",
        "Time Calc",
        "Sample ID",
        "Session ID",
    ]
    + measurement_cols
    + ["machine_serial"]
)


def normalize_integer_string(value):
    if pd.isna(value):
        return pd.NA
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "nan":
        return pd.NA
    if value_str.lstrip("+-").isdigit():
        return str(int(value_str))
    if "." in value_str:
        whole_part, fractional_part = value_str.split(".", 1)
        if whole_part.lstrip("+-").isdigit() and fractional_part != "" and set(fractional_part) <= {"0"}:
            return str(int(whole_part))
    return pd.NA


def prepare_uploaded_dataframe(datafr, serial_number):
    # Step 1 normalization: keep the raw Uganda identifiers as-is, and only
    # standardize timestamps, machine serial, and column names for the UI.
    datafr["Time"] = pd.to_datetime(datafr["Time"])
    datafr["Date Calc"] = datafr["Time"].dt.date
    datafr["Time Calc"] = datafr["Time"].dt.time

    if "Sample ID" not in datafr.columns:
        raise ValueError("Missing required column 'Sample ID' for Uganda files")

    datafr["machine_serial"] = serial_number

    datafr = datafr.rename(
        columns={
            "Time": "Time Stamp",
            "Accession number": "Session ID",
            "pCO2 (mmHg)": "pCO2",
            "pO2 (mmHg)": "pO2",
            "sO2 (%)": "sO2",
            "COHb (%)": "COHb",
            "MetHb (%)": "MetHb",
            "tHb (g/dL)": "tHb",
            "K+ (mmol/L)": "K+",
            "Na+ (mmol/L)": "Na+",
            "Ca++ (mmol/L)": "Ca++",
            "Cl- (mmol/L)": "Cl-",
            "Glu (mmol/L)": "Glucose",
            "Lac (mmol/L)": "Lactate",
            "p50(act) (mmHg)": "p50",
            "cBase(Ecf) (mmol/L)": "cBase",
        },
    )

    return datafr[editable_cols]


st.set_page_config(page_title="Uganda ABL Upload", layout="wide")

st.header("Import Uganda ABL files into RedCap")
st.write("This page keeps the Uganda CSV flow isolated from UCSF logic.")
st.markdown(
    """
    1. **Upload files**: Drag and drop the raw CSVs from the ABLs into the spot below.
    2. **Correct errors**: Fix any `Sample ID` or `Session ID` issues in the table below.
    3. **Resolve patient IDs**: The app looks up the final REDCap `patient_id` from `Session ID`.
    4. **Upload to RedCap**: Once the checks pass, upload the final file to RedCap.
    """
)

# abg_key = "Uganda_REDCAP_ABG"
# session_key = "Uganda_REDCAP_SESSION"
# api_url = "https://redcap.ace.ac.ug/api/"
abg_key = "Uganda_REDCAP_ABG_UCSF"
session_key = "Uganda_REDCAP_SESSION_UCSF"
api_url = "https://redcap.ucsf.edu/api/"

project = load_project(abg_key, api_url)

# Step 1: upload raw CSVs and normalize them into the columns we edit below.
st.subheader("Step 1: Upload files")

uploaded_files = st.file_uploader(
    "Choose CSV files",
    accept_multiple_files=True,
    key=state_key("file_uploader"),
)

if uploaded_files:
    st.session_state[state_key("uploaded")] = True
else:
    clear_page_state()
    st.session_state[state_key("uploaded")] = False
    st.write("Please upload a file")

if st.session_state[state_key("uploaded")] is True:
    if st.button("Combine CSVs", key=state_key("combine_button")):
        dfs = []
        current_filename = None
        try:
            for file in uploaded_files:
                current_filename = file.name
                serial_number = extract_serial_number(file.name)
                st.info(f"File: {file.name} -> Machine Serial: {serial_number}")

                df = pd.read_csv(
                    file,
                    encoding="cp1252",
                    converters={"Sample ID": str, "Accession number": str},
                )
                df = prepare_uploaded_dataframe(df, serial_number)
                dfs.append(df)
            df1 = pd.concat(dfs, ignore_index=True)
            st.session_state[state_key("combined")] = True
            st.session_state[state_key("df1")] = df1
        except ValueError as e:
            st.error(str(e))
            if current_filename:
                st.error(
                    "Problem filename: "
                    f"{current_filename}. Please use format 'PatLog - YYYY-MM-DD HH_MM_SS-SERIALNUMBER.csv'."
                )
            else:
                st.error(
                    "Please ensure all filenames follow the format: 'PatLog - YYYY-MM-DD HH_MM_SS-SERIALNUMBER.csv'."
                )
            st.stop()

if state_key("combined") in st.session_state:
    # Step 2: validate only the fields the Uganda workflow actually owns:
    # Sample ID from the CSV and Session ID from the accession-number field.
    st.subheader("Step 2: Correct errors")
    warning_container = (
        st.container()
    )  # container necessary because we write the UI out of linear order, a streamlit quirk.
    current_df = st.session_state[state_key("df1")].sort_values(by="Time Stamp")

    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        key=state_key("data_editor"),
    )

    st.session_state[state_key("errors")] = False
    with warning_container:
        sample_id_normalized = edited_df["Sample ID"].map(normalize_integer_string)
        invalid_sample_id_mask = sample_id_normalized.isna()
        if invalid_sample_id_mask.any():
            st.warning(
                "Sample ID is missing or not in the right format (a simple integer like '21'). "
                f"Please fix Sample ID in these rows:\n\n{edited_df.loc[invalid_sample_id_mask, 'Time Stamp'].tolist()}"
            )
            st.session_state[state_key("errors")] = True

        sample_id_series = edited_df["Sample ID"].astype("string").str.strip()
        if sum(sample_id_series == "0000") > 0:
            st.write("The row with Sample ID 0000 will be dropped")

        session_str = edited_df["Session ID"].astype("string").str.strip()
        invalid_session_mask = session_str.eq("") | session_str.isna()
        session_num = pd.to_numeric(session_str, errors="coerce")
        invalid_session_mask |= session_num.isna()
        if invalid_session_mask.any():
            st.warning(
                f"Session ID column has missing or invalid values: {edited_df.loc[invalid_session_mask, 'Time Stamp'].tolist()}"
            )
            st.session_state[state_key("errors")] = True

    st.session_state[state_key("edited_df")] = edited_df

if state_key("errors") not in st.session_state:
    st.write("")
elif st.session_state[state_key("errors")] is False:
    # Step 3: turn Session ID into the final REDCap patient_id, then confirm
    # we are not uploading a duplicate session/patient/machine combination.
    st.subheader("Step 3: Resolve patient IDs")
    st.write(
        "`Session ID` comes directly from the CSV accession-number field. If a value needs correction, edit it in Step 2."
    )

    if st.button("Resolve patient IDs and prepare file", key=state_key("resolve_button")):
        edited_df = st.session_state.get(state_key("edited_df"), st.session_state[state_key("df1")]).copy()
        edited_df["Date Calc"] = pd.to_datetime(edited_df["Date Calc"], errors="coerce")
        edited_df["_SampleIDStr"] = edited_df["Sample ID"].map(normalize_integer_string)
        edited_df["_SessionStr"] = edited_df["Session ID"].astype("string").str.strip().map(normalize_id)

        # Re-check the user-edited identifiers before touching REDCap.
        invalid_sample_id_mask = edited_df["_SampleIDStr"].isna()
        if invalid_sample_id_mask.any():
            st.write(
                "Sample ID column has missing or invalid values:",
                edited_df.loc[invalid_sample_id_mask, "Time Stamp"].tolist(),
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        invalid_session_mask = edited_df["_SessionStr"].isna()
        if invalid_session_mask.any():
            st.write(
                "Session ID column has missing or invalid values:",
                edited_df.loc[invalid_session_mask, "Time Stamp"].tolist(),
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        # REDCap SESSION is the source of truth for mapping session -> patient_id.
        session_proj = load_project(session_key, api_url)
        df_session = pd.DataFrame(session_proj.export_records())
        required_session_cols = {"record_id", "patient_id"}
        if df_session.empty or not required_session_cols.issubset(df_session.columns):
            st.error(
                "REDCap SESSION must contain record_id and patient_id values for Uganda uploads. "
                "We need that lookup to resolve the final patient_id from the session number in the accession field."
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        df_session["_record_str"] = df_session["record_id"].astype("string").str.strip().map(normalize_id)
        df_session["_patient_str"] = df_session["patient_id"].astype("string").str.strip().map(normalize_id)
        df_session["_session_date"] = pd.to_datetime(df_session["session_date"], errors="coerce").dt.date
        df_session = df_session[["_record_str", "_patient_str", "_session_date"]]
        df_session["_record_str"] = df_session["_record_str"].replace("", pd.NA)
        df_session["_patient_str"] = df_session["_patient_str"].replace("", pd.NA)

        edited_df = edited_df.merge(
            df_session.rename(columns={"_patient_str": "Resolved Patient ID"}),
            left_on="_SessionStr",
            right_on="_record_str",
            how="left",
        )

        # Every Session ID in the upload must resolve to a SESSION record.
        missing_sessions = edited_df[edited_df["Resolved Patient ID"].isna()]
        if not missing_sessions.empty:
            missing_summary = [
                f"Session {row['_SessionStr']}"
                for _, row in missing_sessions[["_SessionStr"]].drop_duplicates().iterrows()
            ]
            missing_lines = "\n".join([f"- {item} not found in REDCap SESSION" for item in missing_summary])
            st.error(
                "Some session numbers from the accession field do not exist in REDCap SESSION.\n\n"
                "Conflicts found:\n"
                f"{missing_lines}\n\n"
                "Please check and update."
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        # SESSION records also need a real patient_id before we can upload ABGs.
        edited_df["_ResolvedPatientIDStr"] = (
            edited_df["Resolved Patient ID"].astype("string").str.strip().map(normalize_id)
        )
        blank_patient_ids = edited_df["_ResolvedPatientIDStr"].isna()
        if blank_patient_ids.any():
            st.error(
                "Some session numbers matched REDCap SESSION records that are missing patient_id values. "
                "Please fix those SESSION records before uploading."
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        # Simple sanity check for mistyped Session IDs: the ABG file date should
        # match the session's session_date in REDCap SESSION.
        edited_df["_AbgDate"] = pd.to_datetime(edited_df["Date Calc"], errors="coerce").dt.date
        session_date_mismatches = edited_df[edited_df["_session_date"] != edited_df["_AbgDate"]]
        if not session_date_mismatches.empty:
            mismatch_summary = []
            mismatch_unique = session_date_mismatches[["_SessionStr", "_AbgDate", "_session_date"]].drop_duplicates()
            for _, row in mismatch_unique.iterrows():
                mismatch_summary.append(
                    f"Session {row['_SessionStr']} has ABG date {format_date_human(row['_AbgDate'])} but in RedCAP it is {format_date_human(row['_session_date'])}"
                )
            mismatch_lines = "\n".join([f"- {item}" for item in mismatch_summary])
            st.error(
                "Some ABG file dates do not match session_date in REDCap SESSION.\n\n"
                "Conflicts found:\n"
                f"{mismatch_lines}\n\n"
                "Please check and update."
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        # Final safety check: do not upload a session/patient_id/machine tuple
        # that already exists in REDCap ABG.
        df_abg = pd.DataFrame(project.export_records())
        required_abg_cols = {"session", "patient_id", "machine_serial"}
        if df_abg.empty or not required_abg_cols.issubset(df_abg.columns):
            st.info(
                "REDCap ABG has no existing session/patient_id/machine_serial records yet. "
                "Skipping duplicate Session/patient_id/machine check."
            )
        else:
            df_abg["_SessionStr"] = df_abg["session"].astype("string").str.strip().map(normalize_id)
            df_abg["_PatientIDStr"] = df_abg["patient_id"].astype("string").str.strip().map(normalize_id)
            df_abg["_MachineSerialStr"] = df_abg["machine_serial"].astype("string").str.strip()
            df_abg["_SessionStr"] = df_abg["_SessionStr"].replace("", pd.NA)
            df_abg["_PatientIDStr"] = df_abg["_PatientIDStr"].replace("", pd.NA)
            df_abg["_MachineSerialStr"] = df_abg["_MachineSerialStr"].replace("", pd.NA)
            abg_pairs = (
                df_abg[["_SessionStr", "_PatientIDStr", "_MachineSerialStr"]]
                .dropna(subset=["_SessionStr", "_PatientIDStr", "_MachineSerialStr"])
                .drop_duplicates()
            )

            incoming_pairs = edited_df.copy()
            incoming_pairs["_MachineSerialStr"] = incoming_pairs["machine_serial"].astype("string").str.strip()
            incoming_pairs["_MachineSerialStr"] = incoming_pairs["_MachineSerialStr"].replace("", pd.NA)
            incoming_pairs = (
                incoming_pairs[["Time Stamp", "_SessionStr", "_ResolvedPatientIDStr", "_MachineSerialStr"]]
                .dropna(subset=["_SessionStr", "_ResolvedPatientIDStr", "_MachineSerialStr"])
                .drop_duplicates()
                .rename(columns={"_ResolvedPatientIDStr": "_PatientIDStr"})
            )

            duplicate_rows = incoming_pairs.merge(
                abg_pairs,
                on=["_SessionStr", "_PatientIDStr", "_MachineSerialStr"],
                how="inner",
            ).sort_values(["_SessionStr", "_PatientIDStr", "_MachineSerialStr", "Time Stamp"])
            if not duplicate_rows.empty:
                duplicate_pairs = duplicate_rows[
                    ["_SessionStr", "_PatientIDStr", "_MachineSerialStr"]
                ].drop_duplicates()
                duplicate_items = [
                    f"Session {row['_SessionStr']} with patient_id {row['_PatientIDStr']} on machine {row['_MachineSerialStr']}"
                    for _, row in duplicate_pairs.iterrows()
                ]
                duplicate_lines = "\n".join([f"- {item}" for item in duplicate_items])
                st.error(
                    "These Session, patient_id, and machine serial combinations already exist in REDCap ABG.\n\n"
                    "Conflicts found:\n"
                    f"{duplicate_lines}\n\n"
                    "Please check and update."
                )
                st.session_state[state_key("errors")] = True
                st.session_state.pop(state_key("finaldf"), None)
                st.stop()

        # Build the final REDCap payload only after every validation passes.
        st.session_state[state_key("errors")] = False
        st.session_state[state_key("session_df")] = (
            edited_df[["Date Calc", "Session ID"]].drop_duplicates().reset_index(drop=True)
        )

        finaldf = edited_df.drop(columns=["_SessionStr", "_record_str"], errors="ignore").rename(
            columns={
                "Time Stamp": "time_stamp",
                "Date Calc": "date_calc",
                "Time Calc": "time_calc",
                "Resolved Patient ID": "patient_id",
                "Session ID": "session",
                "pH": "ph",
                "pCO2": "pco2",
                "pO2": "po2",
                "sO2": "so2",
                "COHb": "cohb",
                "MetHb": "methb",
                "tHb": "thb",
                "K+": "k",
                "Na+": "na",
                "Ca++": "ca",
                "Cl-": "cl",
                "Glucose": "glucose",
                "Lactate": "lactate",
                "p50": "p50",
                "cBase": "cbase",
            }
        )
        finaldf["sample"] = finaldf["_SampleIDStr"]
        finaldf = finaldf[
            [
                "time_stamp",
                "date_calc",
                "time_calc",
                "sample",
                "patient_id",
                "session",
                "ph",
                "pco2",
                "po2",
                "so2",
                "cohb",
                "methb",
                "thb",
                "k",
                "na",
                "ca",
                "cl",
                "glucose",
                "lactate",
                "p50",
                "cbase",
                "machine_serial",
            ]
        ]
        finaldf.rename_axis("record_id", inplace=True)
        st.session_state[state_key("finaldf")] = finaldf

# Step 4: show the final payload, then upload or download it.
if state_key("finaldf") in st.session_state and st.session_state.get(state_key("errors")) is False:
    st.subheader("Step 4: Upload & Download")
    st.write(st.session_state[state_key("finaldf")])
    one, two = st.columns(2)
    with one:
        st.write(
            "Estimated upload time: ",
            round(len(st.session_state[state_key("finaldf")]) * 0.1 / 60, 2),
            " minutes",
        )
        if st.button("Upload to RedCap", key=state_key("upload_button")):
            with st.spinner("Uploading to RedCap..."):
                r = project.import_records(
                    st.session_state[state_key("finaldf")],
                    import_format="df",
                    overwrite="normal",
                    force_auto_number=True,
                )
            st.write("Successfully uploaded ", r, " rows")
        st.download_button(
            "Download CSV",
            data=st.session_state[state_key("finaldf")].to_csv(index=False),
            file_name="ABL_upload_uganda.csv",
            mime="text/csv",
            key=state_key("download_button"),
        )
    with two:
        st.write("")
