import numpy as np
import pandas as pd
import streamlit as st

from shared import extract_serial_number, load_project, normalize_id


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
        state_key("upi_df"),
        state_key("finaldf"),
    ]:
        st.session_state.pop(key, None)


fcols = [
    "Time Stamp",
    "Date Calc",
    "Time Calc",
    "Subject",
    "Sample",
    "Patient ID",
    "UPI",
    "pH",
    "pCO2",
    "pO2",
    "sO2",
    "COHb",
    "MetHb",
    "tHb",
]

rcols = [
    "Subject",
    "Time Stamp",
    "Date Calc",
    "Time Calc",
    "Sample",
    "Patient ID",
    "UPI",
    "Session",
    "pH",
    "pCO2",
    "pO2",
    "sO2",
    "COHb",
    "MetHb",
    "tHb",
    "machine_serial",
]

allcols = fcols + ["K+", "Na+", "Ca++", "Cl-", "Glucose", "Lactate", "p50", "cBase"]
allcols_r = rcols + ["K+", "Na+", "Ca++", "Cl-", "Glucose", "Lactate", "p50", "cBase"]


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


def split_sample_id_columns(sample_id_series):
    sample_id = sample_id_series.astype("string").str.strip()
    sample_part = sample_id.map(normalize_integer_string).astype("string").str.strip()
    invalid_sample_id_mask = sample_id.isna() | sample_id.eq("") | sample_part.isna() | sample_part.eq("")
    subject_part = sample_part.mask(invalid_sample_id_mask, pd.NA)
    sample_part = sample_part.mask(invalid_sample_id_mask, pd.NA)
    return subject_part, sample_part, invalid_sample_id_mask


def feinerize(datafr, serial_number):
    print("feinerizing")
    datafr["Time"] = pd.to_datetime(datafr["Time"])
    print("time converted")
    datafr["Date Calc"] = datafr["Time"].dt.date
    print("date calc")
    datafr["Time Calc"] = datafr["Time"].dt.time

    if "Sample ID" not in datafr.columns:
        raise ValueError("Missing required column 'Sample ID' for Uganda files")
    if "Patient ID" not in datafr.columns:
        datafr["Patient ID"] = pd.NA

    subject_part, sample_part, invalid_sample_id_mask = split_sample_id_columns(datafr["Sample ID"])
    if invalid_sample_id_mask.any():
        bad_rows = datafr.loc[invalid_sample_id_mask, ["Time", "Sample ID"]].copy()
        bad_rows["Time"] = bad_rows["Time"].astype("string")
        bad_rows["Sample ID"] = bad_rows["Sample ID"].astype("string")
        problem_rows = "; ".join(
            [f"Time {row['Time']} with Sample ID '{row['Sample ID']}'" for _, row in bad_rows.iterrows()]
        )
        st.warning(
            'Some Sample ID values could not be split into "Subject" and "Sample".\n\n'
            f"Problem rows:\n\n{problem_rows}. Expected a simple integer like '21'.\n\n"
            "Those rows will stay editable in Step 2 so you can correct them."
        )

    datafr["Subject"] = subject_part
    datafr["Sample"] = sample_part
    datafr["machine_serial"] = serial_number

    datafr = datafr.rename(
        columns={
            "Time": "Time Stamp",
            "pCO2 (mmHg)": "pCO2",
            "pO2 (mmHg)": "pO2",
            "sO2 (%)": "sO2",
            "COHb (%)": "COHb",
            "MetHb (%)": "MetHb",
            "tHb (g/dL)": "tHb",
            "Accession number": "UPI",
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

    return datafr[["Sample ID"] + allcols + ["machine_serial"]]


st.set_page_config(page_title="Uganda ABL Upload", layout="wide")

st.header("Import Uganda ABL files into RedCap")
st.write("This page keeps the Uganda CSV flow isolated from UCSF logic.")
st.markdown(
    """
    1. **Upload files**: Drag and drop the raw CSVs from the ABLs into the spot below.
    2. **Correct errors**: Fix any Sample ID or accession-number issues in the table below.
    3. **Confirm session numbers**: The accession number is used as the starting session value and can be corrected before upload.
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
                    converters={"Patient ID": str, "Sample ID": str, "Accession number": str},
                )
                df = feinerize(df, serial_number)
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
    st.subheader("Step 2: Correct errors")
    edited_df = st.data_editor(
        st.session_state[state_key("df1")].sort_values(by="Time Stamp"),
        num_rows="dynamic",
        key=state_key("data_editor"),
    )
    edited_df["Subject"], edited_df["Sample"], invalid_sample_id_mask = split_sample_id_columns(edited_df["Sample ID"])
    st.session_state[state_key("errors")] = False

    subject_null_mask = edited_df["Subject"].isnull() & ~invalid_sample_id_mask
    sample_null_mask = edited_df["Sample"].isnull() & ~invalid_sample_id_mask
    if subject_null_mask.any():
        st.write("Subject column has null values: ", edited_df.loc[subject_null_mask, "Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True
    if sample_null_mask.any():
        st.write("Sample column has null values: ", edited_df.loc[sample_null_mask, "Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True
    if invalid_sample_id_mask.any():
        st.write(
            "Sample ID is missing or not in the right format (a simple integer like '21'). "
            "Please fix Sample ID in these rows. Subject and Sample are filled automatically from Sample ID: ",
            edited_df.loc[invalid_sample_id_mask, "Time Stamp"].tolist(),
        )
        st.session_state[state_key("errors")] = True

    sample_id_series = edited_df["Sample ID"].astype("string").str.strip()
    if sum(sample_id_series == "0000") > 0:
        st.write("The row with Sample ID 0000 will be dropped")
        edited_df = edited_df[sample_id_series != "0000"]

    accession_str = edited_df["UPI"].astype("string").str.strip()
    invalid_accession_mask = accession_str.eq("") | accession_str.isna()
    accession_num = pd.to_numeric(accession_str, errors="coerce")
    invalid_accession_mask |= accession_num.isna()
    if invalid_accession_mask.any():
        st.write(
            "Accession number column has missing or invalid values:",
            edited_df.loc[invalid_accession_mask, "Time Stamp"].tolist(),
        )
        st.session_state[state_key("errors")] = True

    st.session_state[state_key("edited_df")] = edited_df

if state_key("errors") not in st.session_state:
    st.write("")
elif st.session_state[state_key("errors")] is False:
    st.subheader("Step 3: Confirm session numbers")

    edited_df_for_sessions = st.session_state.get(state_key("edited_df"), st.session_state[state_key("df1")]).copy()
    edited_df_for_sessions["Date Calc"] = pd.to_datetime(edited_df_for_sessions["Date Calc"], errors="coerce")
    edited_df_for_sessions["UPI"] = pd.to_numeric(edited_df_for_sessions["UPI"], errors="coerce").astype("Int64")

    upi_df = (
        edited_df_for_sessions[["Date Calc", "UPI", "Time Stamp"]]
        .drop_duplicates(subset=["Date Calc", "UPI"])
        .sort_values(["Date Calc", "UPI"])
        .reset_index(drop=True)
    )
    upi_df["Session"] = upi_df["UPI"].astype("Int64")
    upi_edits = st.data_editor(upi_df, num_rows="dynamic", key=state_key("upi_editor"))

    sess_str = upi_edits["Session"].astype("string").str.strip()
    upi_edits["Session"] = pd.to_numeric(sess_str, errors="coerce").astype("Int64")
    upi_edits["Date Calc"] = pd.to_datetime(upi_edits["Date Calc"], errors="coerce")
    upi_edits["UPI"] = pd.to_numeric(upi_edits["UPI"], errors="coerce").astype("Int64")

    blank_row_mask = upi_edits[["Date Calc", "UPI", "Session"]].isna().all(axis=1)
    if blank_row_mask.any():
        upi_edits = upi_edits.loc[~blank_row_mask].copy()

    if st.button("Resolve patient IDs and prepare file", key=state_key("session_button")):
        if upi_edits["Session"].isnull().sum() > 0:
            st.write("please fill in all session values")
            st.write(upi_edits[upi_edits["Session"].isnull()]["Time Stamp"].tolist())
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        ed = upi_edits.copy()
        ed["_SessionStr"] = ed["Session"].astype("Int64").astype("string").str.strip()

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
        else:
            df_session["_record_str"] = df_session["record_id"].astype("string").str.strip()
            df_session["_patient_str"] = df_session["patient_id"].astype("string").str.strip()
            df_session = df_session[["_record_str", "_patient_str"]]
            df_session["_record_str"] = df_session["_record_str"].replace("", pd.NA)
            df_session["_patient_str"] = df_session["_patient_str"].replace("", pd.NA)

            check = ed[["Time Stamp", "_SessionStr"]].merge(
                df_session,
                left_on="_SessionStr",
                right_on="_record_str",
                how="left",
            )
            missing_sessions = check[check["_patient_str"].isna()]
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

            ed = ed.merge(
                df_session.rename(columns={"_patient_str": "Resolved Patient ID"}),
                left_on="_SessionStr",
                right_on="_record_str",
                how="left",
            )
            ed["Resolved Patient ID"] = ed["Resolved Patient ID"].astype("string").str.strip()
            blank_patient_ids = ed["Resolved Patient ID"].isna() | ed["Resolved Patient ID"].eq("")
            if blank_patient_ids.any():
                st.error(
                    "Some session numbers matched REDCap SESSION records that are missing patient_id values. "
                    "Please fix those SESSION records before uploading."
                )
                st.session_state[state_key("errors")] = True
                st.session_state.pop(state_key("finaldf"), None)
                st.stop()

        df_abg = pd.DataFrame(project.export_records())
        required_abg_cols = {"session", "patient_id", "machine_serial"}
        if df_abg.empty or not required_abg_cols.issubset(df_abg.columns):
            st.info(
                "REDCap ABG has no existing session/patient_id/machine_serial records yet. "
                "Skipping duplicate Session/patient_id/machine check."
            )
        else:
            df_abg["_SessionStr"] = df_abg["session"].astype("string").str.strip().map(normalize_id)
            df_abg["_UPIStr"] = df_abg["patient_id"].astype("string").str.strip().map(normalize_id)
            df_abg["_MachineSerialStr"] = df_abg["machine_serial"].astype("string").str.strip()
            df_abg["_SessionStr"] = df_abg["_SessionStr"].replace("", pd.NA)
            df_abg["_UPIStr"] = df_abg["_UPIStr"].replace("", pd.NA)
            df_abg["_MachineSerialStr"] = df_abg["_MachineSerialStr"].replace("", pd.NA)
            abg_pairs = (
                df_abg[["_SessionStr", "_UPIStr", "_MachineSerialStr"]]
                .dropna(subset=["_SessionStr", "_UPIStr", "_MachineSerialStr"])
                .drop_duplicates()
            )

            edited_df_for_upload = st.session_state.get(state_key("edited_df"), st.session_state[state_key("df1")]).copy()
            edited_df_for_upload["Date Calc"] = pd.to_datetime(edited_df_for_upload["Date Calc"], errors="coerce")
            edited_df_for_upload["UPI"] = pd.to_numeric(edited_df_for_upload["UPI"], errors="coerce").astype("Int64")

            incoming_pairs = edited_df_for_upload[["Time Stamp", "Date Calc", "UPI", "machine_serial"]].merge(
                ed[["Date Calc", "UPI", "Session", "Resolved Patient ID"]],
                on=["Date Calc", "UPI"],
                how="left",
            )
            incoming_pairs["_SessionStr"] = incoming_pairs["Session"].astype("string").str.strip().map(normalize_id)
            incoming_pairs["_UPIStr"] = incoming_pairs["Resolved Patient ID"].astype("string").str.strip().map(normalize_id)
            incoming_pairs["_MachineSerialStr"] = incoming_pairs["machine_serial"].astype("string").str.strip()
            incoming_pairs["_SessionStr"] = incoming_pairs["_SessionStr"].replace("", pd.NA)
            incoming_pairs["_UPIStr"] = incoming_pairs["_UPIStr"].replace("", pd.NA)
            incoming_pairs["_MachineSerialStr"] = incoming_pairs["_MachineSerialStr"].replace("", pd.NA)
            incoming_pairs = (
                incoming_pairs[["Time Stamp", "_SessionStr", "_UPIStr", "_MachineSerialStr"]]
                .dropna(subset=["_SessionStr", "_UPIStr", "_MachineSerialStr"])
                .drop_duplicates()
            )

            duplicate_rows = incoming_pairs.merge(
                abg_pairs,
                on=["_SessionStr", "_UPIStr", "_MachineSerialStr"],
                how="inner",
            ).sort_values(["_SessionStr", "_UPIStr", "_MachineSerialStr", "Time Stamp"])
            if not duplicate_rows.empty:
                duplicate_pairs = duplicate_rows[["_SessionStr", "_UPIStr", "_MachineSerialStr"]].drop_duplicates()
                duplicate_items = [
                    f"Session {row['_SessionStr']} with patient_id {row['_UPIStr']} on machine {row['_MachineSerialStr']}"
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

        st.session_state[state_key("errors")] = False
        st.session_state[state_key("upi_df")] = ed.reset_index()
        helper_merge_df = ed[["Date Calc", "UPI", "Session", "Resolved Patient ID"]].copy()
        edited_df["Date Calc"] = pd.to_datetime(edited_df["Date Calc"], errors="coerce")
        edited_df["UPI"] = pd.to_numeric(edited_df["UPI"], errors="coerce").astype("Int64")
        st.session_state[state_key("finaldf")] = edited_df.merge(helper_merge_df, on=["Date Calc", "UPI"], how="left")
        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")][allcols_r + ["Resolved Patient ID"]]
        st.session_state[state_key("finaldf")].rename_axis("record_id", inplace=True)

        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")].drop(columns=["Patient ID", "UPI"])
        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")].rename(
            columns={
                "record_id": "record_id",
                "Subject": "subject",
                "Time Stamp": "time_stamp",
                "Date Calc": "date_calc",
                "Time Calc": "time_calc",
                "Sample": "sample",
                "Resolved Patient ID": "patient_id",
                "Session": "session",
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
        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")][
            [
                "subject",
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
