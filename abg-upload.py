import numpy as np
import pandas as pd
import streamlit as st

from shared import extract_serial_number, format_date_human, load_project, normalize_id


PAGE_PREFIX = "ucsf"


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


def feinerize(datafr, serial_number):
    print("feinerizing")
    datafr["Time"] = pd.to_datetime(datafr["Time"])
    print("time converted")
    datafr["Date Calc"] = datafr["Time"].dt.date
    print("date calc")
    datafr["Time Calc"] = datafr["Time"].dt.time

    patient_id = datafr["Patient ID"].astype("string").str.strip()
    split_pid = patient_id.str.split(".", n=1, expand=True)
    if split_pid.shape[1] < 2:
        split_pid[1] = pd.NA

    subject_part = split_pid[0].astype("string").str.strip()
    sample_part = split_pid[1].astype("string").str.strip()
    has_dot = patient_id.str.contains(r"\.", regex=True, na=False)
    invalid_pid_mask = (
        patient_id.isna()
        | patient_id.eq("")
        | (~has_dot)
        | subject_part.isna()
        | subject_part.eq("")
        | sample_part.isna()
        | sample_part.eq("")
    )
    if invalid_pid_mask.any():
        bad_rows = datafr.loc[invalid_pid_mask, ["Time", "Patient ID"]].copy()
        bad_rows["Time"] = bad_rows["Time"].astype("string")
        bad_rows["Patient ID"] = bad_rows["Patient ID"].astype("string")
        problem_rows = "; ".join(
            [f"Time {row['Time']} with Patient ID '{row['Patient ID']}'" for _, row in bad_rows.iterrows()]
        )
        st.error(
            'Could not split Patient ID into "Subject" and "Sample". '
            f"Problem rows: {problem_rows}. Expected format like '3.21'. Please check and update."
        )
        st.stop()

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

    return datafr[allcols + ["machine_serial"]]


st.set_page_config(page_title="UCSF ABL Upload", layout="wide")

st.header("Import UCSF ABL files into RedCap")
st.write("This page keeps the UCSF upload flow separate from the Uganda uploader.")
st.markdown(
    """
    1. **Upload files**: Drag and drop the raw CSVs from the ABLs into the spot below.
    2. **Correct errors**: The app will alert you if there are missing values in the Subject, Sample, Patient ID, or UPI columns.
    3. **Add session numbers**: Every UPI should have a session number.
    4. **Upload to RedCap**: Once you have corrected all errors, you can upload the file to RedCap.
    """
)

abg_key = "REDCAP_ABG"
session_key = "REDCAP_SESSION"
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

                df = pd.read_csv(file, encoding="cp1252", converters={"Patient ID": str})
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
    st.session_state[state_key("errors")] = False
    if edited_df["Subject"].isnull().sum() > 0:
        st.write("Subject column has null values: ", edited_df[edited_df["Subject"].isnull()]["Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True
    if edited_df["Sample"].isnull().sum() > 0:
        st.write("Sample column has null values: ", edited_df[edited_df["Sample"].isnull()]["Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True
    if edited_df["Patient ID"].isnull().sum() > 0:
        st.write(
            "Patient ID column has null values: ", edited_df[edited_df["Patient ID"].isnull()]["Time Stamp"].tolist()
        )
        st.session_state[state_key("errors")] = True
    if edited_df["UPI"].isnull().sum() > 0:
        st.write("UPI column has null values: ", edited_df[edited_df["UPI"].isnull()]["Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True
    if sum(edited_df["Patient ID"] == "0000") > 0:
        st.write("The row with Patient ID 0000 will be dropped")
        edited_df = edited_df[edited_df["Patient ID"] != "0000"]

    upi_str = edited_df["UPI"].astype("string").str.strip()
    invalid_upi_mask = upi_str.eq("") | upi_str.isna()
    upi_num = pd.to_numeric(upi_str, errors="coerce")
    invalid_upi_mask |= upi_num.isna()
    if invalid_upi_mask.any():
        st.write("UPI column has missing or invalid values:", edited_df.loc[invalid_upi_mask, "Time Stamp"].tolist())
        st.session_state[state_key("errors")] = True

    _tmp = edited_df.copy()
    _tmp["Date Calc_norm"] = pd.to_datetime(_tmp["Date Calc"], errors="coerce").dt.date
    _tmp["Subject_norm"] = _tmp["Subject"].astype("string").str.strip()
    _tmp["UPI_norm"] = pd.to_numeric(_tmp["UPI"], errors="coerce").astype("Int64")
    _tmp_valid = _tmp.dropna(subset=["Date Calc_norm", "Subject_norm", "UPI_norm"])
    nuniq_per_group = _tmp_valid.groupby(["Subject_norm", "Date Calc_norm"])["UPI_norm"].nunique()
    err_group = nuniq_per_group[nuniq_per_group > 1]
    if not err_group.empty:
        conflict_rows = _tmp_valid.set_index(["Subject_norm", "Date Calc_norm"]).loc[err_group.index].reset_index()
        conflict_summary = []
        for (subject, date_calc), grp in conflict_rows.groupby(["Subject_norm", "Date Calc_norm"]):
            upis = sorted(grp["UPI_norm"].dropna().astype("Int64").astype("string").unique().tolist())
            conflict_summary.append(f"Subject {subject} on {format_date_human(date_calc)} listed with UPI {', '.join(upis)}")
        conflict_lines = "\n".join([f"- {item}" for item in conflict_summary])
        st.error(
            "Each Subject and Date pair must map to exactly one UPI.\n\n"
            "Conflicts found:\n"
            f"{conflict_lines}\n\n"
            "Please check and update."
        )
        st.session_state[state_key("errors")] = True

    st.session_state[state_key("edited_df")] = edited_df

if state_key("errors") not in st.session_state:
    st.write("")
elif st.session_state[state_key("errors")] is False:
    st.subheader("Step 3: Add session numbers")

    edited_df_for_sessions = st.session_state.get(state_key("edited_df"), st.session_state[state_key("df1")]).copy()
    edited_df_for_sessions["Date Calc"] = pd.to_datetime(edited_df_for_sessions["Date Calc"], errors="coerce")
    edited_df_for_sessions["UPI"] = pd.to_numeric(edited_df_for_sessions["UPI"], errors="coerce").astype("Int64")

    upi_df = (
        edited_df_for_sessions[["Date Calc", "UPI", "Time Stamp"]]
        .drop_duplicates(subset=["Date Calc", "UPI"])
        .sort_values(["Date Calc", "UPI"])
        .reset_index(drop=True)
    )
    upi_df["Session"] = np.nan
    upi_edits = st.data_editor(upi_df, num_rows="dynamic", key=state_key("upi_editor"))

    sess_str = upi_edits["Session"].astype("string").str.strip()
    upi_edits["Session"] = pd.to_numeric(sess_str, errors="coerce").astype("Int64")
    upi_edits["Date Calc"] = pd.to_datetime(upi_edits["Date Calc"], errors="coerce")
    upi_edits["UPI"] = pd.to_numeric(upi_edits["UPI"], errors="coerce").astype("Int64")

    blank_row_mask = upi_edits[["Date Calc", "UPI", "Session"]].isna().all(axis=1)
    if blank_row_mask.any():
        upi_edits = upi_edits.loc[~blank_row_mask].copy()

    if st.button("Add Session Numbers to file", key=state_key("session_button")):
        if upi_edits["Session"].isnull().sum() > 0:
            st.write("please fill in all session values")
            st.write(upi_edits[upi_edits["Session"].isnull()]["Time Stamp"].tolist())
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        ed = upi_edits.copy()
        ed["_SessionStr"] = ed["Session"].astype("Int64").astype("string").str.strip()
        ed["_UPIStr"] = ed["UPI"].astype("Int64").astype("string").str.strip()
        sess_to_upi = ed.dropna(subset=["_SessionStr", "_UPIStr"]).groupby("_SessionStr")["_UPIStr"].nunique()
        err_sess = sess_to_upi[sess_to_upi > 1]
        if not err_sess.empty:
            sess_upi_map = (
                ed.dropna(subset=["_SessionStr", "_UPIStr"])
                .groupby("_SessionStr")["_UPIStr"]
                .apply(lambda s: sorted(set(s.astype("string").tolist())))
            )
            conflict_items = [f"Session {s} listed with UPI {', '.join(sess_upi_map.loc[s])}" for s in err_sess.index]
            conflicting_lines = "\n".join([f"- {item}" for item in conflict_items])
            st.error(
                "Some Session values map to multiple UPIs.\n\n"
                "Conflicts found:\n"
                f"{conflicting_lines}\n\n"
                "Please check and update."
            )
            st.session_state[state_key("errors")] = True
            st.session_state.pop(state_key("finaldf"), None)
            st.stop()

        session_proj = load_project(session_key, api_url)
        df_session = pd.DataFrame(session_proj.export_records())
        required_session_cols = {"record_id", "patient_id"}
        if df_session.empty or not required_session_cols.issubset(df_session.columns):
            st.error(
                "REDCap SESSION must contain record_id and patient_id values before ABG upload. "
                "Please create the session record first, then try again."
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
            df_session = df_session.dropna(subset=["_record_str", "_patient_str"])
            check = ed[["Time Stamp", "_SessionStr", "_UPIStr"]].merge(
                df_session,
                left_on="_SessionStr",
                right_on="_record_str",
                how="left",
            )
            mismatches = check[check["_patient_str"].notna() & (check["_patient_str"] != check["_UPIStr"])]
            if not mismatches.empty:
                mismatch_summary = []
                mismatch_unique = mismatches[["_SessionStr", "_UPIStr", "_patient_str"]].drop_duplicates()
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
                st.error(f"{mismatch_header}\n\nConflicts found:\n{mismatch_lines}\n\nPlease check and update.")
                st.dataframe(
                    mismatches[["Time Stamp", "_SessionStr", "_UPIStr", "_patient_str"]].rename(
                        columns={
                            "_SessionStr": "Session",
                            "_UPIStr": "UPI (ABG file)",
                            "_patient_str": "UPI (REDCap SESSION)",
                        }
                    )
                )
                st.session_state[state_key("errors")] = True
                st.session_state.pop(state_key("finaldf"), None)
                st.stop()

        df_abg = pd.DataFrame(project.export_records())
        required_abg_cols = {"session", "patient_id", "machine_serial"}
        if df_abg.empty or not required_abg_cols.issubset(df_abg.columns):
            st.info(
                "REDCap ABG has no existing session/patient_id/machine_serial records yet. "
                "Skipping duplicate Session/UPI/machine check."
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
                ed[["Date Calc", "UPI", "Session"]],
                on=["Date Calc", "UPI"],
                how="left",
            )
            incoming_pairs["_SessionStr"] = incoming_pairs["Session"].astype("string").str.strip().map(normalize_id)
            incoming_pairs["_UPIStr"] = incoming_pairs["UPI"].astype("string").str.strip().map(normalize_id)
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
                st.session_state[state_key("errors")] = True
                st.session_state.pop(state_key("finaldf"), None)
                st.stop()

        st.session_state[state_key("errors")] = False
        st.session_state[state_key("upi_df")] = ed.reset_index()
        helper_merge_df = ed[["Date Calc", "UPI", "Session"]].copy()
        edited_df["Date Calc"] = pd.to_datetime(edited_df["Date Calc"], errors="coerce")
        edited_df["UPI"] = pd.to_numeric(edited_df["UPI"], errors="coerce").astype("Int64")
        st.session_state[state_key("finaldf")] = edited_df.merge(helper_merge_df, on=["Date Calc", "UPI"], how="left")
        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")][allcols_r]
        st.session_state[state_key("finaldf")].rename_axis("record_id", inplace=True)

        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")].drop(columns=["Subject"])
        st.session_state[state_key("finaldf")] = st.session_state[state_key("finaldf")].rename(
            columns={
                "record_id": "record_id",
                "Time Stamp": "time_stamp",
                "Date Calc": "date_calc",
                "Time Calc": "time_calc",
                "Sample": "sample",
                "Patient ID": "subject",
                "UPI": "patient_id",
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
            file_name="ABL_upload_ucsf.csv",
            mime="text/csv",
            key=state_key("download_button"),
        )
    with two:
        st.write("")
