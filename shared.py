import pandas as pd
import streamlit as st
from redcap import Project


def load_project(key, api_url):
    api_key = st.secrets[key]
    project = Project(api_url, api_key)
    return project


def format_date_human(date_value):
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return str(date_value)
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def normalize_id(value):
    if pd.isna(value):
        return pd.NA
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "nan":
        return pd.NA
    try:
        return str(int(float(value_str)))
    except (TypeError, ValueError):
        return value_str


def extract_serial_number(filename):
    """
    Extract the serial number from filename.
    Expected format: "PatLog - 2026-01-06 10_18_31_I393-092Rxxxxxxx31.csv"
    Serial number is expected to be between the last '-' and '.csv'
    """
    try:
        if not filename.endswith(".csv"):
            raise ValueError(f"File {filename} is not a CSV file")

        name_without_ext = filename[:-4]

        last_dash_index = name_without_ext.rfind("-")
        if last_dash_index == -1:
            raise ValueError(f"Filename format not expected: {filename}")

        serial_number = name_without_ext[last_dash_index + 1 :]
        if not serial_number or serial_number.strip() == "":
            raise ValueError(f"Serial number not found in filename: {filename}")

        return serial_number.strip()

    except Exception as e:
        raise ValueError(f"Error extracting serial number from {filename}: {str(e)}")
