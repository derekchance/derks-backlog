import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import sqlite3

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = '1XjK0d5AJ1FUbZo6fnWBcOr3EjCVYmISjyC1rB1Wrh8o'


def oauth_creds(func):
    def wrapper(*args, **kwargs):
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                  "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
        func(creds=creds, *args, **kwargs)
    return wrapper


@oauth_creds
def _update_values(df: pd.DataFrame, range, creds=None):
    try:
        service = build("sheets", "v4", credentials=creds)
        df = pd.concat([
            df,
            pd.DataFrame(columns=df.columns, index=df.index)
        ]).fillna('')
        values = df.values.tolist()
        body = {'values': values}
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=SPREADSHEET_ID,
                range=range,
                valueInputOption='USER_ENTERED',
                body=body,
            )
            .execute()
        )
        print(f"{result.get('updatedCells')} cells updated.")
        return result
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error


def update_backlog_values():
    with sqlite3.connect('games.db') as con:
        with open('backlog.sql') as f:
            query = f.read()
        df = pd.read_sql(query, con).fillna('')
    _update_values(df=df, range='Backlog!A2:I10000')


def update_log_values():
    with sqlite3.connect('games.db') as con:
        with open('log.sql') as f:
            query = f.read()
        df = pd.read_sql(query, con).fillna('')
    _update_values(df=df, range='Log!A2:I10000')