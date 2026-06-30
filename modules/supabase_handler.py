import os
from supabase import create_client

def get_supabase():
    # Creates Supabase client using credentials from .env
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def upload_csv(uid, dataset_id, file_bytes):
    # Uploads CSV bytes to Supabase Storage
    # Path: {uid}/{dataset_id}.csv — unique per user per dataset
    supabase = get_supabase()
    path = f"{uid}/{dataset_id}.csv"

    supabase.storage.from_("csvfiles").upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": "text/csv","upsert": "true"}
    )
    return path

def download_csv(uid, dataset_id):
    # Downloads CSV bytes from Supabase Storage
    # Returns bytes — Flask will wrap in BytesIO for pandas
    supabase = get_supabase()
    path = f"{uid}/{dataset_id}.csv"
    response = supabase.storage.from_("csvfiles").download(path)
    return response

def delete_csv(uid, dataset_id):
    # Deletes CSV from Supabase Storage
    # Called if we ever need cleanup
    supabase = get_supabase()
    path = f"{uid}/{dataset_id}.csv"
    supabase.storage.from_("csvfiles").remove([path])