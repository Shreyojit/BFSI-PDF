import time
from typing import List, Tuple, Dict, Any
from .aws_clients import s3_client, textract_client
from .config import get_settings

S = get_settings()


def upload_to_s3(local_bytes: bytes, filename: str) -> str:
    key = f"{S.S3_PREFIX}{filename}"
    s3 = s3_client()
    s3.put_object(Bucket=S.AWS_S3_BUCKET_NAME, Key=key, Body=local_bytes)
    return f"s3://{S.AWS_S3_BUCKET_NAME}/{key}"


def start_textract_analysis(s3_bucket: str, s3_key: str, features=("FORMS", "TABLES")) -> str:
    tx = textract_client()
    resp = tx.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
        FeatureTypes=list(features),
    )
    return resp["JobId"]


def wait_for_job(job_id: str, delay: int = 5, max_tries: int = 240):
    tx = textract_client()
    status = "IN_PROGRESS"
    tries = 0
    while tries < max_tries and status in ("IN_PROGRESS", "SUCCEEDED"):
        resp = tx.get_document_analysis(JobId=job_id, MaxResults=1000)
        status = resp.get("JobStatus", "IN_PROGRESS")
        if status == "SUCCEEDED":
            return
        if status == "FAILED":
            raise RuntimeError(f"Textract job failed: {resp.get('StatusMessage')}")
        time.sleep(delay)
        tries += 1
    raise TimeoutError("Textract job did not complete in allotted time.")


def fetch_all_results(job_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tx = textract_client()
    blocks: List[Dict[str, Any]] = []
    next_token = None
    last_resp: Dict[str, Any] = {}
    while True:
        if next_token:
            resp = tx.get_document_analysis(JobId=job_id, NextToken=next_token, MaxResults=1000)
        else:
            resp = tx.get_document_analysis(JobId=job_id, MaxResults=1000)
        last_resp = resp
        blocks.extend(resp.get("Blocks", []))
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return blocks, last_resp.get("DocumentMetadata", {})
