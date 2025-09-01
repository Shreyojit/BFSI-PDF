import boto3
from .config import get_settings

_settings = get_settings()


def s3_client():
    return boto3.client("s3", region_name=_settings.AWS_REGION)


def textract_client():
    return boto3.client("textract", region_name=_settings.AWS_REGION)


def bedrock_rt_client():
    return boto3.client("bedrock-runtime", region_name=_settings.AWS_REGION)
