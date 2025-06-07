import os
import boto3
import asyncio
import urllib.parse
from pathlib import Path
from botocore.exceptions import ClientError, EndpointConnectionError

from multimodal_rag.asset_store.types import AssetStore
from multimodal_rag.config.schema import S3AssetConfig
from multimodal_rag.document import MetaConfig
from multimodal_rag.utils.retry import backoff
from multimodal_rag.log_config import logger


class S3AssetStore(AssetStore):
    def __init__(self, cfg: S3AssetConfig):
        self.config = cfg

        profile = os.getenv("AWS_PROFILE")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if profile:
            session = boto3.Session(profile_name=profile)
        elif access_key and secret_key:
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.config.region,
            )
        else:
            session = boto3.Session(region_name=self.config.region)

        client_args = {
            "region_name": self.config.region,
            "use_ssl": True,
        }

        if self.config.endpoint_url:
            client_args["endpoint_url"] = self.config.endpoint_url
            client_args["use_ssl"] = self.config.endpoint_url.startswith("https")

        self.client = session.client("s3", **client_args)

    @backoff(exception=(ClientError, EndpointConnectionError))
    async def ensure_storage(self, project_id: str) -> None:
        try:
            await asyncio.to_thread(self.client.head_bucket, Bucket=self.config.bucket)
            logger.info(f"Bucket '{self.config.bucket}' already exists.")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                kwargs = {"Bucket": self.config.bucket}
                # Only add LocationConstraint for AWS S3 (not MinIO) and non-us-east-1 regions
                if self.config.endpoint_url is None and self.config.region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.config.region
                    }
                await asyncio.to_thread(self.client.create_bucket, **kwargs)
                logger.info(f"Bucket '{self.config.bucket}' created successfully.")
            else:
                logger.error(f"Error checking/creating bucket: {e}")
                raise

    @backoff(exception=(ClientError, EndpointConnectionError))
    async def store(
        self,
        project_id: str,
        tmp_path: Path,
        meta: MetaConfig
    ) -> str:
        short_fp = meta.fingerprint[:16]
        name = Path(meta.filename or tmp_path.name).stem
        ext = Path(meta.filename or tmp_path.name).suffix
        object_key = f"{project_id}/{name}_{short_fp}{ext}"

        if not self.config.overwrite:
            try:
                await asyncio.to_thread(
                    self.client.head_object,
                    Bucket=self.config.bucket,
                    Key=object_key
                )
                raise FileExistsError(f"s3://{self.config.bucket}/{object_key} already exists and overwrite=False")
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise

        await asyncio.to_thread(
            self.client.upload_file,
            str(tmp_path),
            self.config.bucket,
            object_key,
            ExtraArgs={"ContentType": meta.mime or "application/octet-stream"}
        )

        return f"s3://{self.config.bucket}/{object_key}"

    @backoff(exception=(ClientError, EndpointConnectionError))
    async def read(self, uri: str) -> bytes:
        parsed = urllib.parse.urlparse(uri)
        key = parsed.path.lstrip("/")

        try:
            obj = await asyncio.to_thread(
                self.client.get_object,
                Bucket=self.config.bucket,
                Key=key
            )
            return await asyncio.to_thread(obj["Body"].read)
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 object not found: {self.config.bucket}/{key}")
