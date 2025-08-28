import os
import base64
import glob
from urllib.parse import urlparse

from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.models.tool_models import ToolBaseModel
from vss_ctx_rag.utils.globals import DEFAULT_NUM_FRAMES_PER_CHUNK


@register_tool_config("image")
class ImageFetcherConfig(ToolBaseModel):
    """Image Fetcher configuration."""

    minio_host: str
    minio_port: str
    minio_username: str
    minio_password: str


@register_tool(config=ImageFetcherConfig)
class ImageFetcher(Tool):
    """Image Fetcher. Save / retrieve images for a file/chunk."""

    def __init__(self, name="image_fetcher", config=None, tools=None) -> None:
        super().__init__(name, config, tools)
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        self.minio_client = None
        self._init_minio_client()

    def _init_minio_client(self):
        """Initialize Minio client from environment variables if available."""
        # Retrieve Minio connection details
        minio_host = self.config.params.minio_host
        minio_port = self.config.params.minio_port
        minio_username = self.config.params.minio_username
        minio_password = self.config.params.minio_password

        if not (minio_host and minio_port and minio_username and minio_password):
            logger.warning(
                "Minio environment variables not set, some functionality may be limited."
            )
            return

        # Only import minio if needed
        try:
            from minio import Minio

            minio_uri = f"http://{minio_host}:{minio_port}"
            parsed_uri = urlparse(minio_uri)
            secure = parsed_uri.scheme == "https"
            endpoint = parsed_uri.netloc or parsed_uri.path

            self.minio_client = Minio(
                endpoint,
                access_key=minio_username,
                secret_key=minio_password,
                secure=secure,
            )
            logger.debug("Minio client initialized successfully")
        except ImportError:
            logger.warning(
                "Minio package not installed. Minio functionality will be disabled."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Minio client: {e}")
            self.minio_client = None

    def _get_image_base64_minio(
        self, asset_dir: str, num_frames_per_chunk: int = DEFAULT_NUM_FRAMES_PER_CHUNK
    ) -> list[str]:
        """Get base64 encoded strings for all JPEG images from the chunk's asset directory in Minio.

        Args:
            asset_dir: The asset directory of the chunk.
            num_frames_per_chunk: The number of frames per chunk.
        Returns:
            A list of base64 encoded image strings, or an empty list if errors occur
            or no JPEGs are found.
        """
        if not self.minio_client:
            logger.warning("Minio client not initialized, cannot retrieve images.")
            return []

        if not asset_dir:
            logger.warning(
                "asset_dir not set in chunk, cannot determine image path in Minio."
            )
            return []

        # Split the asset_dir into bucket and prefix
        # Assuming format: bucket_name/path/to/assets
        parts = asset_dir.split("/", 1)
        if len(parts) < 2:
            logger.error(
                f"Invalid asset_dir format: {asset_dir}. Expected 'bucket/path/to/assets'"
            )
            return []

        bucket_name = parts[0]
        prefix = parts[1]

        # Ensure prefix acts as a directory prefix
        if not prefix.endswith("/"):
            prefix += "/"

        # Verify bucket exists
        try:
            if not self.minio_client.bucket_exists(bucket_name):
                logger.error(f"Bucket {bucket_name} does not exist")
                return []
        except Exception as e:
            logger.error(f"Failed to check if bucket {bucket_name} exists: {e}")
            return []

        base64_images = []
        try:
            logger.debug(
                f"Listing objects in bucket '{bucket_name}' with prefix '{prefix}'"
            )
            objects = self.minio_client.list_objects(
                bucket_name, prefix=prefix, recursive=True
            )

            # Collect all JPEG objects first
            jpeg_objects = []
            for obj in objects:
                object_name = obj.object_name
                if object_name.lower().endswith((".jpg", ".jpeg")):
                    logger.debug(f"Found JPEG image object: {object_name}")
                    jpeg_objects.append(obj)

            # Sort objects numerically by frame number
            def extract_frame_number(object_name):
                """Extract frame number from object name for numerical sorting."""
                import re

                # Look for pattern like frame_123.jpg or frame_123.jpeg
                match = re.search(
                    r"frame_(\d)\.(jpg|jpeg)$", object_name, re.IGNORECASE
                )
                if match:
                    return int(match.group(1))
                # If no frame number found, return a large number to put it at the end
                return 999999

            # Sort by frame number
            jpeg_objects.sort(key=lambda obj: extract_frame_number(obj.object_name))

            total_frames = len(jpeg_objects)
            logger.debug(f"Sorted {total_frames} JPEG objects by frame number")

            # Uniformly sample num_frames_per_chunk from jpeg_objects
            if (
                total_frames > 0
                and num_frames_per_chunk > 0
                and total_frames > num_frames_per_chunk
            ):
                # Uniformly spaced indices
                if num_frames_per_chunk == 1:
                    indices = [total_frames // 2]
                else:
                    indices = [
                        int(round(i * (total_frames - 1) / (num_frames_per_chunk - 1)))
                        for i in range(num_frames_per_chunk)
                    ]
                jpeg_objects = [jpeg_objects[i] for i in indices]
                logger.debug(
                    f"Uniformly sampled {num_frames_per_chunk} frames from {total_frames} total frames"
                )

            for obj in jpeg_objects:
                object_name = obj.object_name
                response = None
                try:
                    response = self.minio_client.get_object(bucket_name, object_name)
                    image_bytes = response.read()
                    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
                    base64_images.append(encoded_string)
                    logger.debug(f"Processed frame: {object_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to download/process object {object_name} from Minio: {e}"
                    )
                finally:
                    if response:
                        response.close()
                        response.release_conn()

            if not base64_images:
                logger.warning(
                    f"No JPEG files found or processed in Minio bucket '{bucket_name}' with prefix '{prefix}'"
                )
        except Exception as e:
            logger.error(
                f"Failed to list objects in Minio bucket '{bucket_name}' with prefix '{prefix}': {e}"
            )
            return []

        return base64_images

    def _get_image_base64(
        self, asset_dir: str, num_frames_per_chunk: int = DEFAULT_NUM_FRAMES_PER_CHUNK
    ) -> list[str]:
        """Get base64 encoded strings for all JPEG images in the chunk's asset directory.

        Args:
            chunk: ChunkInfo object containing the asset directory path.
            num_frames_per_chunk: The number of frames per chunk.
        Returns:
            A list of base64 encoded image strings, or an empty list if errors occur
            or no JPEGs are found.
        """
        base64_images = []

        if not asset_dir or not os.path.isdir(asset_dir):
            logger.error(f"Invalid or non-existent asset directory: {asset_dir}")
            return []

        # Find all jpeg files (case-insensitive)
        jpeg_files = glob.glob(os.path.join(asset_dir, "*.jpg")) + glob.glob(
            os.path.join(asset_dir, "*.jpeg")
        )

        if not jpeg_files:
            logger.warning(f"No JPEG files found in directory: {asset_dir}")
            return []

        # Sort the files by name to ensure consistent order
        def extract_frame_number(filepath):
            """Safely extract frame number with error handling."""
            try:
                basename = os.path.basename(filepath)
                # Try pattern: frame_123.jpg
                parts = basename.split("_")
                if len(parts) >= 2:
                    frame_part = parts[1].split(".")[0]
                    return int(frame_part)
            except (ValueError, IndexError):
                logger.warning(f"Could not extract frame number from: {filepath}")
            return -1

        jpeg_files.sort(key=extract_frame_number)

        # Take num_frames_per_chunk files uniformly spaced
        if (
            len(jpeg_files) > 0
            and num_frames_per_chunk > 0
            and len(jpeg_files) > num_frames_per_chunk
        ):
            # Uniformly spaced indices
            if num_frames_per_chunk == 1:
                indices = [len(jpeg_files) // 2]
            else:
                # We use this logic to select `num_frames_per_chunk` frames uniformly from the available JPEG files.
                # This ensures that the sampled frames are spread as evenly as possible across the entire set,
                # which is important for downstream tasks (e.g., VLM retrieval) to get a representative view of the chunk.
                # The formula below calculates the index for each sample by dividing the range [0, len(jpeg_files)-1]
                # into (num_frames_per_chunk-1) intervals, and rounding to the nearest integer.
                indices = [
                    int(round(i * (len(jpeg_files) - 1) / (num_frames_per_chunk - 1)))
                    for i in range(num_frames_per_chunk)
                ]
            jpeg_files = [jpeg_files[i] for i in indices]
            logger.debug(
                f"Uniformly sampled {num_frames_per_chunk} frames from {len(jpeg_files)} total frames"
            )

        for img_path in jpeg_files:
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    base64_images.append(encoded_string)
            except Exception as e:
                logger.error(f"Error processing image file {img_path}: {e}")

        return base64_images

    def get_image_base64(
        self, asset_dir: str, num_frames_per_chunk: int = DEFAULT_NUM_FRAMES_PER_CHUNK
    ) -> list[str]:
        """Get base64 encoded strings for all JPEG images in the chunk's asset directory.

        Args:
            asset_dir: The asset directory of the chunk.
            num_frames_per_chunk: The number of frames per chunk.
        Returns:
            A list of base64 encoded image strings, or an empty list if errors occur
            or no JPEGs are found.
        """
        if not asset_dir:
            logger.warning("No asset directory found in chunk.")
            return []
        if asset_dir.startswith("minio://"):
            return self._get_image_base64_minio(
                asset_dir.split("minio://")[1], num_frames_per_chunk
            )
        else:
            return self._get_image_base64(asset_dir, num_frames_per_chunk)
