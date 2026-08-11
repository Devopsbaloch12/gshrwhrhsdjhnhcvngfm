import os

# hf_xet's fast-transfer client can misreport "not enough disk space" on Windows even
# when there's plenty free (observed: 1.9GB free, 327MB file, still refused). Fall back
# to the plain HTTP downloader, which doesn't hit this.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
