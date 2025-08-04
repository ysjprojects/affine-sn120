import time
import base64
import json
from io import BytesIO
from minio import Minio

# === STEP 0: User Setup ===
seed_phrase = "race hungry company town transfer review horn base flip joke hour moral"  # Replace with your sub-account seed phrase
access_key = base64.b64encode(seed_phrase.encode("utf-8")).decode("utf-8")
bucket_name = f"demo-public-{int(time.time())}"
object_name = "hello.txt"
file_content = b"Hello, Hippius! This file is now on IPFS."

# === STEP 1: Create Minio Client ===
client = Minio(
    "s3.hippius.com",
    access_key=access_key,
    secret_key=seed_phrase,
    secure=True,
    region="decentralized"
)

# === STEP 2: Create Empty Bucket ===
client.make_bucket(bucket_name)
print(f"✓ Created empty bucket: {bucket_name}")

# === STEP 3: Apply Public Read Policy ===
public_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": "*",
        "Action": ["s3:GetObject"],
        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
    }]
}
client.set_bucket_policy(bucket_name, json.dumps(public_policy))
print(f"✓ Bucket '{bucket_name}' made public")

# === STEP 4: Upload File to Bucket ===
client.put_object(
    bucket_name,
    object_name,
    BytesIO(file_content),
    length=len(file_content),
    content_type="text/plain"
)
print(f"✓ Uploaded file: {object_name}")

# === STEP 5: Retrieve IPFS CID ===
stat = client.stat_object(bucket_name, object_name)
cid = stat.etag.strip('"')
print(f"✓ IPFS CID: {cid}")

# === STEP 6: Output Public IPFS URLs ===
print("\n📂 Public IPFS Access URLs:")
print(f"- Hippius:  https://get.hippius.network/ipfs/{cid}")
print(f"- IPFS.io:  https://ipfs.io/ipfs/{cid}")
print(f"- Pinata:   https://gateway.pinata.cloud/ipfs/{cid}")
print(f"- Cloudflare: https://cloudflare-ipfs.com/ipfs/{cid}")
