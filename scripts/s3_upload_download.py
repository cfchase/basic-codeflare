import ray
import os
import boto3
import botocore
import json
from time import strftime, gmtime



@ray.remote
def s3_upload_read_delete():
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)


    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    def create_json_object(file_key, content):
        json_str = json.dumps(content)
        bucket.put_object(Body=bytes(json_str.encode("UTF-8")), Key=file_key)

    def read_json_object(file_key):
        s3_object = s3_resource.Object(bucket_name, file_key)
        file_content = s3_object.get()['Body'].read().decode('utf-8')
        json_content = json.loads(file_content)
        return json_content

    def delete_object(file_key):
        s3object = s3_resource.Object(bucket_name, file_key)
        s3object.delete()

    def list_objects(prefix):
        filter = bucket.objects.filter(Prefix=prefix)
        keys = [obj.key for obj in filter.all()]

        return keys

    file_key = f"tmp/time.json"

    create_json_object(file_key, {"time": strftime("%Y-%m-%d %H:%M:%S", gmtime())})
    time = read_json_object(file_key)
    print(time)
    print(list_objects(""))

    delete_object(file_key)
    print(list_objects(""))

    return time


# Automatically connect to the running Ray cluster.
ray.init()
time = ray.get(s3_upload_read_delete.remote())
print(time)
