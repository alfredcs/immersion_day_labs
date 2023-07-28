import boto3

s3 = boto3.resource('s3')
ecr_client = boto3.client('ecr')

def delete_endpoint(sm_client, endpoint_name, endpoint_conf_name):
    try:
        response = sm_client.delete_endpoint_config(
            EndpointConfigName=endpoint_conf_name
        )
        print(response)
    except:
        print(f"deleting {endpoint_conf_name} failed")

    try:
        response = sm_client.delete_endpoint(
            EndpointName=endpoint_name
        )
        print(response)
    except:
        print(f"deleting {endpoint_name} failed")

def delete_model(sm_client, model_name, model_package_arn, model_package_group_name):
    try:
        response = sm_client.delete_model(
            ModelName=model_name
        )
        print(response)
    except:
        print(f"deleting {model_name} failed")

    try:
        response = sm_client.delete_model_package(
            ModelPackageName=model_package_arn
        )
        print(response)
    except:
        print(f"deleting {model_package_arn} failed")

    try:
        response = sm_client.delete_model_package_group(
            ModelPackageGroupName=model_package_group_name
        )
        print(response)
    except:
        print(f"deleting {model_package_group_name} failed")

def delete_fg(sm_client, feature_group_name):
    try:
        response = sm_client.delete_feature_group(
            FeatureGroupName=feature_group_name
        )
        print(response)
    except:
        print(f"deleting {feature_group_name} failed")

def delete_s3(bucket, prefix):
    try:
        bucket = s3.Bucket(bucket)
        bucket.objects.filter(Prefix=f"{prefix}/").delete()
        print(f"deleted s3://{bucket}/{prefix} successfully.")
    except:
        print(f"deleting s3://{bucket}/{prefix} failed")

def delete_ecr(image_name):
    try:
        response = ecr_client.delete_repository(
            repositoryName=image_name,
            force=True
        )
        print(response)
    except:
        print(f"deleting {image_name} failed")